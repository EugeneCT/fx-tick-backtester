# src/fxtrader/strategy/examples/price_action.py
"""Trend-gated price-action strategy: fire a famous PA pattern only with the long-MA trend."""

from __future__ import annotations

from collections import deque

from fxtrader.core.types import Bar, Side
from fxtrader.strategy.base import Strategy
from fxtrader.strategy.exits import ExitPolicy, NoExit
from fxtrader.strategy.patterns import engulfing, is_inside_bar, pin_bar, swing_break

_PATTERNS = ("pin", "engulf", "inside", "swingbreak")


class PriceActionTrend(Strategy):
    def __init__(
        self,
        symbol: str,
        pattern: str,
        trend: int,
        size: float,
        wick_ratio: float = 2.0,
        body_frac: float = 0.34,
        swing_lookback: int = 20,
        inside_break_bars: int = 3,
        max_hold_bars: int = 0,
        exit_policy: ExitPolicy | None = None,
    ) -> None:
        if pattern not in _PATTERNS:
            raise ValueError(f"pattern must be one of {_PATTERNS}")
        if trend < 2:
            raise ValueError("trend window must be >= 2")
        if swing_lookback < 1 or inside_break_bars < 1 or max_hold_bars < 0:
            raise ValueError("swing_lookback/inside_break_bars >= 1, max_hold_bars >= 0")
        self.symbol = symbol
        self.pattern = pattern
        self.trend = trend
        self.size = size
        self.wick_ratio = wick_ratio
        self.body_frac = body_frac
        self.swing_lookback = swing_lookback
        self.inside_break_bars = inside_break_bars
        self.max_hold_bars = max_hold_bars
        self.exit_policy: ExitPolicy = exit_policy or NoExit()
        self._closes: deque[float] = deque(maxlen=trend)
        self._bars: deque[Bar] = deque(maxlen=max(swing_lookback + 1, 2))
        self._entry_price: float | None = None
        self._stop_dist: float | None = None
        self._tp_dist: float | None = None
        self._pos_dir = 0
        self._hold = 0
        self._mother: tuple[float, float, int] | None = None  # (hi, lo, bars_left)
        self._last_exit_reason: str | None = None

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol != self.symbol:
            return
        self.exit_policy.update(bar)
        self._closes.append(bar.close)
        self._bars.append(bar)

        # 1. EXIT FIRST — stop, then TP (stop-first same-bar), then time.
        position = self.portfolio.position(self.symbol).quantity
        if position != 0 and self._entry_price is not None:
            entry = self._entry_price
            reason: str | None = None
            if position > 0:
                if self._stop_dist is not None and bar.low <= entry - self._stop_dist:
                    reason = "stop"
                elif self._tp_dist is not None and bar.high >= entry + self._tp_dist:
                    reason = "tp"
            else:
                if self._stop_dist is not None and bar.high >= entry + self._stop_dist:
                    reason = "stop"
                elif self._tp_dist is not None and bar.low <= entry - self._tp_dist:
                    reason = "tp"
            if reason is None and self.max_hold_bars > 0:
                self._hold += 1
                if self._hold >= self.max_hold_bars:
                    reason = "time"
            if reason:
                self._flatten(position, reason)
                return

        if len(self._closes) < self.trend:
            return

        trend_ma = sum(self._closes) / len(self._closes)
        trend_dir = 1 if bar.close > trend_ma else (-1 if bar.close < trend_ma else 0)
        if trend_dir == 0:
            return

        # 2. ENTRY — only when flat and the pattern agrees with the trend.
        if self.portfolio.position(self.symbol).quantity != 0:
            return
        signal = self._detect(bar)
        if signal != 0 and signal == trend_dir:
            self._enter(bar, signal * self.size)

    def _detect(self, bar: Bar) -> int:
        if self.pattern == "pin":
            return pin_bar(bar, wick_ratio=self.wick_ratio, body_frac=self.body_frac)
        if self.pattern == "engulf":
            if len(self._bars) < 2:
                return 0
            return engulfing(self._bars[-2], bar)
        if self.pattern == "swingbreak":
            return swing_break(list(self._bars), self.swing_lookback)
        if self.pattern == "inside":
            return self._inside_break(bar)
        return 0

    def _inside_break(self, bar: Bar) -> int:
        sig = 0
        if self._mother is not None:
            hi, lo, left = self._mother
            if bar.close > hi:
                sig, self._mother = 1, None
            elif bar.close < lo:
                sig, self._mother = -1, None
            else:
                left -= 1
                self._mother = (hi, lo, left) if left > 0 else None
        if self._mother is None and len(self._bars) >= 2:
            prev = self._bars[-2]
            if is_inside_bar(prev, bar):
                self._mother = (prev.high, prev.low, self.inside_break_bars)
        return sig

    def _enter(self, bar: Bar, desired: float) -> None:
        side = Side.BUY if desired > 0 else Side.SELL
        self.order(self.symbol, side, abs(desired))
        self._pos_dir = 1 if desired > 0 else -1
        self._entry_price = bar.close
        self._stop_dist, self._tp_dist = self.exit_policy.distances()
        self._hold = 0

    def _flatten(self, position: float, reason: str) -> None:
        side = Side.SELL if position > 0 else Side.BUY
        self.order(self.symbol, side, abs(position))
        self._last_exit_reason = reason
        self._pos_dir = 0
        self._entry_price = None
        self._stop_dist = self._tp_dist = None
        self._hold = 0
