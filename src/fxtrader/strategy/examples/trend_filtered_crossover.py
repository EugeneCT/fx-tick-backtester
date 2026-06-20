"""Trend-filtered MA crossover: trade crosses only with the long trend; flat otherwise."""

from __future__ import annotations

from collections import deque

from fxtrader.core.types import Bar, Side
from fxtrader.strategy.base import Strategy
from fxtrader.strategy.exits import ExitPolicy, NoExit


class TrendFilteredCrossover(Strategy):
    def __init__(
        self,
        symbol: str,
        fast: int,
        slow: int,
        trend: int,
        size: float,
        exit_policy: ExitPolicy | None = None,
    ) -> None:
        if not (fast < slow < trend):
            raise ValueError("windows must satisfy fast < slow < trend")
        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self.trend = trend
        self.size = size
        self.exit_policy: ExitPolicy = exit_policy or NoExit()
        self._closes: deque[float] = deque(maxlen=trend)
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None
        self._bias = 0          # -1 / 0 / +1 from the last cross
        self._pos_dir = 0       # sign of the position we currently intend to hold
        self._entry_price: float | None = None
        self._stop_dist: float | None = None
        self._tp_dist: float | None = None
        self._last_exit_reason: str | None = None

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol != self.symbol:
            return
        self.exit_policy.update(bar)
        self._closes.append(bar.close)
        if len(self._closes) < self.trend:
            return

        closes = list(self._closes)
        fast = sum(closes[-self.fast:]) / self.fast
        slow = sum(closes[-self.slow:]) / self.slow
        trend_ma = sum(closes) / self.trend
        trend_dir = 1 if bar.close > trend_ma else (-1 if bar.close < trend_ma else 0)

        # --- exit check FIRST (against the open position), bar-granular ---
        position = self.portfolio.position(self.symbol).quantity
        if position != 0 and self._entry_price is not None:
            entry = self._entry_price
            hit = False
            if position > 0:  # long
                if self._stop_dist is not None and bar.low <= entry - self._stop_dist:
                    hit = True
                    self._last_exit_reason = "stop"
                elif self._tp_dist is not None and bar.high >= entry + self._tp_dist:
                    hit = True
                    self._last_exit_reason = "tp"
            else:  # short
                if self._stop_dist is not None and bar.high >= entry + self._stop_dist:
                    hit = True
                    self._last_exit_reason = "stop"
                elif self._tp_dist is not None and bar.low <= entry - self._tp_dist:
                    hit = True
                    self._last_exit_reason = "tp"
            if hit:
                side = Side.SELL if position > 0 else Side.BUY
                self.order(self.symbol, side, abs(position))
                self._bias = 0
                self._pos_dir = 0
                self._entry_price = None
                self._stop_dist = self._tp_dist = None
                return  # done with this bar

        if self._prev_fast is not None and self._prev_slow is not None:
            if self._prev_fast <= self._prev_slow and fast > slow:
                self._bias = 1
            elif self._prev_fast >= self._prev_slow and fast < slow:
                self._bias = -1
        self._prev_fast = fast
        self._prev_slow = slow

        desired = self._bias * self.size if (self._bias != 0 and self._bias == trend_dir) else 0
        self._reconcile(bar, desired)

    def _reconcile(self, bar: Bar, desired: float) -> None:
        position = self.portfolio.position(self.symbol).quantity
        delta = desired - position
        if delta != 0:
            side = Side.BUY if delta > 0 else Side.SELL
            self.order(self.symbol, side, abs(delta))
        new_dir = 0 if desired == 0 else (1 if desired > 0 else -1)
        if new_dir == 0:
            self._entry_price = None
            self._stop_dist = self._tp_dist = None
        elif new_dir != self._pos_dir:
            self._entry_price = bar.close
            self._stop_dist, self._tp_dist = self.exit_policy.distances()
        self._pos_dir = new_dir
