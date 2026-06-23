"""Z-score mean-reversion fader, gated to ranging regimes by Kaufman Efficiency Ratio."""

from __future__ import annotations

import statistics
from collections import deque

from fxtrader.core.types import Bar, Side
from fxtrader.strategy.base import Strategy
from fxtrader.strategy.exits import ExitPolicy, NoExit
from fxtrader.strategy.indicators import efficiency_ratio, zscore


class MeanReversionFader(Strategy):
    def __init__(
        self,
        symbol: str,
        lookback: int,
        entry_z: float,
        exit_z: float,
        er_threshold: float,
        size: float,
        exit_policy: ExitPolicy | None = None,
    ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if not (0 <= exit_z < entry_z):
            raise ValueError("require 0 <= exit_z < entry_z")
        if not (0 < er_threshold <= 1):
            raise ValueError("er_threshold must be in (0, 1]")
        self.symbol = symbol
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.er_threshold = er_threshold
        self.size = size
        self.exit_policy: ExitPolicy = exit_policy or NoExit()
        self._closes: deque[float] = deque(maxlen=lookback + 1)
        self._pos_dir = 0
        self._entry_price: float | None = None
        self._stop_dist: float | None = None
        self._tp_dist: float | None = None
        self._last_exit_reason: str | None = None

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol != self.symbol:
            return
        self.exit_policy.update(bar)
        self._closes.append(bar.close)
        if len(self._closes) < self.lookback + 1:
            return
        closes = list(self._closes)
        window = closes[-self.lookback:]
        if statistics.pstdev(window) == 0.0:
            return
        z = zscore(window)
        er = efficiency_ratio(closes)

        position = self.portfolio.position(self.symbol).quantity

        # NOTE: the stop/TP ExitPolicy overlay is effectively a DEAD PATH for this strategy —
        # the signal exit (|z| <= exit_z) fires the bar after entry and pre-empts the brackets
        # every time (verified empirically: research/findings-2026-06-21.md). Tuning stop/TP here
        # cannot change behavior; the no-exit baseline is the strategy's ceiling.
        # stop/TP overlay (risk-first), bar-granular, stop-first same-bar rule
        if position != 0 and self._entry_price is not None:
            entry = self._entry_price
            reason = None
            if position > 0:  # long
                if self._stop_dist is not None and bar.low <= entry - self._stop_dist:
                    reason = "stop"
                elif self._tp_dist is not None and bar.high >= entry + self._tp_dist:
                    reason = "tp"
            else:  # short
                if self._stop_dist is not None and bar.high >= entry + self._stop_dist:
                    reason = "stop"
                elif self._tp_dist is not None and bar.low <= entry - self._tp_dist:
                    reason = "tp"
            if reason:
                self._flatten(position, reason)
                return

        # signal exit: reverted to the mean
        if position != 0 and abs(z) <= self.exit_z:
            self._flatten(position, "signal")
            return

        # entry: only when flat AND ranging
        desired = 0.0
        if position == 0 and er < self.er_threshold:
            if z > self.entry_z:
                desired = -self.size
            elif z < -self.entry_z:
                desired = self.size
        self._reconcile(bar, desired)

    def _flatten(self, position: float, reason: str) -> None:
        side = Side.SELL if position > 0 else Side.BUY
        self.order(self.symbol, side, abs(position))
        self._last_exit_reason = reason
        self._pos_dir = 0
        self._entry_price = None
        self._stop_dist = self._tp_dist = None

    def _reconcile(self, bar: Bar, desired: float) -> None:
        position = self.portfolio.position(self.symbol).quantity
        delta = desired - position
        if delta != 0:
            side = Side.BUY if delta > 0 else Side.SELL
            self.order(self.symbol, side, abs(delta))
        new_dir = 0 if desired == 0 else (1 if desired > 0 else -1)
        if new_dir != 0 and new_dir != self._pos_dir:
            self._entry_price = bar.close
            self._stop_dist, self._tp_dist = self.exit_policy.distances()
        self._pos_dir = new_dir
