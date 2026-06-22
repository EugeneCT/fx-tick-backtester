"""Market-neutral cross-pair spread reversion: fade z-score of log(primary)-log(secondary)."""

from __future__ import annotations

import math
import statistics
from collections import deque

from fxtrader.core.types import Bar, Side
from fxtrader.strategy.base import Strategy
from fxtrader.strategy.indicators import zscore


class PairsSpreadReversion(Strategy):
    def __init__(
        self,
        primary: str,
        secondary: str,
        lookback: int,
        entry_z: float,
        exit_z: float,
        stop_z: float,
        size: float,
    ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if not (0 <= exit_z < entry_z < stop_z):
            raise ValueError("require 0 <= exit_z < entry_z < stop_z")
        self.primary = primary
        self.secondary = secondary
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.size = size
        self._last: dict[str, float] = {}
        self._spreads: deque[float] = deque(maxlen=lookback)
        self._pos_dir = 0
        self._last_exit_reason: str | None = None

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol == self.secondary:
            self._last[self.secondary] = bar.close
            return
        if bar.symbol != self.primary:
            return
        self._last[self.primary] = bar.close
        if self.secondary not in self._last:
            return
        s = math.log(self._last[self.primary]) - math.log(self._last[self.secondary])
        self._spreads.append(s)
        if len(self._spreads) < self.lookback:
            return
        window = list(self._spreads)
        if statistics.pstdev(window) == 0.0:
            return
        z = zscore(window)

        pos = self.portfolio.position(self.primary).quantity
        if pos != 0:
            if abs(z) <= self.exit_z:
                self._close("signal")
                return
            if abs(z) >= self.stop_z:
                self._close("stop")
                return
        if pos == 0:
            if z > self.entry_z:
                self._open(-1)   # short the spread: SELL primary + BUY secondary
            elif z < -self.entry_z:
                self._open(1)    # long the spread:  BUY primary + SELL secondary

    def _open(self, direction: int) -> None:
        prim_side = Side.BUY if direction > 0 else Side.SELL
        sec_side = Side.SELL if direction > 0 else Side.BUY
        self.order(self.primary, prim_side, self.size)
        self.order(self.secondary, sec_side, self.size)
        self._pos_dir = direction

    def _close(self, reason: str) -> None:
        for sym in (self.primary, self.secondary):
            q = self.portfolio.position(sym).quantity
            if q != 0:
                side = Side.SELL if q > 0 else Side.BUY
                self.order(sym, side, abs(q))
        self._last_exit_reason = reason
        self._pos_dir = 0
