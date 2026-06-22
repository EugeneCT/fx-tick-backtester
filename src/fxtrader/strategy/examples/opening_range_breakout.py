"""Opening-Range Breakout: build the Asian-session range, break it during London (one trade/day)."""

from __future__ import annotations

from fxtrader.core.types import Bar, Side
from fxtrader.strategy.base import Strategy
from fxtrader.strategy.exits import ExitPolicy, NoExit


class OpeningRangeBreakout(Strategy):
    def __init__(
        self,
        symbol: str,
        range_start_hour: int,
        range_end_hour: int,
        entry_end_hour: int,
        exit_hour: int,
        size: float,
        exit_policy: ExitPolicy | None = None,
    ) -> None:
        hours = (range_start_hour, range_end_hour, entry_end_hour, exit_hour)
        if not all(0 <= h < 24 for h in hours):
            raise ValueError("hours must be in [0, 24)")
        if not (range_start_hour < range_end_hour <= entry_end_hour < exit_hour):
            raise ValueError("require range_start < range_end <= entry_end < exit_hour")
        self.symbol = symbol
        self.range_start_hour = range_start_hour
        self.range_end_hour = range_end_hour
        self.entry_end_hour = entry_end_hour
        self.exit_hour = exit_hour
        self.size = size
        self.exit_policy: ExitPolicy = exit_policy or NoExit()
        self._day = None
        self._hi: float | None = None
        self._lo: float | None = None
        self._traded = False
        self._pos_dir = 0
        self._entry_price: float | None = None
        self._stop_dist: float | None = None
        self._tp_dist: float | None = None
        self._last_exit_reason: str | None = None

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol != self.symbol:
            return
        self.exit_policy.update(bar)
        day = bar.ts.date()
        hour = bar.ts.hour
        if day != self._day:
            self._day = day
            self._hi = self._lo = None
            self._traded = False
        position = self.portfolio.position(self.symbol).quantity

        # 1. EXIT FIRST
        # (Task 2 inserts the stop/TP exit check HERE, before the time exit.)
        if position != 0 and hour >= self.exit_hour:
            self._flatten(position, "time")
            return

        # 2. RANGE BUILD (Asian window) — never trade here
        if self.range_start_hour <= hour < self.range_end_hour:
            self._hi = bar.high if self._hi is None else max(self._hi, bar.high)
            self._lo = bar.low if self._lo is None else min(self._lo, bar.low)
            return

        # 3. BREAKOUT ENTRY (one per day, London window)
        if (self._hi is not None and not self._traded and position == 0
                and self.range_end_hour <= hour < self.entry_end_hour):
            desired = 0.0
            if bar.close > self._hi:
                desired = self.size
            elif bar.close < self._lo:
                desired = -self.size
            if desired != 0.0:
                self._traded = True
                self._enter(bar, desired)

    def _enter(self, bar: Bar, desired: float) -> None:
        side = Side.BUY if desired > 0 else Side.SELL
        self.order(self.symbol, side, abs(desired))   # entered from flat -> qty == |desired|
        self._pos_dir = 1 if desired > 0 else -1
        self._entry_price = bar.close
        self._stop_dist, self._tp_dist = self.exit_policy.distances()

    def _flatten(self, position: float, reason: str) -> None:
        side = Side.SELL if position > 0 else Side.BUY
        self.order(self.symbol, side, abs(position))
        self._last_exit_reason = reason
        self._pos_dir = 0
        self._entry_price = None
        self._stop_dist = self._tp_dist = None
