"""Moving-average crossover reference strategy (bar-based, single symbol)."""

from __future__ import annotations

from collections import deque

from fxtrader.core.types import Bar, Side
from fxtrader.strategy.base import Strategy


class MaCrossover(Strategy):
    def __init__(self, symbol: str, fast: int, slow: int, size: float) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self.size = size
        self._closes: deque[float] = deque(maxlen=slow)
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol != self.symbol:
            return
        self._closes.append(bar.close)
        if len(self._closes) < self.slow:
            return

        closes = list(self._closes)
        fast = sum(closes[-self.fast:]) / self.fast
        slow = sum(closes) / self.slow

        if self._prev_fast is not None and self._prev_slow is not None:
            crossed_up = self._prev_fast <= self._prev_slow and fast > slow
            crossed_down = self._prev_fast >= self._prev_slow and fast < slow
            signal = 1 if crossed_up else (-1 if crossed_down else 0)
            if signal != 0:
                target = signal * self.size
                position = self.portfolio.position(self.symbol).quantity
                delta = target - position
                if delta != 0:
                    side = Side.BUY if delta > 0 else Side.SELL
                    self.order(self.symbol, side, abs(delta))

        self._prev_fast = fast
        self._prev_slow = slow
