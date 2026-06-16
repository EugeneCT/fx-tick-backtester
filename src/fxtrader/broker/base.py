"""Broker interface shared by the backtest and live engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from fxtrader.core.types import Fill, Order, Tick

FillCallback = Callable[[Fill], None]


class Broker(ABC):
    """Receives market data, accepts orders, emits fills via a callback."""

    _on_fill: FillCallback | None = None

    def set_fill_callback(self, callback: FillCallback) -> None:
        self._on_fill = callback

    def _emit_fill(self, fill: Fill) -> None:
        if self._on_fill is not None:
            self._on_fill(fill)

    @abstractmethod
    def update_market(self, tick: Tick) -> None:
        """Inform the broker of the latest market for a symbol."""

    @abstractmethod
    def submit(self, order: Order) -> None:
        """Accept an order; fills are delivered asynchronously via the callback."""
