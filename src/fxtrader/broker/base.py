"""Broker interface shared by the backtest and live engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from fxtrader.core.types import Fill, Order, Tick

FillCallback = Callable[[Fill], None]
TickCallback = Callable[[Tick], None]


class Broker(ABC):
    """Receives market data, accepts orders, emits fills via a callback."""

    _on_fill: FillCallback | None = None
    _on_tick: TickCallback | None = None

    def set_fill_callback(self, callback: FillCallback) -> None:
        self._on_fill = callback

    def _emit_fill(self, fill: Fill) -> None:
        if self._on_fill is not None:
            self._on_fill(fill)

    def set_tick_callback(self, callback: TickCallback) -> None:
        self._on_tick = callback

    def _emit_tick(self, tick: Tick) -> None:
        if self._on_tick is not None:
            self._on_tick(tick)

    # --- Live lifecycle: safe no-op defaults; IBBroker overrides these. ---
    def connect(self) -> None:
        """Connect to the venue (no-op for in-memory brokers)."""

    def disconnect(self) -> None:
        """Disconnect from the venue (no-op for in-memory brokers)."""

    def cancel(self, order: Order) -> None:
        """Cancel a resting order (no-op for brokers that fill synchronously)."""

    def positions(self) -> dict[str, float]:
        """Venue-reported positions {symbol: signed_qty}. Empty by default."""
        return {}

    def account_value(self) -> float | None:
        """Venue-reported account value, or None if unavailable."""
        return None

    @abstractmethod
    def update_market(self, tick: Tick) -> None:
        """Inform the broker of the latest market for a symbol."""

    @abstractmethod
    def submit(self, order: Order) -> None:
        """Accept an order; fills are delivered asynchronously via the callback."""
