"""Strategy base class — pure logic, no I/O. The engine binds it to a context."""

from __future__ import annotations

from collections.abc import Callable

from fxtrader.core.clock import Clock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Fill, Order, OrderType, Side, Tick

OrderSink = Callable[[Order], None]


class Strategy:
    """Subclass and override the on_* hooks. Use self.order(...) to trade.

    The engine calls bind() before any hook. self.clock / self.portfolio are
    read-only views; never mutate the portfolio directly.
    """

    clock: Clock
    portfolio: Portfolio
    _submit: OrderSink

    def bind(self, clock: Clock, portfolio: Portfolio, submit: OrderSink) -> None:
        self.clock = clock
        self.portfolio = portfolio
        self._submit = submit

    def order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        *,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> None:
        self._submit(
            Order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
            )
        )

    # Lifecycle hooks — override as needed. Defaults are no-ops.
    def on_start(self) -> None: ...
    def on_tick(self, tick: Tick) -> None: ...
    def on_bar(self, bar: Bar) -> None: ...
    def on_fill(self, fill: Fill) -> None: ...
    def on_stop(self) -> None: ...
