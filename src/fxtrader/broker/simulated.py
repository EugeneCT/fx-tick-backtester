"""SimulatedBroker — immediate market-order fills with slippage + commission."""

from __future__ import annotations

from fxtrader.broker.base import Broker
from fxtrader.core.types import Fill, Order, OrderType, Side, Tick


class SimulatedBroker(Broker):
    def __init__(
        self,
        *,
        commission_rate: float = 0.00002,  # 0.2 bps of notional
        min_commission: float = 2.0,
        slippage: float = 0.0,  # price units added to the touch
    ) -> None:
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self._market: dict[str, Tick] = {}
        self._next_id = 0

    def update_market(self, tick: Tick) -> None:
        self._market[tick.symbol] = tick

    def submit(self, order: Order) -> None:
        if order.order_type is not OrderType.MARKET:
            raise NotImplementedError("SimulatedBroker supports MARKET orders only (P2a)")
        tick = self._market[order.symbol]  # KeyError if no quote yet — intentional
        if order.side is Side.BUY:
            price = tick.ask + self.slippage
        else:
            price = tick.bid - self.slippage
        notional = order.quantity * price
        commission = max(self.min_commission, notional * self.commission_rate)
        self._next_id += 1
        self._emit_fill(
            Fill(
                order_id=order.order_id or f"sim-{self._next_id}",
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=price,
                ts=tick.ts,
                commission=commission,
            )
        )
