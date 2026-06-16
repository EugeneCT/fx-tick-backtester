"""IBBroker — live execution via Interactive Brokers (ib_async).

The ib_async IB client is injected so every translation path is unit-testable with a
fake. Real connection is exercised by the opt-in integration smoke test.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from ib_async import Forex, LimitOrder, MarketOrder

from fxtrader.broker.base import Broker
from fxtrader.core.types import Fill, Order, OrderType, Side, Tick


class IBBroker(Broker):
    def __init__(
        self,
        ib=None,
        *,
        host: str = "127.0.0.1",
        port: int = 7497,  # 7497 = TWS paper, 4002 = Gateway paper
        client_id: int = 1,
        symbols: list[str] | None = None,
    ) -> None:
        self._ib = ib
        self.host = host
        self.port = port
        self.client_id = client_id
        self.symbols = symbols or []
        self._last_tick: dict[str, Tick] = {}

    def _client(self):
        if self._ib is None:
            from ib_async import IB
            self._ib = IB()
        return self._ib

    def connect(self) -> None:
        ib = self._client()
        ib.connect(self.host, self.port, clientId=self.client_id)
        ib.execDetailsEvent += self._on_exec_details
        ib.pendingTickersEvent += self._on_pending_tickers
        for symbol in self.symbols:
            ib.reqMktData(self._contract(symbol))

    def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    @staticmethod
    def _contract(symbol: str):
        return Forex(symbol)

    @staticmethod
    def _symbol_of(contract) -> str:
        return f"{contract.symbol}{contract.currency}"

    def submit(self, order: Order) -> None:
        action = "BUY" if order.side is Side.BUY else "SELL"
        if order.order_type is OrderType.MARKET:
            ib_order = MarketOrder(action, order.quantity)
        elif order.order_type is OrderType.LIMIT:
            ib_order = LimitOrder(action, order.quantity, order.limit_price)
        else:
            raise NotImplementedError(f"IBBroker does not support {order.order_type}")
        self._client().placeOrder(self._contract(order.symbol), ib_order)

    def cancel(self, order: Order) -> None:
        # Cancel any open IB orders for the symbol (best-effort; live order tracking is future work).
        ib = self._client()
        for trade in ib.openTrades():
            if self._symbol_of(trade.contract) == order.symbol:
                ib.cancelOrder(trade.order)

    def update_market(self, tick: Tick) -> None:
        self._last_tick[tick.symbol] = tick

    def positions(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for pos in self._client().positions():
            out[self._symbol_of(pos.contract)] = pos.position
        return out

    def account_value(self) -> float | None:
        for value in self._client().accountValues():
            if value.tag == "NetLiquidation":
                return float(value.value)
        return None

    def _on_exec_details(self, trade, fill) -> None:
        execution = fill.execution
        side = Side.BUY if execution.side == "BOT" else Side.SELL
        commission = 0.0
        report = getattr(fill, "commissionReport", None)
        if report is not None and getattr(report, "commission", None) is not None:
            commission = float(report.commission)
        self._emit_fill(
            Fill(
                order_id=str(execution.orderId),
                symbol=self._symbol_of(fill.contract),
                side=side,
                quantity=float(execution.shares),
                price=float(execution.price),
                ts=execution.time,
                commission=commission,
            )
        )

    def _on_pending_tickers(self, tickers) -> None:
        for ticker in tickers:
            bid, ask = ticker.bid, ticker.ask
            if bid is None or ask is None or math.isnan(bid) or math.isnan(ask):
                continue
            tick = Tick(
                symbol=self._symbol_of(ticker.contract),
                ts=ticker.time or datetime.now(timezone.utc),
                bid=float(bid),
                ask=float(ask),
                bid_volume=float(ticker.bidSize or 0.0),
                ask_volume=float(ticker.askSize or 0.0),
            )
            self._last_tick[tick.symbol] = tick
            self._emit_tick(tick)
