from datetime import datetime, timezone
from types import SimpleNamespace

from fxtrader.broker.ib import IBBroker
from fxtrader.core.types import Order, Side


class _Event:
    """Minimal stand-in for an ib_async eventkit Event (supports += / -=)."""
    def __init__(self):
        self.handlers = []
    def __iadd__(self, fn):
        self.handlers.append(fn)
        return self
    def __isub__(self, fn):
        if fn in self.handlers:
            self.handlers.remove(fn)
        return self


class FakeIB:
    def __init__(self):
        self.execDetailsEvent = _Event()
        self.pendingTickersEvent = _Event()
        self.placed = []
        self.cancelled = []
        self._positions = []
        self._connected = False

    def connect(self, host, port, clientId, **kw):
        self._connected = True

    def isConnected(self):
        return self._connected

    def disconnect(self):
        self._connected = False

    def reqMktData(self, contract, *a, **k):
        return SimpleNamespace(contract=contract)

    def placeOrder(self, contract, order):
        trade = SimpleNamespace(contract=contract, order=order)
        self.placed.append((contract, order))
        return trade

    def cancelOrder(self, order, *a, **k):
        self.cancelled.append(order)

    def positions(self, account=""):
        return self._positions


def _fx(symbol, currency):
    return SimpleNamespace(symbol=symbol, currency=currency)


def test_submit_market_order_places_ib_order():
    ib = FakeIB()
    b = IBBroker(ib=ib, symbols=["EURUSD"])
    b.submit(Order("EURUSD", Side.BUY, 10_000))
    assert len(ib.placed) == 1
    contract, order = ib.placed[0]
    assert order.action == "BUY"
    assert order.totalQuantity == 10_000


def test_exec_details_emits_core_fill():
    ib = FakeIB()
    b = IBBroker(ib=ib, symbols=["EURUSD"])
    fills = []
    b.set_fill_callback(fills.append)

    ts = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    execution = SimpleNamespace(orderId=42, side="BOT", shares=10_000, price=1.1005, time=ts)
    fill = SimpleNamespace(
        contract=_fx("EUR", "USD"),
        execution=execution,
        commissionReport=SimpleNamespace(commission=2.5),
    )
    b._on_exec_details(SimpleNamespace(), fill)

    assert len(fills) == 1
    f = fills[0]
    assert f.symbol == "EURUSD"
    assert f.side is Side.BUY
    assert f.quantity == 10_000
    assert f.price == 1.1005
    assert f.commission == 2.5
    assert f.ts == ts


def test_sell_execution_maps_to_sell_side():
    ib = FakeIB()
    b = IBBroker(ib=ib, symbols=["EURUSD"])
    fills = []
    b.set_fill_callback(fills.append)
    execution = SimpleNamespace(orderId=1, side="SLD", shares=5_000, price=1.10,
                                time=datetime(2024, 1, 2, tzinfo=timezone.utc))
    fill = SimpleNamespace(contract=_fx("EUR", "USD"), execution=execution, commissionReport=None)
    b._on_exec_details(SimpleNamespace(), fill)
    assert fills[0].side is Side.SELL
    assert fills[0].commission == 0.0


def test_pending_tickers_emits_core_tick():
    ib = FakeIB()
    b = IBBroker(ib=ib, symbols=["EURUSD"])
    ticks = []
    b.set_tick_callback(ticks.append)
    ts = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    ticker = SimpleNamespace(contract=_fx("EUR", "USD"), bid=1.1000, ask=1.1002,
                             bidSize=1.0, askSize=2.0, time=ts)
    b._on_pending_tickers([ticker])
    assert len(ticks) == 1
    assert ticks[0].symbol == "EURUSD"
    assert ticks[0].bid == 1.1000 and ticks[0].ask == 1.1002


def test_pending_tickers_skips_nan_quotes():
    ib = FakeIB()
    b = IBBroker(ib=ib, symbols=["EURUSD"])
    ticks = []
    b.set_tick_callback(ticks.append)
    nan = float("nan")
    ticker = SimpleNamespace(contract=_fx("EUR", "USD"), bid=nan, ask=nan,
                             bidSize=0.0, askSize=0.0, time=None)
    b._on_pending_tickers([ticker])
    assert ticks == []


def test_positions_maps_to_symbol_qty():
    ib = FakeIB()
    ib._positions = [SimpleNamespace(contract=_fx("EUR", "USD"), position=10_000.0)]
    b = IBBroker(ib=ib, symbols=["EURUSD"])
    assert b.positions() == {"EURUSD": 10_000.0}


def test_connect_wires_events_and_disconnect():
    ib = FakeIB()
    b = IBBroker(ib=ib, symbols=["EURUSD"])
    b.connect()
    assert ib.isConnected()
    assert b._on_exec_details in ib.execDetailsEvent.handlers
    assert b._on_pending_tickers in ib.pendingTickersEvent.handlers
    b.disconnect()
    assert not ib.isConnected()
