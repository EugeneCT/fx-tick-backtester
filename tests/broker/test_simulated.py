from datetime import datetime, timezone

import pytest

from fxtrader.broker.simulated import SimulatedBroker
from fxtrader.core.types import Order, OrderType, Side, Tick


def _tick(bid, ask, minute=0):
    return Tick("EURUSD", datetime(2024, 1, 2, 9, minute, tzinfo=timezone.utc), bid, ask)


def test_buy_fills_at_ask_plus_slippage():
    b = SimulatedBroker(commission_rate=0.0, min_commission=0.0, slippage=0.00001)
    fills = []
    b.set_fill_callback(fills.append)
    b.update_market(_tick(1.10000, 1.10020))
    b.submit(Order("EURUSD", Side.BUY, 10_000))
    assert len(fills) == 1
    assert round(fills[0].price, 5) == round(1.10020 + 0.00001, 5)
    assert fills[0].quantity == 10_000
    assert fills[0].ts == datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)


def test_sell_fills_at_bid_minus_slippage():
    b = SimulatedBroker(commission_rate=0.0, min_commission=0.0, slippage=0.00001)
    fills = []
    b.set_fill_callback(fills.append)
    b.update_market(_tick(1.10000, 1.10020))
    b.submit(Order("EURUSD", Side.SELL, 10_000))
    assert round(fills[0].price, 5) == round(1.10000 - 0.00001, 5)


def test_commission_is_bps_of_notional_with_floor():
    b = SimulatedBroker(commission_rate=0.00002, min_commission=2.0)
    fills = []
    b.set_fill_callback(fills.append)
    b.update_market(_tick(1.10000, 1.10000))
    # notional = 1_000_000 * 1.10 = 1_100_000; comm = 1_100_000 * 0.00002 = 22.0 (> floor 2.0)
    b.submit(Order("EURUSD", Side.BUY, 1_000_000))
    assert round(fills[0].commission, 4) == 22.0
    # small order hits the floor: notional = 1000*1.1=1100; *0.00002=0.022 -> floor 2.0
    b.submit(Order("EURUSD", Side.BUY, 1_000))
    assert fills[1].commission == 2.0


def test_market_order_without_quote_raises():
    b = SimulatedBroker()
    b.set_fill_callback(lambda f: None)
    with pytest.raises(KeyError):
        b.submit(Order("EURUSD", Side.BUY, 10_000))


def test_limit_order_not_supported():
    b = SimulatedBroker()
    b.set_fill_callback(lambda f: None)
    b.update_market(_tick(1.1, 1.1002))
    with pytest.raises(NotImplementedError):
        b.submit(Order("EURUSD", Side.BUY, 10_000, order_type=OrderType.LIMIT, limit_price=1.1))
