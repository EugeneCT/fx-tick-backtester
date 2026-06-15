import dataclasses
from datetime import datetime, timezone

import pytest

from fxtrader.core.types import Bar, Fill, Order, OrderType, Position, Side, Tick


def _ts() -> datetime:
    return datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)


def test_tick_mid_and_spread():
    tick = Tick(symbol="EURUSD", ts=_ts(), bid=1.1000, ask=1.1002)
    assert tick.mid == 1.1001
    assert round(tick.spread, 5) == 0.0002


def test_tick_is_immutable():
    tick = Tick(symbol="EURUSD", ts=_ts(), bid=1.1, ask=1.1002)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tick.bid = 2.0  # type: ignore[misc]


def test_bar_fields():
    bar = Bar(symbol="EURUSD", ts=_ts(), open=1.1, high=1.2, low=1.05, close=1.15, volume=10.0)
    assert (bar.open, bar.high, bar.low, bar.close) == (1.1, 1.2, 1.05, 1.15)


def test_order_defaults_to_market():
    order = Order(symbol="EURUSD", side=Side.BUY, quantity=10_000)
    assert order.order_type is OrderType.MARKET
    assert order.limit_price is None


def test_fill_records_commission():
    fill = Fill(order_id="o1", symbol="EURUSD", side=Side.SELL,
                quantity=10_000, price=1.1001, ts=_ts(), commission=0.2)
    assert fill.commission == 0.2


def test_position_defaults_flat():
    pos = Position(symbol="EURUSD")
    assert pos.quantity == 0.0
    assert pos.avg_price == 0.0
