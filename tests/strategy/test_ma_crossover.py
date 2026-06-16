from datetime import datetime, timezone

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Side
from fxtrader.strategy.examples.ma_crossover import MaCrossover


def _bar(close, minute):
    ts = datetime(2024, 1, 2, 9, minute, tzinfo=timezone.utc)
    return Bar("EURUSD", ts, close, close, close, close, 1.0)


def _bound_strategy(fast=2, slow=4, size=10_000):
    s = MaCrossover(symbol="EURUSD", fast=fast, slow=slow, size=size)
    orders = []
    s.bind(SimClock(), Portfolio(100_000), orders.append)
    return s, orders


def test_no_order_before_slow_window_filled():
    s, orders = _bound_strategy()
    for i, c in enumerate([1.10, 1.11, 1.12]):  # only 3 bars, slow=4
        s.on_bar(_bar(c, i))
    assert orders == []


def test_golden_cross_goes_long():
    s, orders = _bound_strategy(fast=2, slow=4, size=10_000)
    # rising series so fast MA crosses above slow MA
    for i, c in enumerate([1.10, 1.10, 1.10, 1.10, 1.20, 1.30]):
        s.on_bar(_bar(c, i))
    assert orders, "expected at least one order"
    first = orders[0]
    assert first.side is Side.BUY
    assert first.quantity == 10_000


def test_ignores_other_symbol_bars():
    s, orders = _bound_strategy()
    other = Bar("GBPUSD", datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc),
                1.27, 1.27, 1.27, 1.27, 1.0)
    for _ in range(10):
        s.on_bar(other)
    assert orders == []
