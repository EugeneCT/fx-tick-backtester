from datetime import datetime, timezone

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import OrderType, Side, Tick
from fxtrader.strategy.base import Strategy


class _Spy(Strategy):
    def __init__(self):
        self.ticks = []
        self.started = False

    def on_start(self):
        self.started = True

    def on_tick(self, tick):
        self.ticks.append(tick)
        self.order(tick.symbol, Side.BUY, 10_000)


def _bind(strategy):
    clock = SimClock()
    portfolio = Portfolio(100_000)
    sent = []
    strategy.bind(clock, portfolio, sent.append)
    return clock, portfolio, sent


def test_bind_exposes_clock_and_portfolio():
    s = _Spy()
    clock, portfolio, _ = _bind(s)
    assert s.clock is clock
    assert s.portfolio is portfolio


def test_order_helper_builds_and_submits_order():
    s = _Spy()
    _, _, sent = _bind(s)
    tick = Tick("EURUSD", datetime(2024, 1, 2, tzinfo=timezone.utc), 1.1, 1.1002)
    s.on_tick(tick)
    assert len(sent) == 1
    order = sent[0]
    assert order.symbol == "EURUSD"
    assert order.side is Side.BUY
    assert order.quantity == 10_000
    assert order.order_type is OrderType.MARKET


def test_default_hooks_are_noops():
    s = Strategy()
    _bind(s)
    # None of these should raise
    s.on_start()
    s.on_bar(None)
    s.on_fill(None)
    s.on_stop()


def test_on_start_hook_runs():
    s = _Spy()
    _bind(s)
    s.on_start()
    assert s.started is True
