# tests/strategy/test_price_action.py
from datetime import datetime, timedelta, timezone

import pytest

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Fill, Order, Side
from fxtrader.strategy.examples.price_action import PriceActionTrend
from fxtrader.strategy.exits import FixedPips

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _drive(strategy, ohlc, *, symbol="EURUSD"):
    """Feed (open, high, low, close) bars; fills execute at bar.close into the portfolio."""
    pf = Portfolio(100_000.0)
    clock = SimClock()
    orders: list[Order] = []

    def sink(order: Order) -> None:
        orders.append(order)
        pf.apply_fill(Fill("x", order.symbol, order.side, order.quantity,
                           _drive.price, clock.now(), 0.0))

    strategy.bind(clock, pf, sink)
    strategy.on_start()
    for i, (o, h, l, c) in enumerate(ohlc):
        ts = T0 + timedelta(minutes=i)
        clock.set_time(ts)
        _drive.price = c
        strategy.on_bar(Bar(symbol, ts, o, h, l, c, 1.0))
    return pf, orders


# Uptrend warmup (4 bars, rising closes) then a bullish PIN on the 5th bar.
_UPTREND_PIN = [
    (1.0000, 1.0010, 0.9990, 1.0000),
    (1.0000, 1.0020, 1.0000, 1.0010),
    (1.0010, 1.0030, 1.0010, 1.0020),
    (1.0020, 1.0040, 1.0020, 1.0030),
    (1.0035, 1.0040, 0.9990, 1.0038),  # bullish pin; close 1.0038 > trend MA → long
]


def test_rejects_bad_pattern():
    with pytest.raises(ValueError):
        PriceActionTrend("EURUSD", pattern="nope", trend=4, size=10_000)


def test_no_trades_before_warmup():
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000)
    _pf, orders = _drive(s, _UPTREND_PIN[:3])  # < trend bars
    assert orders == []


def test_pin_in_uptrend_goes_long():
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000)
    pf, orders = _drive(s, _UPTREND_PIN)
    assert pf.position("EURUSD").quantity > 0
    assert len(orders) == 1 and orders[0].side == Side.BUY


def test_bullish_pin_against_downtrend_blocked():
    # Falling closes → trend down; a bullish pin must be suppressed by the gate.
    series = [
        (1.0030, 1.0040, 1.0020, 1.0030),
        (1.0020, 1.0030, 1.0010, 1.0020),
        (1.0010, 1.0020, 1.0000, 1.0010),
        (1.0000, 1.0010, 0.9990, 1.0000),
        (0.9995, 1.0000, 0.9950, 0.9998),  # bullish pin, but close < trend MA → blocked
    ]
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000)
    pf, orders = _drive(s, series)
    assert pf.position("EURUSD").quantity == 0
    assert orders == []


def test_exit_first_stop_flattens():
    # Enter long on bar 5 at close 1.0038; FixedPips stop_dist=0.0020 → stop level 1.0018.
    s = PriceActionTrend(
        "EURUSD", pattern="pin", trend=4, size=10_000,
        exit_policy=FixedPips(stop_pips=20, tp_pips=200, pip_size=0.0001),
    )
    stop_bar = (1.0030, 1.0035, 1.0010, 1.0020)  # low 1.0010 <= 1.0018 → stop
    pf, orders = _drive(s, _UPTREND_PIN + [stop_bar])
    assert pf.position("EURUSD").quantity == 0
    assert orders[-1].side == Side.SELL
    assert s._last_exit_reason == "stop"


def test_max_hold_bars_flattens():
    # No exit policy; max_hold_bars=2 → time exit two bars after entry.
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000, max_hold_bars=2)
    hold1 = (1.0038, 1.0045, 1.0030, 1.0040)
    hold2 = (1.0040, 1.0048, 1.0035, 1.0044)  # 2nd bar held → time exit
    pf, orders = _drive(s, _UPTREND_PIN + [hold1, hold2])
    assert pf.position("EURUSD").quantity == 0
    assert s._last_exit_reason == "time"


def test_ignores_other_symbols():
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000)
    _pf, orders = _drive(s, _UPTREND_PIN, symbol="GBPUSD")
    assert orders == []
