# tests/strategy/test_trend_filtered_crossover.py
from datetime import datetime, timedelta, timezone

import pytest

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Fill, Order, Side
from fxtrader.strategy.examples.trend_filtered_crossover import TrendFilteredCrossover

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _drive(strategy, closes, *, highs=None, lows=None, symbol="EURUSD"):
    """Feed a close series as bars; fills execute immediately at bar.close into the portfolio."""
    pf = Portfolio(100_000.0)
    clock = SimClock()
    orders: list[Order] = []

    def sink(order: Order) -> None:
        orders.append(order)
        pf.apply_fill(Fill("x", order.symbol, order.side, order.quantity,
                           _drive.price, clock.now(), 0.0))

    strategy.bind(clock, pf, sink)
    strategy.on_start()
    for i, c in enumerate(closes):
        ts = T0 + timedelta(minutes=i)
        clock.set_time(ts)  # SimClock advance (verified API)
        hi = highs[i] if highs else c
        lo = lows[i] if lows else c
        _drive.price = c
        strategy.on_bar(Bar(symbol, ts, c, hi, lo, c, 1.0))
    return pf, orders


def test_rejects_unordered_windows():
    with pytest.raises(ValueError):
        TrendFilteredCrossover("EURUSD", fast=5, slow=5, trend=20, size=10_000)
    with pytest.raises(ValueError):
        TrendFilteredCrossover("EURUSD", fast=2, slow=20, trend=10, size=10_000)


def test_no_trades_before_warmup():
    s = TrendFilteredCrossover("EURUSD", fast=2, slow=3, trend=5, size=10_000)
    _pf, orders = _drive(s, [1.0, 1.1, 1.2, 1.3])  # < trend bars
    assert orders == []


def test_bullish_cross_in_uptrend_goes_long():
    # Dip-then-rise: fast falls below slow on the dip, then crosses back above while
    # price is above the trend MA (bullish trend gate satisfied).
    # fast=2, slow=4, trend=6: warmup needs 6 bars; cross fires at bar 6.
    s = TrendFilteredCrossover("EURUSD", fast=2, slow=4, trend=6, size=10_000)
    closes = [1.00, 1.00, 1.00, 0.95, 0.90, 0.92, 0.96, 1.00, 1.05, 1.10]
    pf, orders = _drive(s, closes)
    assert pf.position("EURUSD").quantity > 0


def test_bullish_cross_below_trend_stays_flat():
    # High start declining into downtrend; fast eventually crosses above slow during a
    # brief recovery, but price is still below the long-term trend MA → no long opened.
    s = TrendFilteredCrossover("EURUSD", fast=2, slow=4, trend=6, size=10_000)
    closes = [1.10, 1.10, 1.10, 1.05, 0.95, 0.90, 0.93, 0.97]
    pf, _orders = _drive(s, closes)
    assert pf.position("EURUSD").quantity == 0


def test_trend_flip_against_open_position_flattens():
    # Establish a long in an uptrend (same as test_bullish_cross_in_uptrend_goes_long),
    # then add one bar that drops close below the trend MA without triggering a new cross
    # (fast stays above slow on that bar) — the trend gate removes the position.
    s = TrendFilteredCrossover("EURUSD", fast=2, slow=4, trend=6, size=10_000)
    up = [1.00, 1.00, 1.00, 0.95, 0.90, 0.92, 0.96, 1.00, 1.05, 1.10]
    down = [0.96]  # close < trend_ma; fast still > slow so no new cross fires
    pf, _orders = _drive(s, up + down)
    assert pf.position("EURUSD").quantity == 0


def test_ignores_other_symbols():
    s = TrendFilteredCrossover("EURUSD", fast=2, slow=4, trend=6, size=10_000)
    _pf, orders = _drive(s, [1.00, 1.00, 1.00, 0.95, 0.90, 0.92, 0.96, 1.00, 1.05, 1.10],
                          symbol="GBPUSD")
    assert orders == []
