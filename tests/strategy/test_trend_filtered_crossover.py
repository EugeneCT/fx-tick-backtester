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


# ---------------------------------------------------------------------------
# Task 5: stop/TP exit tests
#
# Entry mechanics: fast=2, slow=4, trend=6.
# _UP_ENTRY (7 bars) is the minimal series that opens a long on the LAST bar:
#   bar 6 (close=0.96) — fill price = close = 0.96 = _ENTRY.
# After _UP_ENTRY the strategy holds long 10_000 with entry_price=0.96.
# The trigger bar is bar 7 (first bar after _UP_ENTRY).
#
# With stop_pips=20, tp_pips=200, pip_size=0.0001:
#   stop_dist = 0.0020  →  stop level = 0.9580
#   tp_dist   = 0.0200  →  TP level   = 0.9800
#
# With stop_pips=tp_pips=20:
#   stop_dist = tp_dist = 0.0020
#   stop level = 0.9580, tp level = 0.9620
# ---------------------------------------------------------------------------

from fxtrader.strategy.exits import FixedPips  # noqa: E402

# Minimal 7-bar series: warmup=6 bars then entry fires at bar-6 close=0.96.
_UP_ENTRY = [1.00, 1.00, 1.00, 0.95, 0.90, 0.92, 0.96]
_ENTRY = 0.96        # confirmed by tracing _drive bar-by-bar
_N = len(_UP_ENTRY)  # 7: index of trigger bar in highs/lows arrays


def test_long_stop_hit_flattens_and_disarms():
    """Trigger bar low (0.9570) <= stop level (0.9580): stop fires, position=0."""
    s = TrendFilteredCrossover(
        "EURUSD", fast=2, slow=4, trend=6, size=10_000,
        exit_policy=FixedPips(stop_pips=20, tp_pips=200, pip_size=0.0001),
        # stop_dist=0.0020, tp_dist=0.0200
        # stop level=0.9580, tp level=0.9800
    )
    # Trigger bar: low=0.9570 <= 0.9580 (stop hit); high=0.9790 < 0.9800 (no TP touch)
    trigger_highs = [None] * _N + [0.9790]
    trigger_lows  = [None] * _N + [0.9570]
    closes = _UP_ENTRY + [0.9700]
    pf, orders = _drive(s, closes, highs=trigger_highs, lows=trigger_lows)
    assert pf.position("EURUSD").quantity == 0, (
        f"Expected flat after stop; got {pf.position('EURUSD').quantity}"
    )
    # Two orders: initial BUY at entry, then SELL to flatten
    assert len(orders) == 2
    assert orders[-1].side == Side.SELL


def test_long_tp_hit_flattens():
    """Trigger bar high (0.9810) >= tp level (0.9800): TP fires, position=0.
    Stop is NOT touched: low=0.9700 > stop level 0.9580."""
    s = TrendFilteredCrossover(
        "EURUSD", fast=2, slow=4, trend=6, size=10_000,
        exit_policy=FixedPips(stop_pips=20, tp_pips=200, pip_size=0.0001),
    )
    trigger_highs = [None] * _N + [0.9810]  # 0.9810 >= 0.9800: TP touched
    trigger_lows  = [None] * _N + [0.9700]  # 0.9700 > 0.9580: no stop touch
    closes = _UP_ENTRY + [0.9750]
    pf, orders = _drive(s, closes, highs=trigger_highs, lows=trigger_lows)
    assert pf.position("EURUSD").quantity == 0, (
        f"Expected flat after TP; got {pf.position('EURUSD').quantity}"
    )
    assert len(orders) == 2
    assert orders[-1].side == Side.SELL


def test_same_bar_stop_and_tp_stop_wins():
    """Trigger bar touches BOTH stop (low<=0.9580) and TP (high>=0.9620); stop wins.
    stop_pips=tp_pips=20 → stop_dist=tp_dist=0.0020.
    Stop level=0.9580, TP level=0.9620.
    low=0.9570 <=0.9580 AND high=0.9630 >=0.9620 → both touched same bar → stop wins."""
    s = TrendFilteredCrossover(
        "EURUSD", fast=2, slow=4, trend=6, size=10_000,
        exit_policy=FixedPips(stop_pips=20, tp_pips=20, pip_size=0.0001),
    )
    trigger_highs = [None] * _N + [0.9630]  # 0.9630 >= 0.9620: TP touched
    trigger_lows  = [None] * _N + [0.9570]  # 0.9570 <= 0.9580: stop touched
    closes = _UP_ENTRY + [0.9600]
    pf, orders = _drive(s, closes, highs=trigger_highs, lows=trigger_lows)
    assert pf.position("EURUSD").quantity == 0, (
        f"Expected flat (stop wins same-bar); got {pf.position('EURUSD').quantity}"
    )
    assert len(orders) == 2
    assert orders[-1].side == Side.SELL


def test_no_exit_policy_no_stop_tp():
    """With NoExit (default), stop/TP distances are None → exit check never fires."""
    s = TrendFilteredCrossover("EURUSD", fast=2, slow=4, trend=6, size=10_000)
    # Wide-range trigger bar: no stop/TP since distances=None; position stays open.
    trigger_highs = [None] * _N + [1.2000]
    trigger_lows  = [None] * _N + [0.9000]
    closes = _UP_ENTRY + [0.9700]
    pf, _orders = _drive(s, closes, highs=trigger_highs, lows=trigger_lows)
    # No stop/TP exit — position still open (trend gate: close=0.97 > trend_ma still long)
    assert pf.position("EURUSD").quantity > 0
