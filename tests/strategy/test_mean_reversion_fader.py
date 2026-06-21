"""Tests for MeanReversionFader — entry + ER regime gate + signal exit."""

from datetime import datetime, timedelta, timezone

import pytest

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Fill, Order, Side
from fxtrader.strategy.exits import FixedPips
from fxtrader.strategy.examples.mean_reversion_fader import MeanReversionFader

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _drive(strategy, closes, *, highs=None, lows=None, symbol="EURUSD"):
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
        clock.set_time(ts)
        hi = highs[i] if highs else c
        lo = lows[i] if lows else c
        _drive.price = c
        strategy.on_bar(Bar(symbol, ts, c, hi, lo, c, 1.0))
    return pf, orders


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        MeanReversionFader("EURUSD", lookback=1, entry_z=2, exit_z=0.5, er_threshold=0.3, size=10_000)
    with pytest.raises(ValueError):
        MeanReversionFader("EURUSD", lookback=5, entry_z=2, exit_z=2, er_threshold=0.3, size=10_000)
    with pytest.raises(ValueError):
        MeanReversionFader("EURUSD", lookback=5, entry_z=2, exit_z=0.5, er_threshold=0, size=10_000)


def test_no_trades_before_warmup():
    s = MeanReversionFader("EURUSD", lookback=5, entry_z=1.0, exit_z=0.3, er_threshold=0.5, size=10_000)
    _pf, orders = _drive(s, [1.0, 1.0, 1.0, 1.0])  # < lookback+1 bars
    assert orders == []


def test_fades_short_when_stretched_high_and_ranging():
    # Oscillating series (low ER, ranging) ending with a spike (high z > entry_z=1.0).
    # At trigger bar (index 6): z=1.9052, er=0.4286 < er_threshold=0.6 -> SHORT entry.
    s = MeanReversionFader("EURUSD", lookback=5, entry_z=1.0, exit_z=0.3, er_threshold=0.6, size=10_000)
    closes = [1.0000, 1.0010, 1.0000, 1.0010, 1.0000, 1.0010, 1.0040]
    pf, _ = _drive(s, closes)
    assert pf.position("EURUSD").quantity < 0   # short the stretched-high price


def test_no_entry_when_stretched_but_trending():
    # Monotonic ramp: ER=1.0 (trending) even though z=1.414 > entry_z=1.0
    # -> regime gate (er >= er_threshold=0.5) blocks entry.
    s = MeanReversionFader("EURUSD", lookback=5, entry_z=1.0, exit_z=0.3, er_threshold=0.5, size=10_000)
    closes = [1.0000, 1.0010, 1.0020, 1.0030, 1.0040, 1.0050, 1.0060]
    pf, _ = _drive(s, closes)
    assert pf.position("EURUSD").quantity == 0


def test_signal_exit_on_reversion_to_mean():
    # Ranging series, spike high (enter short), then price reverts to near the mean.
    # Entry bar (index 6): z=1.9052, er=0.4286 < er_threshold=0.9 -> SHORT.
    # Exit bar (index 7): close=1.0010, z=-0.2949, |z|=0.2949 <= exit_z=0.3 -> flat.
    s = MeanReversionFader("EURUSD", lookback=5, entry_z=1.0, exit_z=0.3, er_threshold=0.9, size=10_000)
    closes = [1.0000, 1.0010, 1.0000, 1.0010, 1.0000, 1.0010, 1.0040, 1.0010]
    pf, _ = _drive(s, closes)
    assert pf.position("EURUSD").quantity == 0
    assert s._last_exit_reason == "signal"


def test_ignores_other_symbols():
    s = MeanReversionFader("EURUSD", lookback=5, entry_z=1.0, exit_z=0.3, er_threshold=0.9, size=10_000)
    _pf, orders = _drive(s, [1.0, 1.001, 1.0, 1.001, 1.0, 1.001, 1.004], symbol="GBPUSD")
    assert orders == []
