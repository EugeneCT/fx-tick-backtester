from datetime import datetime, timezone

import pytest

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Fill, Order
from fxtrader.strategy.examples.opening_range_breakout import OpeningRangeBreakout


def _bar(hour, o, hi, lo, c, *, day=2, symbol="EURUSD"):
    ts = datetime(2025, 1, day, hour, 0, tzinfo=timezone.utc)
    return Bar(symbol, ts, o, hi, lo, c, 1.0)


def _drive_bars(strategy, bars):
    """Feed explicit Bars (chosen UTC ts); fills execute at bar.close into the portfolio."""
    pf = Portfolio(100_000.0)
    clock = SimClock()
    orders: list[Order] = []
    px = {"v": 0.0}

    def sink(order: Order) -> None:
        orders.append(order)
        pf.apply_fill(Fill("x", order.symbol, order.side, order.quantity, px["v"], clock.now(), 0.0))

    strategy.bind(clock, pf, sink)
    strategy.on_start()
    for b in bars:
        clock.set_time(b.ts)
        px["v"] = b.close
        strategy.on_bar(b)
    return pf, orders


def _orb(**kw):
    base = dict(symbol="EURUSD", range_start_hour=0, range_end_hour=7,
                entry_end_hour=16, exit_hour=21, size=10_000)
    base.update(kw)
    return OpeningRangeBreakout(**base)


def test_rejects_bad_hours():
    with pytest.raises(ValueError):
        _orb(range_start_hour=7, range_end_hour=7)         # range_start !< range_end
    with pytest.raises(ValueError):
        _orb(range_end_hour=17, entry_end_hour=16)         # range_end > entry_end
    with pytest.raises(ValueError):
        _orb(exit_hour=24)                                 # hour out of [0,24)


def test_no_trade_during_range_window():
    s = _orb()
    # bars only inside the Asian window 0..6
    pf, orders = _drive_bars(s, [_bar(h, 1.10, 1.10, 1.10, 1.10) for h in range(0, 7)])
    assert orders == []


def test_breakout_long_after_range():
    s = _orb()
    bars = [_bar(h, 1.10, 1.1010, 1.0990, 1.10) for h in range(0, 7)]  # range hi=1.1010 lo=1.0990
    bars.append(_bar(8, 1.10, 1.1030, 1.10, 1.1025))                    # close 1.1025 > hi -> LONG
    pf, _ = _drive_bars(s, bars)
    assert pf.position("EURUSD").quantity > 0


def test_breakout_short_after_range():
    s = _orb()
    bars = [_bar(h, 1.10, 1.1010, 1.0990, 1.10) for h in range(0, 7)]
    bars.append(_bar(8, 1.10, 1.10, 1.0970, 1.0975))                    # close 1.0975 < lo -> SHORT
    pf, _ = _drive_bars(s, bars)
    assert pf.position("EURUSD").quantity < 0


def test_one_trade_per_day():
    s = _orb()
    bars = [_bar(h, 1.10, 1.1010, 1.0990, 1.10) for h in range(0, 7)]
    bars.append(_bar(8, 1.10, 1.1030, 1.10, 1.1025))   # long entry
    bars.append(_bar(9, 1.10, 1.1040, 1.10, 1.1035))   # another break — must NOT add a 2nd entry
    pf, orders = _drive_bars(s, bars)
    assert len(orders) == 1


def test_no_entry_after_entry_window():
    s = _orb(entry_end_hour=9)
    bars = [_bar(h, 1.10, 1.1010, 1.0990, 1.10) for h in range(0, 7)]
    bars.append(_bar(10, 1.10, 1.1030, 1.10, 1.1025))  # hour 10 >= entry_end 9 -> no entry
    pf, orders = _drive_bars(s, bars)
    assert orders == []


def test_time_exit_flattens():
    s = _orb()
    bars = [_bar(h, 1.10, 1.1010, 1.0990, 1.10) for h in range(0, 7)]
    bars.append(_bar(8, 1.10, 1.1030, 1.10, 1.1025))   # long
    bars.append(_bar(21, 1.1025, 1.1025, 1.1025, 1.1025))  # hour 21 >= exit_hour -> flatten
    pf, _ = _drive_bars(s, bars)
    assert pf.position("EURUSD").quantity == 0
    assert s._last_exit_reason == "time"


def test_day_rollover_resets_and_allows_new_entry():
    s = _orb()
    day1 = [_bar(h, 1.10, 1.1010, 1.0990, 1.10) for h in range(0, 7)] + \
           [_bar(8, 1.10, 1.1030, 1.10, 1.1025), _bar(21, 1.1025, 1.1025, 1.1025, 1.1025)]
    day2 = [_bar(h, 1.10, 1.1010, 1.0990, 1.10, day=3) for h in range(0, 7)] + \
           [_bar(8, 1.10, 1.1030, 1.10, 1.1025, day=3)]
    pf, orders = _drive_bars(s, day1 + day2)
    # day1: entry + time-exit (2 orders); day2: a fresh entry (1 more)
    assert len(orders) == 3
    assert pf.position("EURUSD").quantity > 0


def test_ignores_other_symbols():
    s = _orb()
    bars = [_bar(h, 1.10, 1.1010, 1.0990, 1.10, symbol="GBPUSD") for h in range(0, 7)]
    bars.append(_bar(8, 1.10, 1.1030, 1.10, 1.1025, symbol="GBPUSD"))
    pf, orders = _drive_bars(s, bars)
    assert orders == []
