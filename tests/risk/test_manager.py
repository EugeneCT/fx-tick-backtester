from datetime import datetime, timedelta, timezone

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Fill, Order, Side
from fxtrader.risk.manager import RiskLimits, RiskManager


def _clock(ts):
    c = SimClock()
    c.set_time(ts)
    return c


def _portfolio_long(qty, price=1.10):
    p = Portfolio(100_000)
    p.apply_fill(Fill("o", "EURUSD", Side.BUY, qty, price,
                      datetime(2024, 1, 2, tzinfo=timezone.utc)))
    return p


T0 = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)


def test_allows_order_within_limits():
    rm = RiskManager(RiskLimits(max_position=50_000))
    p = Portfolio(100_000)
    assert rm.check(Order("EURUSD", Side.BUY, 10_000), p, {"EURUSD": 1.10}, _clock(T0)) is True


def test_rejects_order_exceeding_max_position():
    rm = RiskManager(RiskLimits(max_position=15_000))
    p = _portfolio_long(10_000)
    # +10_000 already; another +10_000 -> 20_000 > 15_000
    assert rm.check(Order("EURUSD", Side.BUY, 10_000), p, {"EURUSD": 1.10}, _clock(T0)) is False


def test_rejects_order_exceeding_max_gross_notional():
    rm = RiskManager(RiskLimits(max_gross_notional=20_000))
    p = Portfolio(100_000)
    # 10_000 * 1.10 = 11_000 ok; 20_000 * 1.10 = 22_000 > 20_000
    assert rm.check(Order("EURUSD", Side.BUY, 10_000), p, {"EURUSD": 1.10}, _clock(T0)) is True
    assert rm.check(Order("EURUSD", Side.BUY, 20_000), p, {"EURUSD": 1.10}, _clock(T0)) is False


def test_rejects_when_orders_per_minute_exceeded():
    rm = RiskManager(RiskLimits(max_orders_per_minute=2))
    p = Portfolio(100_000)
    prices = {"EURUSD": 1.10}
    assert rm.check(Order("EURUSD", Side.BUY, 1_000), p, prices, _clock(T0)) is True
    assert rm.check(Order("EURUSD", Side.BUY, 1_000), p, prices, _clock(T0)) is True
    assert rm.check(Order("EURUSD", Side.BUY, 1_000), p, prices, _clock(T0)) is False
    # 61s later the window has cleared
    later = _clock(T0 + timedelta(seconds=61))
    assert rm.check(Order("EURUSD", Side.BUY, 1_000), p, prices, later) is True


def test_daily_loss_limit_trips_kill_switch():
    rm = RiskManager(RiskLimits(daily_loss_limit=500))
    c = _clock(T0)
    rm.update_equity(100_000, c)  # day start
    assert rm.halted is False
    rm.update_equity(99_600, c)  # -400, still ok
    assert rm.halted is False
    rm.update_equity(99_400, c)  # -600 -> breach
    assert rm.halted is True
    # halted rejects everything
    assert rm.check(Order("EURUSD", Side.SELL, 1_000), Portfolio(100_000), {"EURUSD": 1.1}, c) is False


def test_day_rollover_resets_day_start_equity():
    rm = RiskManager(RiskLimits(daily_loss_limit=500))
    rm.update_equity(100_000, _clock(T0))
    # next day, lower equity becomes the new baseline
    next_day = _clock(T0 + timedelta(days=1))
    rm.update_equity(99_000, next_day)
    assert rm.halted is False
    rm.update_equity(98_600, next_day)  # -400 from new baseline
    assert rm.halted is False


def test_kill_switch_rearms_next_day():
    # Daily-loss halt is per-session: it trips on the breached day, then resets
    # at the next day's rollover (a permanent halt would void a multi-year backtest).
    rm = RiskManager(RiskLimits(daily_loss_limit=500))
    c = _clock(T0)
    rm.update_equity(100_000, c)
    rm.update_equity(99_000, c)  # -1000 -> breach
    assert rm.halted is True

    next_day = _clock(T0 + timedelta(days=1))
    rm.update_equity(99_000, next_day)  # new session, new baseline -> re-armed
    assert rm.halted is False
    # ...and it can trip again on the new day
    rm.update_equity(98_400, next_day)  # -600 from the new baseline
    assert rm.halted is True
