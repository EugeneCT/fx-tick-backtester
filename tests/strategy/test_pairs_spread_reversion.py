# tests/strategy/test_pairs_spread_reversion.py
from datetime import datetime, timedelta, timezone

import pytest

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Fill, Order
from fxtrader.strategy.examples.pairs_spread_reversion import PairsSpreadReversion

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _bar(symbol, i, close):
    ts = T0 + timedelta(minutes=i)
    return Bar(symbol, ts, close, close, close, close, 1.0)


def _drive_pairs(strategy, bars):
    """Feed interleaved primary/secondary Bars (time order); fills at bar.close per symbol."""
    pf = Portfolio(100_000.0)
    clock = SimClock()
    orders: list[Order] = []
    px: dict[str, float] = {}

    def sink(order: Order) -> None:
        orders.append(order)
        pf.apply_fill(Fill("x", order.symbol, order.side, order.quantity,
                           px[order.symbol], clock.now(), 0.0))

    strategy.bind(clock, pf, sink)
    strategy.on_start()
    for b in bars:
        clock.set_time(b.ts)
        px[b.symbol] = b.close
        strategy.on_bar(b)
    return pf, orders


def _seq(eur_closes, gbp=1.2500):
    """Interleave (GBP const, EUR varying) bars per period so each EUR bar samples a spread."""
    bars = []
    for i, e in enumerate(eur_closes):
        bars.append(_bar("GBPUSD", 2 * i, gbp))      # secondary first -> sets _last[GBP]
        bars.append(_bar("EURUSD", 2 * i + 1, e))    # primary -> samples spread, may trade
    return bars


def _x(**kw):
    base = dict(primary="EURUSD", secondary="GBPUSD", lookback=5,
                entry_z=1.0, exit_z=0.3, stop_z=3.0, size=10_000)
    base.update(kw)
    return PairsSpreadReversion(**base)


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        _x(lookback=1)
    with pytest.raises(ValueError):
        _x(entry_z=1.0, exit_z=1.0)        # exit_z !< entry_z
    with pytest.raises(ValueError):
        _x(entry_z=3.0, stop_z=3.0)        # entry_z !< stop_z


def test_no_trade_before_warmup():
    s = _x()
    pf, orders = _drive_pairs(s, _seq([1.10, 1.10, 1.10]))   # < lookback EUR bars
    assert orders == []


def test_secondary_bar_alone_does_not_trade():
    s = _x()
    pf, orders = _drive_pairs(s, [_bar("GBPUSD", 0, 1.25)])
    assert orders == []


def test_enter_short_spread_when_z_high():
    # EUR oscillates (ranging spread) then spikes UP -> spread z high -> SELL EUR + BUY GBP
    s = _x(lookback=5, entry_z=1.0)
    eur = [1.1000, 1.1010, 1.1000, 1.1010, 1.1000, 1.1010, 1.1100]
    pf, orders = _drive_pairs(s, _seq(eur))
    assert pf.position("EURUSD").quantity < 0     # short the rich leg
    assert pf.position("GBPUSD").quantity > 0     # long the cheap leg


def test_enter_long_spread_when_z_low():
    s = _x(lookback=5, entry_z=1.0)
    eur = [1.1000, 1.1010, 1.1000, 1.1010, 1.1000, 1.1010, 1.0900]   # spike DOWN
    pf, _ = _drive_pairs(s, _seq(eur))
    assert pf.position("EURUSD").quantity > 0
    assert pf.position("GBPUSD").quantity < 0


def test_signal_exit_on_reversion():
    s = _x(lookback=5, entry_z=1.0, exit_z=0.3, stop_z=5.0)
    # bar6 spike -> z≈1.99 -> enter short; bar7 revert -> z≈-0.22, |z|<=0.3 -> signal exit
    eur = [1.1000, 1.1010, 1.1000, 1.1010, 1.1000, 1.1010, 1.1100, 1.1020]
    pf, _ = _drive_pairs(s, _seq(eur))
    assert pf.position("EURUSD").quantity == 0
    assert pf.position("GBPUSD").quantity == 0
    assert s._last_exit_reason == "signal"


def test_divergence_stop():
    s = _x(lookback=5, entry_z=1.0, exit_z=0.3, stop_z=2.0)
    # bar6 spike -> z≈2.0 -> enter short; bars7-10 flat (|z|≈0.5>exit_z, no exit);
    # bar11 big spike -> z=2.0 exactly -> stop (entry bar has rolled out of window)
    eur = [1.1000, 1.1001, 1.1000, 1.1001, 1.1000, 1.1001, 1.1080,
           1.1000, 1.1000, 1.1000, 1.1000, 1.2000]
    pf, _ = _drive_pairs(s, _seq(eur))
    assert pf.position("EURUSD").quantity == 0
    assert s._last_exit_reason == "stop"


def test_ignores_third_symbol():
    s = _x()
    pf, orders = _drive_pairs(s, [_bar("USDJPY", i, 150.0) for i in range(10)])
    assert orders == []
