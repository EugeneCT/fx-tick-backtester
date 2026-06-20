from datetime import datetime, timedelta, timezone

import pytest

from fxtrader.core.types import Bar, Fill, Side
from fxtrader.metrics.excursions import trade_excursions

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _fill(side, price, minute, qty=10_000):
    return Fill("x", "EURUSD", side, qty, price, T0 + timedelta(minutes=minute), 0.0)


def _bar(minute, hi, lo):
    ts = T0 + timedelta(minutes=minute)
    return Bar("EURUSD", ts, lo, hi, lo, lo, 1.0)


def test_long_trade_mae_mfe_in_pips():
    fills = [_fill(Side.BUY, 1.1000, 0), _fill(Side.SELL, 1.1010, 3)]
    bars = [_bar(0, 1.1000, 1.1000), _bar(1, 1.1005, 1.0990),  # adverse 10 pips, fav 5
            _bar(2, 1.1020, 1.1000), _bar(3, 1.1010, 1.1010)]  # fav 20 pips
    recs = trade_excursions(fills, bars, pip_size=0.0001)
    assert len(recs) == 1
    r = recs[0]
    assert r.direction == 1
    assert r.mae_pips == pytest.approx(10.0)   # 1.1000 - 1.0990
    assert r.mfe_pips == pytest.approx(20.0)   # 1.1020 - 1.1000
    assert r.pnl == pytest.approx((1.1010 - 1.1000) * 10_000)
