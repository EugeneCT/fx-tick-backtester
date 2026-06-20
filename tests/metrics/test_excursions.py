from datetime import datetime, timedelta, timezone

import pytest

from fxtrader.core.types import Bar, Fill, Side
from fxtrader.metrics.excursions import TradeRecord, trade_excursions, summarize_excursions

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


def test_reversal_through_zero_pnl_and_excursions():
    """SELL 20k reversal fill must produce two TradeRecords with pnl scaled by
    closing=10k (not the fill quantity of 20k).

    Long leg:  BUY 10k @ 1.1000, reversed by SELL 20k @ 1.1015
               closing = min(20k, 10k) = 10k, direction = +1
               pnl = (1.1015 - 1.1000) * 10_000 * 1 = 15.0

    Short leg: opened at 1.1015, closed by BUY 10k @ 1.1005
               closing = min(10k, 10k) = 10k, direction = -1
               pnl = (1.1005 - 1.1015) * 10_000 * (-1) = 10.0

    Under the old _pnl helper the long leg would be scaled by exit_fill.quantity
    = 20_000, giving pnl = 30.0 (wrong).
    """
    p0, p1, p2 = 1.1000, 1.1015, 1.1005
    fills = [
        _fill(Side.BUY,  p0, 0, qty=10_000),   # open long 10k
        _fill(Side.SELL, p1, 5, qty=20_000),   # close long + open short (reversal)
        _fill(Side.BUY,  p2, 10, qty=10_000),  # close short
    ]
    # Bars for the long leg (minutes 0..5): dip to 1.0985, spike to 1.1020
    # Bars for the short leg (minutes 5..10): spike to 1.1025, dip to 1.0995
    bars = [
        _bar(0,  1.1000, 1.1000),
        _bar(1,  1.1010, 1.0985),  # long: adverse 15 pips (1.1000-1.0985), fav 10
        _bar(2,  1.1020, 1.0990),  # long: fav 20 pips (1.1020-1.1000)
        _bar(5,  1.1015, 1.1015),  # boundary (both legs share this ts)
        _bar(6,  1.1025, 1.1010),  # short: adverse 10 pips (1.1025-1.1015)
        _bar(8,  1.1018, 1.0995),  # short: fav 20 pips (1.1015-1.0995)
        _bar(10, 1.1005, 1.1005),
    ]
    recs = trade_excursions(fills, bars, pip_size=0.0001)
    assert len(recs) == 2

    long_rec = recs[0]
    short_rec = recs[1]

    # --- long leg ---
    assert long_rec.direction == 1
    expected_long_pnl = (p1 - p0) * 10_000 * 1   # = 15.0
    assert long_rec.pnl == pytest.approx(expected_long_pnl)
    # mae: min low in [0..5] vs entry 1.1000 → 1.1000-1.0985 = 15 pips
    assert long_rec.mae_pips == pytest.approx(15.0)
    # mfe: max high in [0..5] vs entry 1.1000 → 1.1020-1.1000 = 20 pips
    assert long_rec.mfe_pips == pytest.approx(20.0)

    # --- short leg ---
    assert short_rec.direction == -1
    expected_short_pnl = (p2 - p1) * 10_000 * (-1)  # = 10.0
    assert short_rec.pnl == pytest.approx(expected_short_pnl)
    # mae: max high in [5..10] vs entry 1.1015 → 1.1025-1.1015 = 10 pips
    assert short_rec.mae_pips == pytest.approx(10.0)
    # mfe: min low in [5..10] vs entry 1.1015 → 1.1015-1.0995 = 20 pips
    assert short_rec.mfe_pips == pytest.approx(20.0)


def test_short_round_trip_mae_mfe():
    """Clean SELL/BUY round trip: mirrored MAE/MFE relative to entry price."""
    fills = [_fill(Side.SELL, 1.2000, 0), _fill(Side.BUY, 1.1990, 4)]
    bars = [
        _bar(0, 1.2000, 1.2000),
        _bar(1, 1.2008, 1.1998),  # adverse 8 pips (1.2008-1.2000), fav 2
        _bar(2, 1.2005, 1.1985),  # fav 15 pips (1.2000-1.1985)
        _bar(4, 1.1990, 1.1990),
    ]
    recs = trade_excursions(fills, bars, pip_size=0.0001)
    assert len(recs) == 1
    r = recs[0]
    assert r.direction == -1
    # pnl = (exit - entry) * closing * direction = (1.1990 - 1.2000) * 10000 * (-1) = 10.0
    assert r.pnl == pytest.approx((1.1990 - 1.2000) * 10_000 * (-1))
    # mae: max high vs entry 1.2000 → 1.2008-1.2000 = 8 pips
    assert r.mae_pips == pytest.approx(8.0)
    # mfe: min low vs entry 1.2000 → 1.2000-1.1985 = 15 pips
    assert r.mfe_pips == pytest.approx(15.0)


def _rec(pnl, mae, mfe):
    return TradeRecord(T0, T0, 1, pnl, mae, mfe)


def test_summary_splits_winners_and_losers():
    recs = [_rec(5, 4, 30), _rec(8, 6, 40), _rec(-3, 25, 5), _rec(-9, 35, 3)]
    s = summarize_excursions(recs)
    assert s["n_trades"] == 4
    assert s["n_winners"] == 2 and s["n_losers"] == 2
    assert s["winners"]["mae_p50"] <= s["losers"]["mae_p50"]  # winners bleed less


def test_summary_empty_is_zeroed():
    s = summarize_excursions([])
    assert s["n_trades"] == 0
