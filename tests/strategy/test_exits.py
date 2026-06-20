from datetime import datetime, timezone

import pytest

from fxtrader.core.types import Bar
from fxtrader.strategy.exits import AtrMultiple, FixedPips, NoExit


def _bar(o=1.0, h=1.0, l=1.0, c=1.0):
    return Bar("EURUSD", datetime(2025, 1, 1, tzinfo=timezone.utc), o, h, l, c, 1.0)


def test_no_exit_returns_none_none():
    p = NoExit()
    p.update(_bar())
    assert p.distances() == (None, None)


def test_fixed_pips_converts_pips_to_price_distance():
    p = FixedPips(stop_pips=10, tp_pips=20, pip_size=0.0001)
    assert p.distances() == (pytest.approx(0.0010), pytest.approx(0.0020))


def test_fixed_pips_zero_leg_is_none():
    p = FixedPips(stop_pips=10, tp_pips=0, pip_size=0.0001)
    assert p.distances() == (pytest.approx(0.0010), None)


def test_fixed_pips_rejects_negative():
    with pytest.raises(ValueError):
        FixedPips(stop_pips=-1, tp_pips=10, pip_size=0.0001)


def test_atr_multiple_none_until_warm():
    p = AtrMultiple(period=3, stop_mult=2.0, tp_mult=3.0)
    p.update(_bar(o=1.0, h=1.0010, l=0.9990, c=1.0))   # TR = 0.0020
    assert p.distances() == (None, None)               # not enough bars yet


def test_atr_multiple_scales_by_atr():
    p = AtrMultiple(period=2, stop_mult=2.0, tp_mult=3.0)
    # two bars each with true range 0.0020 (no gaps) -> ATR = 0.0020
    p.update(_bar(o=1.0, h=1.0010, l=0.9990, c=1.0))
    p.update(_bar(o=1.0, h=1.0010, l=0.9990, c=1.0))
    stop, tp = p.distances()
    assert stop == pytest.approx(0.0040)   # 2.0 * 0.0020
    assert tp == pytest.approx(0.0060)     # 3.0 * 0.0020


def test_atr_multiple_rejects_bad_params():
    with pytest.raises(ValueError):
        AtrMultiple(period=0, stop_mult=2.0, tp_mult=3.0)
    with pytest.raises(ValueError):
        AtrMultiple(period=3, stop_mult=-1.0, tp_mult=3.0)
