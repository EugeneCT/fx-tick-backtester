import pytest

from fxtrader.strategy.indicators import efficiency_ratio, zscore


def test_zscore_known_series():
    # mean=3, pstdev=sqrt(2)~1.41421, last=5 -> z = 2/1.41421
    assert zscore([1, 2, 3, 4, 5]) == pytest.approx(2 / (2 ** 0.5))


def test_zscore_flat_window_is_zero():
    assert zscore([2.0, 2.0, 2.0]) == 0.0


def test_efficiency_ratio_monotonic_is_one():
    assert efficiency_ratio([1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_efficiency_ratio_oscillating_is_zero():
    # net move 0 over the window, lots of gross movement
    assert efficiency_ratio([1, 2, 1, 2, 1]) == pytest.approx(0.0)


def test_efficiency_ratio_flat_is_zero():
    assert efficiency_ratio([1.0, 1.0, 1.0]) == 0.0
