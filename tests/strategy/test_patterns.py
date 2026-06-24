from datetime import datetime, timezone

from fxtrader.core.types import Bar
from fxtrader.strategy.patterns import pin_bar, engulfing, is_inside_bar, swing_break

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _bar(o, h, l, c):
    return Bar("EURUSD", T0, o, h, l, c, 1.0)


# --- pin_bar ---
def test_bullish_pin_long_lower_wick():
    # rng=0.01, tiny body near the top, long lower wick → +1
    b = _bar(1.0090, 1.0100, 1.0000, 1.0095)
    assert pin_bar(b, wick_ratio=2.0, body_frac=0.34) == 1


def test_bearish_pin_long_upper_wick():
    b = _bar(1.0010, 1.0100, 1.0000, 1.0005)
    assert pin_bar(b, wick_ratio=2.0, body_frac=0.34) == -1


def test_big_body_is_not_a_pin():
    b = _bar(1.0000, 1.0100, 0.9990, 1.0090)  # body dominates the range
    assert pin_bar(b, wick_ratio=2.0, body_frac=0.34) == 0


def test_zero_range_bar_is_not_a_pin():
    b = _bar(1.0000, 1.0000, 1.0000, 1.0000)
    assert pin_bar(b, wick_ratio=2.0, body_frac=0.34) == 0


# --- engulfing ---
def test_bullish_engulfing():
    prev = _bar(1.0050, 1.0055, 0.9995, 1.0000)  # bearish
    cur = _bar(0.9990, 1.0065, 0.9985, 1.0060)   # bullish, body engulfs prev body
    assert engulfing(prev, cur) == 1


def test_bearish_engulfing():
    prev = _bar(1.0000, 1.0055, 0.9995, 1.0050)  # bullish
    cur = _bar(1.0060, 1.0065, 0.9985, 0.9990)   # bearish, body engulfs prev body
    assert engulfing(prev, cur) == -1


def test_non_engulfing_returns_zero():
    prev = _bar(1.0050, 1.0055, 0.9995, 1.0000)  # bearish
    cur = _bar(1.0010, 1.0045, 1.0005, 1.0040)   # bullish but does not exceed prev.open
    assert engulfing(prev, cur) == 0


# --- is_inside_bar ---
def test_inside_bar_true():
    prev = _bar(1.0000, 1.0100, 1.0000, 1.0050)
    cur = _bar(1.0030, 1.0080, 1.0020, 1.0040)   # contained within prev range
    assert is_inside_bar(prev, cur) is True


def test_inside_bar_false_when_breaks_high():
    prev = _bar(1.0000, 1.0100, 1.0000, 1.0050)
    cur = _bar(1.0030, 1.0120, 1.0020, 1.0040)   # high exceeds prev high
    assert is_inside_bar(prev, cur) is False


# --- swing_break ---
def test_swing_break_up():
    bars = [_bar(1.00, 1.01, 1.00, 1.005) for _ in range(3)]
    bars.append(_bar(1.01, 1.025, 1.01, 1.02))   # close above swing high (1.01)
    assert swing_break(bars, lookback=3) == 1


def test_swing_break_down():
    bars = [_bar(1.00, 1.01, 1.00, 1.005) for _ in range(3)]
    bars.append(_bar(1.00, 1.00, 0.985, 0.99))   # close below swing low (1.00)
    assert swing_break(bars, lookback=3) == -1


def test_swing_break_none_inside_range():
    bars = [_bar(1.00, 1.01, 1.00, 1.005) for _ in range(3)]
    bars.append(_bar(1.005, 1.008, 1.002, 1.006))
    assert swing_break(bars, lookback=3) == 0


def test_swing_break_insufficient_history():
    bars = [_bar(1.00, 1.01, 1.00, 1.005) for _ in range(2)]
    assert swing_break(bars, lookback=3) == 0
