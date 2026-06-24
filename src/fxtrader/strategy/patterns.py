"""Pure price-action pattern detectors. No I/O, no state. Each returns -1 / 0 / +1."""

from __future__ import annotations

from collections.abc import Sequence

from fxtrader.core.types import Bar


def pin_bar(bar: Bar, *, wick_ratio: float, body_frac: float) -> int:
    """Rejection candle: small body, one dominant wick, close in the far third.

    +1 bullish (long lower wick, close in upper third); -1 bearish (mirror); else 0.
    """
    rng = bar.high - bar.low
    if rng <= 0:
        return 0
    body = abs(bar.close - bar.open)
    if body > body_frac * rng:
        return 0  # body too large to be a pin
    upper = bar.high - max(bar.open, bar.close)
    lower = min(bar.open, bar.close) - bar.low
    eps = rng * 1e-9
    if lower >= wick_ratio * max(body, eps) and bar.close >= bar.low + (2.0 / 3.0) * rng:
        return 1
    if upper >= wick_ratio * max(body, eps) and bar.close <= bar.low + (1.0 / 3.0) * rng:
        return -1
    return 0


def engulfing(prev: Bar, bar: Bar) -> int:
    """Current real body fully engulfs the prior real body and flips direction."""
    prev_bear = prev.close < prev.open
    prev_bull = prev.close > prev.open
    cur_bull = bar.close > bar.open
    cur_bear = bar.close < bar.open
    if prev_bear and cur_bull and bar.close >= prev.open and bar.open <= prev.close:
        return 1
    if prev_bull and cur_bear and bar.close <= prev.open and bar.open >= prev.close:
        return -1
    return 0


def is_inside_bar(prev: Bar, bar: Bar) -> bool:
    """True when `bar`'s range is fully contained within `prev`'s range."""
    return bar.high <= prev.high and bar.low >= prev.low


def swing_break(bars: Sequence[Bar], lookback: int) -> int:
    """Close beyond the swing high/low of the `lookback` bars preceding the current bar.

    +1 close > prior swing high; -1 close < prior swing low; else 0.
    """
    if len(bars) < lookback + 1:
        return 0
    window = bars[-(lookback + 1):-1]
    cur = bars[-1]
    swing_high = max(b.high for b in window)
    swing_low = min(b.low for b in window)
    if cur.close > swing_high:
        return 1
    if cur.close < swing_low:
        return -1
    return 0
