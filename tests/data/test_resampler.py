from datetime import datetime, timedelta, timezone

from fxtrader.core.types import Tick
from fxtrader.data.feed import Resampler


def _tick(second, bid, ask):
    return Tick(symbol="EURUSD",
                ts=datetime(2024, 1, 2, 9, 0, second, tzinfo=timezone.utc),
                bid=bid, ask=ask)


def test_resampler_emits_bar_on_bucket_rollover():
    r = Resampler(timedelta(minutes=1))
    # all within minute 09:00
    assert r.update(_tick(0, 1.1000, 1.1002)) is None   # mid 1.1001
    assert r.update(_tick(20, 1.1010, 1.1012)) is None   # mid 1.1011 (high)
    assert r.update(_tick(40, 1.0990, 1.0992)) is None   # mid 1.0991 (low)
    # crossing into 09:01 closes the 09:00 bar
    bar = r.update(Tick(symbol="EURUSD",
                        ts=datetime(2024, 1, 2, 9, 1, 0, tzinfo=timezone.utc),
                        bid=1.1000, ask=1.1002))
    assert bar is not None
    assert bar.ts == datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    assert round(bar.open, 5) == 1.1001
    assert round(bar.high, 5) == 1.1011
    assert round(bar.low, 5) == 1.0991
    assert round(bar.close, 5) == 1.0991


def test_resampler_flush_returns_last_partial_bar():
    r = Resampler(timedelta(minutes=1))
    r.update(_tick(0, 1.1000, 1.1002))
    r.update(_tick(30, 1.1010, 1.1012))
    bar = r.flush()
    assert bar is not None
    assert round(bar.open, 5) == 1.1001
    assert round(bar.close, 5) == 1.1011
    assert r.flush() is None  # nothing left


def test_resampler_counts_ticks_as_volume():
    r = Resampler(timedelta(minutes=1))
    r.update(_tick(0, 1.1, 1.1002))
    r.update(_tick(10, 1.1, 1.1002))
    bar = r.flush()
    assert bar.volume == 2
