from datetime import datetime, timezone

import pytest

from fxtrader.core.clock import SimClock, WallClock


def test_simclock_returns_set_time():
    clock = SimClock()
    ts = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    clock.set_time(ts)
    assert clock.now() == ts


def test_simclock_raises_before_time_set():
    clock = SimClock()
    with pytest.raises(RuntimeError):
        clock.now()


def test_wallclock_is_utc_and_advances():
    clock = WallClock()
    first = clock.now()
    assert first.tzinfo is timezone.utc
    assert clock.now() >= first
