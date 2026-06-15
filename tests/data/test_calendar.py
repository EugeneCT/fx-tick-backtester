from datetime import datetime, timezone

from fxtrader.data.calendar import is_market_open, iter_months, open_hours


def _h(y, mo, d, hh):
    return datetime(y, mo, d, hh, tzinfo=timezone.utc)


def test_weekday_hour_is_open():
    # Tuesday 2024-01-02 10:00 UTC
    assert is_market_open(_h(2024, 1, 2, 10)) is True


def test_saturday_is_closed_all_day():
    # Saturday 2024-01-06
    assert is_market_open(_h(2024, 1, 6, 3)) is False
    assert is_market_open(_h(2024, 1, 6, 23)) is False


def test_sunday_opens_at_22():
    # Sunday 2024-01-07
    assert is_market_open(_h(2024, 1, 7, 21)) is False
    assert is_market_open(_h(2024, 1, 7, 22)) is True


def test_friday_closes_at_22():
    # Friday 2024-01-05
    assert is_market_open(_h(2024, 1, 5, 21)) is True
    assert is_market_open(_h(2024, 1, 5, 22)) is False


def test_open_hours_filters_closed_and_respects_half_open_range():
    start = _h(2024, 1, 5, 21)   # Fri 21:00 (open)
    end = _h(2024, 1, 7, 23)     # Sun 23:00 (exclusive)
    hours = list(open_hours(start, end))
    # Fri 21:00 open; Fri 22:00..Sun 21:00 closed; Sun 22:00 open; end exclusive at 23:00 so 22:00 only
    assert hours == [_h(2024, 1, 5, 21), _h(2024, 1, 7, 22)]


def test_iter_months_covers_overlapping_calendar_months():
    start = _h(2024, 1, 15, 9)
    end = _h(2024, 3, 2, 0)
    got = [(ym, ms.month, me.month) for ym, ms, me in iter_months(start, end)]
    assert got == [
        ("2024-01", 1, 2),
        ("2024-02", 2, 3),
        ("2024-03", 3, 4),
    ]
