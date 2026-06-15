"""FX market calendar and time iteration helpers.

The spot FX market trades continuously from Sunday 22:00 UTC to Friday 22:00 UTC.
Dukascopy has no data outside that window, so we skip those hours entirely.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone


def is_market_open(hour: datetime) -> bool:
    """True if the FX market is open at this hour. Closed Fri 22:00 -> Sun 22:00 UTC."""
    weekday = hour.weekday()  # Mon=0 .. Sat=5, Sun=6
    if weekday == 5:  # Saturday: closed all day
        return False
    if weekday == 6 and hour.hour < 22:  # Sunday before 22:00
        return False
    if weekday == 4 and hour.hour >= 22:  # Friday from 22:00
        return False
    return True


def open_hours(start: datetime, end: datetime) -> Iterator[datetime]:
    """Yield each market-open hour timestamp in [start, end), aligned to the hour."""
    cursor = start.replace(minute=0, second=0, microsecond=0)
    while cursor < end:
        if is_market_open(cursor):
            yield cursor
        cursor += timedelta(hours=1)


def iter_months(start: datetime, end: datetime):
    """Yield (ym, month_start, next_month_start) for each calendar month overlapping [start, end).

    ym is a 'YYYY-MM' string. month_start/next_month_start are tz-aware UTC.
    """
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cursor < end:
        if cursor.month == 12:
            nxt = datetime(cursor.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            nxt = datetime(cursor.year, cursor.month + 1, 1, tzinfo=timezone.utc)
        yield f"{cursor.year:04d}-{cursor.month:02d}", cursor, nxt
        cursor = nxt
