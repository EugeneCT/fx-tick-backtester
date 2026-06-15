"""Time source abstraction: simulated (backtest) vs wall-clock (live).

Strategies read time ONLY via a Clock so the same code works in both engines.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


class SimClock:
    """Driven by event timestamps during a backtest."""

    def __init__(self) -> None:
        self._now: datetime | None = None

    def set_time(self, ts: datetime) -> None:
        self._now = ts

    def now(self) -> datetime:
        if self._now is None:
            raise RuntimeError("SimClock time not set; advance an event first")
        return self._now


class WallClock:
    """Real time, used live."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)
