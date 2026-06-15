"""Resilient bulk Dukascopy backfill: retries, resume, concurrency, monthly writes."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from fxtrader.data.dukascopy import Downloader

Sleep = Callable[[float], None]
Jitter = Callable[[], float]


def resilient_download(
    downloader: Downloader,
    *,
    retries: int = 3,
    backoff: float = 0.5,
    sleep: Sleep = time.sleep,
    jitter: Jitter | None = None,
) -> Downloader:
    """Wrap a Downloader with retry + exponential backoff on transient (OSError) failures.

    A 404 is already converted to b"" upstream, so it neither raises nor retries.
    """
    import random

    _jitter: Jitter = jitter if jitter is not None else random.random

    def _wrapped(url: str) -> bytes:
        for attempt in range(retries + 1):
            try:
                return downloader(url)
            except OSError:
                if attempt == retries:
                    raise
                sleep(backoff * (2**attempt) + _jitter())
        raise AssertionError("unreachable")  # pragma: no cover

    return _wrapped


class CoverageLedger:
    """Tracks which (symbol, 'YYYY-MM') months have been fully downloaded, for resume."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._done: dict[str, set[str]] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._done = {sym: set(months) for sym, months in raw.items()}

    def is_done(self, symbol: str, ym: str) -> bool:
        return ym in self._done.get(symbol, set())

    def mark_done(self, symbol: str, ym: str) -> None:
        self._done.setdefault(symbol, set()).add(ym)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {sym: sorted(months) for sym, months in self._done.items()}
        self.path.write_text(json.dumps(serializable, indent=2, sort_keys=True))
