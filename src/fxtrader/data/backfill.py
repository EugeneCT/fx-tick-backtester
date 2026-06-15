"""Resilient bulk Dukascopy backfill: retries, resume, concurrency, monthly writes."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from fxtrader.data.calendar import iter_months, open_hours
from fxtrader.data.dukascopy import Downloader, _http_download, build_url, decode_bi5
from fxtrader.data.store import TickStore

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
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(serializable, indent=2, sort_keys=True))
        os.replace(tmp, self.path)


Progress = Callable[[str], None]


def backfill(
    symbols: list[str],
    start: datetime,
    end: datetime,
    store: TickStore,
    *,
    downloader: Downloader = _http_download,
    workers: int = 8,
    retries: int = 3,
    ledger: "CoverageLedger | None" = None,
    progress: Progress = lambda msg: None,
) -> dict[str, int]:
    """Resiliently download [start, end) for each symbol into the store, month by month.

    Concurrent per-hour fetches, one write per month, resumable via the ledger.
    Returns {symbol: total_ticks_written_this_run}.
    """
    fetch = resilient_download(downloader, retries=retries, jitter=random.random)
    summary: dict[str, int] = {}
    for symbol in symbols:
        total = 0
        for ym, month_start, month_end in iter_months(start, end):
            if ledger is not None and ledger.is_done(symbol, ym):
                progress(f"{symbol} {ym}: skip (already complete)")
                continue
            lo = max(month_start, start)
            hi = min(month_end, end)
            hours = list(open_hours(lo, hi))
            ticks = []
            if hours:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for hour_ticks in pool.map(
                        lambda h: decode_bi5(fetch(build_url(symbol, h)), h, symbol), hours
                    ):
                        ticks.extend(hour_ticks)
                ticks.sort(key=lambda t: t.ts)
                store.write_ticks(symbol, ticks)
            if ledger is not None:
                ledger.mark_done(symbol, ym)
            total += len(ticks)
            progress(f"{symbol} {ym}: {len(ticks)} ticks from {len(hours)} open hours")
        summary[symbol] = total
    return summary
