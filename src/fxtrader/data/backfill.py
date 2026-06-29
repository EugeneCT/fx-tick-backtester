"""Resilient bulk Dukascopy backfill: retries, resume, concurrency, monthly writes."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _fetch_hour(fetch: Downloader, symbol: str, hour: datetime) -> list:
    """Download + decode a single open hour. Raises OSError on transient network failure."""
    return decode_bi5(fetch(build_url(symbol, hour)), hour, symbol)


def backfill(
    symbols: list[str],
    start: datetime,
    end: datetime,
    store: TickStore,
    *,
    downloader: Downloader = _http_download,
    workers: int = 8,
    retries: int = 3,
    month_rounds: int = 4,
    round_backoff: float = 5.0,
    sleep: Sleep = time.sleep,
    ledger: "CoverageLedger | None" = None,
    progress: Progress = lambda msg: None,
) -> dict[str, int]:
    """Resiliently download [start, end) for each symbol into the store, month by month.

    Concurrent per-hour fetches, one write per month, resumable via the ledger.

    Resilience: within a month, hours that still fail after ``retries`` per-request
    attempts are collected and re-fetched in up to ``month_rounds`` further rounds
    (Dukascopy intermittently throttles/hangs individual hours under concurrency).
    A 404 / closed hour decodes to no ticks and is never a failure, so this never
    introduces silent gaps — the month is only marked done once every open hour was
    fetched. If hours remain unfetchable after all rounds, the month aborts (raises)
    so an outer retry loop can resume it later.

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
            n_ticks = 0
            if hours:
                # Convert each hour's ticks to a compact Arrow frame as it arrives and
                # drop the heavy per-tick Python objects immediately — holding a whole
                # dense month (~6M Tick objects) at once OOMs small boxes. Only the
                # *failed* hours are tracked for re-fetch in later rounds.
                frames = []
                pending = hours
                for round_i in range(month_rounds + 1):
                    failed: list[datetime] = []
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = {pool.submit(_fetch_hour, fetch, symbol, h): h for h in pending}
                        for fut in as_completed(futures):
                            h = futures[fut]
                            try:
                                hour_ticks = fut.result()
                            except OSError:
                                failed.append(h)
                                continue
                            if hour_ticks:
                                frames.append(store.frame_from_ticks(hour_ticks))
                                n_ticks += len(hour_ticks)
                    if not failed:
                        break
                    if round_i == month_rounds:
                        raise TimeoutError(
                            f"{symbol} {ym}: {len(failed)} of {len(hours)} hours "
                            f"unfetchable after {month_rounds + 1} rounds"
                        )
                    progress(
                        f"{symbol} {ym}: {len(failed)} hours failed, "
                        f"retry round {round_i + 1}/{month_rounds}"
                    )
                    sleep(round_backoff)
                    pending = failed
                store.write_frames(symbol, frames)
            if ledger is not None:
                ledger.mark_done(symbol, ym)
            total += n_ticks
            progress(f"{symbol} {ym}: {n_ticks} ticks from {len(hours)} open hours")
        summary[symbol] = total
    return summary
