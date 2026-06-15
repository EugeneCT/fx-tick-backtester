# Resilient Bulk Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make Dukascopy downloading robust enough for a real 3-pair × ~3-year backfill: skip market-closed hours, retry transient errors with backoff, download concurrently, write once per month, and resume from where it left off.

**Architecture:** A new `data/calendar.py` (FX session filter + month/hour iteration) and `data/backfill.py` (`resilient_download` retry wrapper, `CoverageLedger` for resume, and a `backfill()` orchestrator that fetches a month's open hours concurrently, writes them to the `TickStore` in one shot, and marks the month done). A `fxtrader backfill` CLI command drives it. The existing exact-range `download` command/`run_download` are left untouched.

**Tech Stack:** Python 3.12 (`.venv`), stdlib `concurrent.futures` (ThreadPoolExecutor), `urllib` (errors are all `OSError` subclasses), polars (via existing `TickStore`), typer, pytest.

**Conventions:**
- All timestamps tz-aware UTC. Use `.venv/bin/pytest` / `.venv/bin/python` (shell `source` doesn't persist).
- No network in unit tests — inject fake downloaders and a no-op `sleep`.
- Commit after each task with the message in its final step.
- FX market hours: **closed Friday 22:00 UTC → Sunday 22:00 UTC** (covers the weekend; Dukascopy returns empty/404 there anyway, we just avoid the doomed requests).
- Backfill granularity is **one calendar month** per write/ledger entry (avoids the O(n²) re-read+rewrite that per-hour writes to `TickStore.write_ticks` would cause). Hours are still clamped to the requested `[start, end)`.

---

## File structure

```
src/fxtrader/data/calendar.py    # Task 1 — is_market_open, open_hours, iter_months
src/fxtrader/data/backfill.py    # Task 2 — resilient_download, CoverageLedger
                                 # Task 3 — backfill() orchestrator
src/fxtrader/cli.py              # Task 4 — `backfill` command + run_backfill (MODIFY)
tests/data/test_calendar.py      # Task 1
tests/data/test_backfill.py      # Tasks 2 & 3
tests/test_cli_backfill.py       # Task 4
```

---

## Task 1: FX market calendar + iteration helpers

**Files:**
- Create: `src/fxtrader/data/calendar.py`
- Create: `tests/data/test_calendar.py`

- [ ] **Step 1: Write the failing test `tests/data/test_calendar.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_calendar.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.data.calendar'`

- [ ] **Step 3: Write `src/fxtrader/data/calendar.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_calendar.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/data/calendar.py tests/data/test_calendar.py
git commit -m "feat: FX market calendar (session filter + hour/month iteration)"
```

---

## Task 2: Resilient downloader + coverage ledger

**Files:**
- Create: `src/fxtrader/data/backfill.py`
- Create: `tests/data/test_backfill.py`

`resilient_download` wraps any `Downloader` with retry + exponential backoff on transient errors (all `urllib`/socket errors are `OSError` subclasses; a 404 is already turned into `b""` by `_http_download`, so it never raises and never retries). `CoverageLedger` records completed `(symbol, 'YYYY-MM')` months as JSON for resume.

- [ ] **Step 1: Write the failing test `tests/data/test_backfill.py`**

```python
import json

import pytest

from fxtrader.data.backfill import CoverageLedger, resilient_download


def test_resilient_download_passes_through_on_success():
    calls = []

    def ok(url):
        calls.append(url)
        return b"payload"

    wrapped = resilient_download(ok, retries=3, sleep=lambda s: None)
    assert wrapped("u") == b"payload"
    assert len(calls) == 1  # no retries on success


def test_resilient_download_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky(url):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("boom")  # OSError subclass = transient
        return b"ok"

    slept = []
    wrapped = resilient_download(flaky, retries=3, backoff=0.5,
                                 sleep=slept.append, jitter=lambda: 0.0)
    assert wrapped("u") == b"ok"
    assert attempts["n"] == 3
    assert slept == [0.5, 1.0]  # backoff * 2**attempt for attempts 0,1


def test_resilient_download_gives_up_after_retries():
    def always_fail(url):
        raise TimeoutError("nope")  # OSError subclass

    wrapped = resilient_download(always_fail, retries=2, sleep=lambda s: None,
                                 jitter=lambda: 0.0)
    with pytest.raises(TimeoutError):
        wrapped("u")


def test_resilient_download_does_not_retry_empty_payload():
    # 404 -> _http_download returns b"" (no raise); wrapper must pass it straight through.
    calls = []

    def empty(url):
        calls.append(url)
        return b""

    wrapped = resilient_download(empty, retries=3, sleep=lambda s: None)
    assert wrapped("u") == b""
    assert len(calls) == 1


def test_coverage_ledger_marks_and_persists(tmp_path):
    path = tmp_path / "coverage.json"
    ledger = CoverageLedger(path)
    assert ledger.is_done("EURUSD", "2024-01") is False
    ledger.mark_done("EURUSD", "2024-01")
    assert ledger.is_done("EURUSD", "2024-01") is True

    # persisted and reloadable
    reloaded = CoverageLedger(path)
    assert reloaded.is_done("EURUSD", "2024-01") is True
    assert reloaded.is_done("EURUSD", "2024-02") is False
    on_disk = json.loads(path.read_text())
    assert on_disk == {"EURUSD": ["2024-01"]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/data/test_backfill.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.data.backfill'`

- [ ] **Step 3: Write `src/fxtrader/data/backfill.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_backfill.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/data/backfill.py tests/data/test_backfill.py
git commit -m "feat: resilient_download (retry/backoff) and CoverageLedger (resume)"
```

---

## Task 3: backfill() orchestrator

**Files:**
- Modify: `src/fxtrader/data/backfill.py` (add `backfill`)
- Modify: `tests/data/test_backfill.py` (add orchestration tests)

`backfill` walks each symbol's months in `[start, end)`, skips ledger-complete months, fetches that month's open hours **concurrently** (ThreadPoolExecutor), decodes, writes the month's ticks to the store in one `write_ticks`, then marks the month done. Returns `{symbol: total_ticks}`.

- [ ] **Step 1: Write the failing tests (append to `tests/data/test_backfill.py`)**

```python
import lzma
import struct
from datetime import datetime, timezone

from fxtrader.data.backfill import backfill
from fxtrader.data.store import TickStore


def _payload(n=1):
    rec = b"".join(struct.pack(">IIIff", i, 110020, 110000, 1.0, 1.0) for i in range(n))
    return lzma.compress(rec, format=lzma.FORMAT_ALONE)


def test_backfill_populates_store_and_skips_closed_hours(tmp_path):
    seen = []

    def fake(url):
        seen.append(url)
        return _payload(1)

    store = TickStore(tmp_path)
    # Fri 2024-01-05 21:00 (open) through Sun 2024-01-07 23:00 -> only Fri21 + Sun22 open
    summary = backfill(
        ["EURUSD"],
        datetime(2024, 1, 5, 21, tzinfo=timezone.utc),
        datetime(2024, 1, 7, 23, tzinfo=timezone.utc),
        store,
        downloader=fake,
        workers=2,
    )
    assert summary["EURUSD"] == 2  # one tick per open hour, 2 open hours
    # No Saturday URLs were requested
    assert not any("/2024/00/06/" in u for u in seen)  # Jan is month index 00; day 06 = Saturday
    df = store.read_ticks("EURUSD",
                          datetime(2024, 1, 1, tzinfo=timezone.utc),
                          datetime(2024, 2, 1, tzinfo=timezone.utc))
    assert df.height == 2


def test_backfill_resumes_via_ledger(tmp_path):
    calls = {"n": 0}

    def counting(url):
        calls["n"] += 1
        return _payload(1)

    store = TickStore(tmp_path)
    from fxtrader.data.backfill import CoverageLedger
    ledger = CoverageLedger(tmp_path / "coverage.json")

    start = datetime(2024, 1, 2, 9, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 12, tzinfo=timezone.utc)  # Tue, 3 open hours, single month

    backfill(["EURUSD"], start, end, store, downloader=counting, ledger=ledger, workers=2)
    first = calls["n"]
    assert first == 3
    assert ledger.is_done("EURUSD", "2024-01")

    # Re-run: month is complete, so no further downloads
    backfill(["EURUSD"], start, end, store, downloader=counting, ledger=ledger, workers=2)
    assert calls["n"] == first  # unchanged


def test_backfill_reports_progress(tmp_path):
    msgs = []
    store = TickStore(tmp_path)
    backfill(
        ["EURUSD"],
        datetime(2024, 1, 2, 9, tzinfo=timezone.utc),
        datetime(2024, 1, 2, 11, tzinfo=timezone.utc),
        store,
        downloader=lambda url: _payload(1),
        workers=2,
        progress=msgs.append,
    )
    assert any("EURUSD" in m and "2024-01" in m for m in msgs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/data/test_backfill.py -v`
Expected: FAIL with `ImportError: cannot import name 'backfill'`

- [ ] **Step 3: Append `backfill` to `src/fxtrader/data/backfill.py`**

Add to the imports at the top (merge with existing):

```python
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from fxtrader.data.calendar import iter_months, open_hours
from fxtrader.data.dukascopy import _http_download, build_url, decode_bi5
from fxtrader.data.store import TickStore
```

(Remove the local `import random` inside `resilient_download` if you hoist it to the top — either is fine, just no duplicate.)

Append at the end of the file:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/data/test_backfill.py -v`
Expected: PASS (8 passed total in this file)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS (everything; the new tests added on top of the existing suite)

- [ ] **Step 6: Commit**

```bash
git add src/fxtrader/data/backfill.py tests/data/test_backfill.py
git commit -m "feat: backfill() orchestrator (concurrent, monthly writes, resumable)"
```

---

## Task 4: `fxtrader backfill` CLI command

**Files:**
- Modify: `src/fxtrader/cli.py` (add `run_backfill` + `backfill` command)
- Create: `tests/test_cli_backfill.py`

- [ ] **Step 1: Write the failing test `tests/test_cli_backfill.py`**

```python
import lzma
import struct
from datetime import datetime, timezone

from fxtrader.cli import run_backfill
from fxtrader.data.store import TickStore


def _payload():
    rec = struct.pack(">IIIff", 0, 110020, 110000, 1.0, 1.0)
    return lzma.compress(rec, format=lzma.FORMAT_ALONE)


def test_run_backfill_populates_store_with_resume(tmp_path):
    summary = run_backfill(
        symbols=["EURUSD"],
        start=datetime(2024, 1, 2, 9, tzinfo=timezone.utc),
        end=datetime(2024, 1, 2, 12, tzinfo=timezone.utc),  # Tue, 3 open hours
        data_dir=tmp_path,
        downloader=lambda url: _payload(),
        workers=2,
        resume=True,
    )
    assert summary["EURUSD"] == 3
    # coverage ledger written
    assert (tmp_path / "coverage.json").exists()
    store = TickStore(tmp_path)
    df = store.read_ticks("EURUSD",
                          datetime(2024, 1, 1, tzinfo=timezone.utc),
                          datetime(2024, 2, 1, tzinfo=timezone.utc))
    assert df.height == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_backfill.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_backfill'`

- [ ] **Step 3: Add to `src/fxtrader/cli.py`**

Add to the imports at the top (merge with existing):

```python
from fxtrader.data.backfill import CoverageLedger, backfill
```

Add this helper after `run_download` (keep `Downloader`/`_http_download` already imported):

```python
def run_backfill(
    symbols: list[str],
    start: datetime,
    end: datetime,
    data_dir: str | Path,
    downloader: Downloader = _http_download,
    workers: int = 8,
    retries: int = 3,
    resume: bool = True,
    progress=lambda msg: None,
) -> dict[str, int]:
    """Resilient bulk download into a TickStore, resumable via a coverage ledger."""
    store = TickStore(data_dir)
    ledger = CoverageLedger(Path(data_dir) / "coverage.json") if resume else None
    return backfill(
        symbols, start, end, store,
        downloader=downloader, workers=workers, retries=retries,
        ledger=ledger, progress=progress,
    )
```

Add this command (after the `download` command):

```python
@app.command()
def backfill(
    symbols: str = typer.Option(..., help="Comma-separated, e.g. EURUSD,GBPUSD,USDJPY"),
    start: str = typer.Option(..., help="Inclusive start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Exclusive end date YYYY-MM-DD"),
    data_dir: str = typer.Option("data", help="Parquet store root"),
    workers: int = typer.Option(8, help="Concurrent hourly downloads"),
    retries: int = typer.Option(3, help="Retries per hour on transient errors"),
    resume: bool = typer.Option(True, help="Skip months already in the coverage ledger"),
) -> None:
    """Resilient bulk download of Dukascopy ticks (concurrent, resumable)."""
    summary = run_backfill(
        symbols=[s.strip() for s in symbols.split(",") if s.strip()],
        start=_parse_day(start),
        end=_parse_day(end),
        data_dir=data_dir,
        workers=workers,
        retries=retries,
        resume=resume,
        progress=typer.echo,
    )
    for sym, count in summary.items():
        typer.echo(f"TOTAL {sym}: {count} ticks")
```

Note: the command function is named `backfill`, shadowing the imported `backfill` from `fxtrader.data.backfill` *within the module namespace* once defined. To avoid the collision, import the orchestrator under an alias instead: change the import to `from fxtrader.data.backfill import CoverageLedger, backfill as _backfill` and call `_backfill(...)` inside `run_backfill`. Apply this alias so both names coexist. Verify with the test.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli_backfill.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Verify the command is wired**

Run: `.venv/bin/fxtrader backfill --help`
Expected: usage listing `--symbols`, `--start`, `--end`, `--data-dir`, `--workers`, `--retries`, `--resume`.
Also confirm `download` still works: `.venv/bin/fxtrader --help` lists BOTH `download` and `backfill` commands.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS (all tests).

- [ ] **Step 7: Commit**

```bash
git add src/fxtrader/cli.py tests/test_cli_backfill.py
git commit -m "feat: fxtrader backfill CLI command (resilient bulk download)"
```

---

## Self-review

**Spec coverage (§5.1 robustness items):** retries/backoff (Task 2 `resilient_download`), resumable (Task 2/3 `CoverageLedger` + month-skip), gap-detection (ledger + market calendar Task 1), rate-limiting/politeness (bounded `workers` + backoff), monthly bulk writes (Task 3, avoids O(n²) store rewrites), progress reporting (Task 3/4). ✓

**Placeholder scan:** none — all code/tests are complete and runnable.

**Type/signature consistency:** `is_market_open(hour)`, `open_hours(start, end)`, `iter_months(start, end) -> (ym, month_start, next_month_start)`, `resilient_download(downloader, *, retries, backoff, sleep, jitter)`, `CoverageLedger(path).is_done/mark_done`, `backfill(symbols, start, end, store, *, downloader, workers, retries, ledger, progress) -> {symbol:int}`, `run_backfill(symbols, start, end, data_dir, downloader, workers, retries, resume, progress)`. Used consistently across tasks. The `backfill` function vs CLI command name collision is explicitly resolved via the `as _backfill` alias in Task 4.

**Scope:** Stops at a resilient, resumable, concurrent bulk downloader + CLI. The actual download RUN (1-month validation then 3-year backfill) is performed by the operator after these tasks, not part of the plan's code.
