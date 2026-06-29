import json
import lzma
import struct
from datetime import datetime, timezone

import pytest

from fxtrader.data.backfill import CoverageLedger, backfill, resilient_download
from fxtrader.data.store import TickStore


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


def test_backfill_retries_failed_hours_across_rounds(tmp_path):
    # An hour that fails its first fetch but succeeds on a later round must NOT abort
    # the month; the month completes once every open hour is fetched.
    calls = {}

    def flaky(url):
        calls[url] = calls.get(url, 0) + 1
        if "/11h_" in url and calls[url] == 1:  # the 11:00 hour fails once, then succeeds
            raise TimeoutError("throttled")
        return _payload(1)

    store = TickStore(tmp_path)
    ledger = CoverageLedger(tmp_path / "coverage.json")
    slept = []
    start = datetime(2024, 1, 2, 9, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 12, tzinfo=timezone.utc)  # Tue -> 9,10,11 open hours

    summary = backfill(
        ["EURUSD"], start, end, store,
        downloader=flaky, retries=0, workers=2,
        round_backoff=0.0, sleep=slept.append, ledger=ledger,
    )
    assert summary["EURUSD"] == 3  # all 3 hours fetched, incl. the retried 11:00
    assert ledger.is_done("EURUSD", "2024-01")
    assert len(slept) == 1  # exactly one inter-round backoff


def test_backfill_aborts_month_when_hours_never_fetch(tmp_path):
    # A persistently unfetchable hour must raise (not silently skip) -> no data gaps.
    def always_fail_11(url):
        if "/11h_" in url:
            raise TimeoutError("down")
        return _payload(1)

    store = TickStore(tmp_path)
    ledger = CoverageLedger(tmp_path / "coverage.json")
    with pytest.raises(TimeoutError):
        backfill(
            ["EURUSD"],
            datetime(2024, 1, 2, 9, tzinfo=timezone.utc),
            datetime(2024, 1, 2, 12, tzinfo=timezone.utc),
            store, downloader=always_fail_11, retries=0,
            workers=2, month_rounds=2, round_backoff=0.0, sleep=lambda s: None,
            ledger=ledger,
        )
    assert not ledger.is_done("EURUSD", "2024-01")  # month not marked complete


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
