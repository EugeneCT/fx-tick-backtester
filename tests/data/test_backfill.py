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
