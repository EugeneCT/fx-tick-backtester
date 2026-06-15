import lzma
import struct
from datetime import datetime, timezone

from fxtrader.data.dukascopy import download_range, fetch_hour


def _payload(records):
    raw = b"".join(struct.pack(">IIIff", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def test_fetch_hour_decodes_downloaded_payload():
    payload = _payload([(0, 110020, 110000, 1.0, 1.0)])
    calls = []

    def fake_download(url: str) -> bytes:
        calls.append(url)
        return payload

    hour = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    ticks = fetch_hour("EURUSD", hour, downloader=fake_download)

    assert len(ticks) == 1
    assert calls[0].endswith("/EURUSD/2024/00/02/09h_ticks.bi5")


def test_fetch_hour_treats_empty_as_no_ticks():
    ticks = fetch_hour(
        "EURUSD",
        datetime(2024, 1, 6, 9, 0, tzinfo=timezone.utc),  # a Saturday
        downloader=lambda url: b"",
    )
    assert ticks == []


def test_download_range_iterates_each_hour():
    seen = []

    def fake_download(url: str) -> bytes:
        seen.append(url)
        return _payload([(0, 110020, 110000, 1.0, 1.0)])

    start = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)  # exclusive
    ticks = download_range("EURUSD", start, end, downloader=fake_download)

    assert len(seen) == 3            # hours 09, 10, 11
    assert len(ticks) == 3           # one tick per hour
    assert ticks == sorted(ticks, key=lambda t: t.ts)
