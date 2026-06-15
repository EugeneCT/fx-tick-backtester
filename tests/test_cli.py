import lzma
import struct
from datetime import datetime, timezone

from fxtrader.cli import run_download
from fxtrader.data.store import TickStore


def _payload(records):
    raw = b"".join(struct.pack(">IIIff", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def test_run_download_populates_store(tmp_path):
    payload = _payload([(0, 110020, 110000, 1.0, 1.0)])

    summary = run_download(
        symbols=["EURUSD"],
        start=datetime(2024, 1, 2, 9, tzinfo=timezone.utc),
        end=datetime(2024, 1, 2, 11, tzinfo=timezone.utc),  # 2 hours
        data_dir=tmp_path,
        downloader=lambda url: payload,
    )

    assert summary["EURUSD"] == 2  # one tick per hour
    store = TickStore(tmp_path)
    df = store.read_ticks("EURUSD",
                          datetime(2024, 1, 2, tzinfo=timezone.utc),
                          datetime(2024, 1, 3, tzinfo=timezone.utc))
    assert df.height == 2
