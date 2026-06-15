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
