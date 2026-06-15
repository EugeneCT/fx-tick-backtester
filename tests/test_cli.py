import lzma
import struct
from datetime import datetime, timezone

from typer.testing import CliRunner

from fxtrader.cli import app, run_download
from fxtrader.data.store import TickStore

runner = CliRunner()


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


def test_download_is_a_subcommand():
    # The `download` subcommand must exist (regression: a single-command Typer
    # app collapses the subcommand away, breaking the documented invocation).
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "download" in result.stdout


def test_download_subcommand_runs_offline(tmp_path):
    # start == end => zero hours => no HTTP call; just exercises CLI wiring.
    result = runner.invoke(
        app,
        ["download", "--symbols", "EURUSD",
         "--start", "2024-01-06", "--end", "2024-01-06",
         "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "EURUSD: 0 ticks" in result.stdout
