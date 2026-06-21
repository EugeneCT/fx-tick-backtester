from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from fxtrader.cli import app
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore

runner = CliRunner()


def _seed(tmp_path):
    store = TickStore(tmp_path)
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    ticks = []
    # oscillating 30s ticks (ranging) so the fader has stretched+reverting moves
    for i in range(2000):
        base = 1.10 + 0.0010 * ((i // 5) % 2)  # flip every 5 ticks -> mean-reverting
        ticks.append(Tick("EURUSD", start + timedelta(seconds=i * 30), base, base + 0.00002))
    store.write_ticks("EURUSD", ticks)


def test_backtest_meanrev_runs_end_to_end(tmp_path):
    _seed(tmp_path)
    result = runner.invoke(app, [
        "backtest", "--strategy", "meanrev", "--symbol", "EURUSD",
        "--start", "2024-01-02", "--end", "2024-01-03",
        "--lookback", "20", "--entry-z", "1.5", "--exit-z", "0.3", "--er-threshold", "0.5",
        "--exit", "fixed", "--stop-pips", "15", "--tp-pips", "30",
        "--bar-minutes", "5", "--data-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "meanrev" in result.output.lower()
