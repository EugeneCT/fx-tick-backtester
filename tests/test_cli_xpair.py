from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from fxtrader.cli import app
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore

runner = CliRunner()


def _seed(tmp_path, n=2000):
    """Two-symbol store: GBPUSD ~flat, EURUSD oscillating -> a mean-reverting spread."""
    store = TickStore(tmp_path)
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    eur, gbp = [], []
    for i in range(n):
        ts = start + timedelta(seconds=i * 30)
        e = 1.10 + 0.0010 * ((i // 5) % 2)   # oscillates -> spread mean-reverts
        eur.append(Tick("EURUSD", ts, e, e + 0.00002))
        gbp.append(Tick("GBPUSD", ts, 1.2500, 1.25002))
    store.write_ticks("EURUSD", eur)
    store.write_ticks("GBPUSD", gbp)


def test_backtest_xpair_runs_end_to_end(tmp_path):
    _seed(tmp_path)
    result = runner.invoke(app, [
        "backtest", "--strategy", "xpair", "--symbol", "EURUSD", "--symbol2", "GBPUSD",
        "--start", "2024-01-02", "--end", "2024-01-03",
        "--lookback", "20", "--entry-z", "1.5", "--exit-z", "0.3", "--stop-z", "4.0",
        "--bar-minutes", "5", "--data-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "xpair" in result.output.lower()
