from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from fxtrader.cli import app
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore

runner = CliRunner()


def _seed(tmp_path, days=6):
    """Daily pattern: flat 1.1000 during 00:00-06:59 UTC (the range), then 1.1100 from 07:00
    (a clean upside break), back to 1.1000 after 21:00. 30-min ticks across each weekday."""
    store = TickStore(tmp_path)
    ticks = []
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    for d in range(days):
        day0 = base + timedelta(days=d)
        for slot in range(48):  # 30-min slots across 24h
            ts = day0 + timedelta(minutes=30 * slot)
            hour = ts.hour
            px = 1.1100 if 7 <= hour < 21 else 1.1000
            ticks.append(Tick("EURUSD", ts, px, px + 0.00002))
    store.write_ticks("EURUSD", ticks)


def test_backtest_orb_runs_end_to_end(tmp_path):
    _seed(tmp_path)
    result = runner.invoke(app, [
        "backtest", "--strategy", "orb", "--symbol", "EURUSD",
        "--start", "2024-01-01", "--end", "2024-01-08",
        "--range-start", "0", "--range-end", "7", "--entry-end", "16", "--exit-hour", "21",
        "--exit", "fixed", "--stop-pips", "30", "--tp-pips", "60",
        "--bar-minutes", "30", "--data-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "orb" in result.output.lower()
