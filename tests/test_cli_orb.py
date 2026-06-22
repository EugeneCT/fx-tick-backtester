from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from fxtrader.cli import app, run_optimize
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


def test_optimize_orb_param_grid(tmp_path):
    _seed(tmp_path, days=20)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 20, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy="orb",
        range_start_values=[0], range_end_values=[7],
        entry_end_values=[16], exit_hour_values=[20, 21],
        n_splits=2, bar_minutes=30, objective="sharpe",
    )
    assert out["param_grid"] == {
        "range_start_hour": [0], "range_end_hour": [7],
        "entry_end_hour": [16], "exit_hour": [20, 21],
    }
    assert "folds" in out and "mean_oos_objective" in out


def test_optimize_meanrev_grid_still_works(tmp_path):
    _seed(tmp_path, days=20)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 20, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy="meanrev",
        lookback_values=[20], entry_z_values=[2.0], exit_z_values=[0.5], er_threshold_values=[0.3],
        n_splits=2, bar_minutes=30,
    )
    assert out["param_grid"] == {"lookback": [20], "entry_z": [2.0], "exit_z": [0.5], "er_threshold": [0.3]}
