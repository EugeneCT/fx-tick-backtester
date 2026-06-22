from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from fxtrader.cli import app, run_optimize
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


def test_optimize_xpair_param_grid(tmp_path):
    _seed(tmp_path, n=4000)
    out = run_optimize(
        symbol="EURUSD", symbol2="GBPUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 4, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy="xpair",
        lookback_values=[20], entry_z_values=[1.5, 2.0], exit_z_values=[0.3], stop_z_values=[4.0],
        n_splits=2, bar_minutes=5, objective="sharpe",
    )
    assert out["param_grid"] == {
        "lookback": [20], "entry_z": [1.5, 2.0], "exit_z": [0.3], "stop_z": [4.0],
    }
    assert "folds" in out and "mean_oos_objective" in out


def test_optimize_orb_grid_still_works(tmp_path):
    _seed(tmp_path, n=4000)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 4, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy="orb",
        range_start_values=[0], range_end_values=[7], entry_end_values=[16], exit_hour_values=[21],
        n_splits=2, bar_minutes=60,
    )
    assert out["param_grid"] == {"range_start_hour": [0], "range_end_hour": [7],
                                 "entry_end_hour": [16], "exit_hour": [21]}


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
