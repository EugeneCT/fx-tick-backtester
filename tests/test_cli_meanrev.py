from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from fxtrader.cli import app, run_optimize
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore

runner = CliRunner()


def _seed(tmp_path):
    store = TickStore(tmp_path)
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    ticks = []
    # 5000 oscillating 30s ticks (ranging) so the fader has stretched+reverting moves.
    # 5000 * 30s = 150000s ≈ 41.7h ≈ 1.7 days of data, enough to warm up lookback=20
    # across two folds (each fold covers ~0.85 days of bars).
    for i in range(5000):
        base = 1.10 + 0.0010 * ((i // 5) % 2)  # flip every 5 ticks -> mean-reverting
        ticks.append(Tick("EURUSD", start + timedelta(seconds=i * 30), base, base + 0.00002))
    store.write_ticks("EURUSD", ticks)


def test_optimize_meanrev_param_grid(tmp_path):
    _seed(tmp_path)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 3, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy="meanrev",
        lookback_values=[20], entry_z_values=[1.5, 2.0],
        exit_z_values=[0.3], er_threshold_values=[0.5],
        n_splits=2, bar_minutes=5, objective="sharpe",
    )
    assert out["param_grid"] == {
        "lookback": [20], "entry_z": [1.5, 2.0], "exit_z": [0.3], "er_threshold": [0.5],
    }
    assert "folds" in out and "mean_oos_objective" in out


def test_optimize_ma_grid_unchanged(tmp_path):
    _seed(tmp_path)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 3, tzinfo=timezone.utc),
        data_dir=tmp_path, fast_values=[2, 3], slow_values=[5, 8],
        n_splits=2, bar_minutes=5,
    )
    assert out["param_grid"] == {"fast": [2, 3], "slow": [5, 8]}


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
