from datetime import datetime, timezone, timedelta

from typer.testing import CliRunner

from fxtrader.cli import app, run_backtest, run_optimize
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore

runner = CliRunner()


def _seed_uptrend(tmp_path, days=8):
    """Gently rising 30-min ticks (a clean uptrend) so the trend gate is satisfiable."""
    store = TickStore(tmp_path)
    ticks = []
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    px = 1.1000
    for slot in range(days * 48):
        ts = base + timedelta(minutes=30 * slot)
        px += 0.00005
        ticks.append(Tick("EURUSD", ts, px, px + 0.00002))
    store.write_ticks("EURUSD", ticks)


def test_backtest_pa_cli_runs_end_to_end(tmp_path):
    _seed_uptrend(tmp_path)
    res = runner.invoke(app, [
        "backtest", "--strategy", "pa", "--symbol", "EURUSD",
        "--start", "2024-01-01", "--end", "2024-01-09",
        "--pattern", "pin", "--trend", "5", "--bar-minutes", "30",
        "--exit", "atr", "--atr-period", "5", "--stop-atr", "1.5", "--tp-atr", "2.0",
        "--max-position", "50000", "--data-dir", str(tmp_path),
    ])
    assert res.exit_code == 0, res.output
    assert "pa" in res.output.lower()


def test_optimize_pa_grid(tmp_path):
    _seed_uptrend(tmp_path, days=20)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 20, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy="pa", pattern="pin",
        trend_values=[5], wick_ratio_values=[2.0], body_frac_values=[0.34],
        n_splits=2, bar_minutes=30, objective="sharpe",
    )
    assert out["param_grid"] == {"trend": [5], "wick_ratio": [2.0], "body_frac": [0.34]}
    assert "folds" in out and "mean_oos_objective" in out


def test_run_backtest_pa_end_to_end(tmp_path):
    _seed_uptrend(tmp_path)
    result, metrics = run_backtest(
        symbol="EURUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 9, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy_name="pa", pattern="pin", trend=5,
        bar_minutes=30, exit_kind="atr", atr_period=5, stop_atr=1.5, tp_atr=2.0,
        max_position=50_000,
    )
    # Smoke: completes, returns the standard metrics dict, account not ruined.
    assert "num_trades" in metrics and metrics["num_trades"] >= 0
    assert metrics["ruined"] is False
