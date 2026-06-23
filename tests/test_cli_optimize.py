import json
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
    # ~3 days of 30s ticks with a couple of dips so crosses fire across bars
    for i in range(8000):
        base = 1.10 + 0.01 * ((i // 400) % 3 - 1)  # oscillates to create crossovers
        ticks.append(Tick("EURUSD", start + timedelta(seconds=i * 30), base, base + 0.00002))
    store.write_ticks("EURUSD", ticks)


def test_run_optimize_returns_walk_forward_structure(tmp_path):
    _seed(tmp_path)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        data_dir=tmp_path,
        fast_values=[2, 3],
        slow_values=[5, 8],
        n_splits=2,
        bar_minutes=5,
        objective="sharpe",
    )
    assert out["objective"] == "sharpe"
    assert out["param_grid"] == {"fast": [2, 3], "slow": [5, 8]}
    assert "mean_oos_objective" in out
    # strictly JSON-serializable (no inf/nan tokens)
    text = json.dumps(out, allow_nan=False)
    assert "fast" in text


def test_run_optimize_invalid_combos_skipped(tmp_path):
    _seed(tmp_path)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 4, tzinfo=timezone.utc),
        data_dir=tmp_path,
        fast_values=[5],
        slow_values=[3],   # fast >= slow -> invalid -> no ranked params
        n_splits=1,
        bar_minutes=5,
        objective="sharpe",
    )
    # no valid params -> folds skipped
    assert out["folds"] == []


def test_optimize_rejects_unknown_strategy(tmp_path):
    result = runner.invoke(app, [
        "optimize", "--strategy", "bogus", "--symbol", "EURUSD",
        "--start", "2024-01-02", "--end", "2024-01-05",
        "--data-dir", str(tmp_path),
    ])
    assert result.exit_code != 0
    assert "strategy must be" in result.output.lower() or "bogus" in result.output.lower()
