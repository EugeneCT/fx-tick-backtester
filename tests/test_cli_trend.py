"""CLI test: backtest --strategy trend end-to-end."""
import json
from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from fxtrader.cli import app, run_optimize
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore

runner = CliRunner()


def _price(i: int) -> float:
    # Dip then rise pattern so fast MA starts below slow MA and crosses above.
    # Same pattern as test_cli_backtest.py but extended for 600+ ticks so a
    # trend window of 5 bars can warm up before the cross fires.
    if i < 60:
        return 1.100
    if i < 120:
        return 1.099
    if i < 180:
        return 1.100 + (i - 120) * 0.00001
    return 1.100 + (i - 180) * 0.001


def _seed_store(tmp_path):
    store = TickStore(tmp_path)
    # 660 one-second ticks (≈11 one-minute bars) so trend MA(5) can warm up
    # and a fast(2)/slow(3) cross can fire.
    start = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    ticks = []
    for i in range(660):
        p = _price(i)
        ticks.append(Tick("EURUSD", start + timedelta(seconds=i), p, p + 0.00002))
    store.write_ticks("EURUSD", ticks)
    return store


def _seed_optimize(tmp_path):
    """Seed ~2.8 days of 30s oscillating ticks (800 five-minute bars) so trend=13 warms up."""
    store = TickStore(tmp_path)
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    ticks = []
    for i in range(8000):
        base = 1.10 + 0.01 * ((i // 400) % 3 - 1)  # oscillates to create crossovers
        ticks.append(Tick("EURUSD", start + timedelta(seconds=i * 30), base, base + 0.00002))
    store.write_ticks("EURUSD", ticks)


def test_optimize_trend_signal_param_grid(tmp_path):
    """run_optimize with strategy='trend' returns the correct param_grid and structure."""
    _seed_optimize(tmp_path)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        data_dir=tmp_path,
        strategy="trend",
        fast_values=[2, 3],
        slow_values=[5, 8],
        trend_values=[13],
        n_splits=2,
        bar_minutes=5,
        objective="sharpe",
        exit_kind="none",
    )
    assert out["param_grid"] == {"fast": [2, 3], "slow": [5, 8], "trend": [13]}
    assert "folds" in out
    assert "mean_oos_objective" in out
    # strictly JSON-serializable (no inf/nan tokens)
    json.dumps(out, allow_nan=False)


def test_optimize_trend_cli_runs(tmp_path):
    """optimize --strategy trend invoked via CLI succeeds and prints mean OOS."""
    _seed_optimize(tmp_path)
    result = runner.invoke(app, [
        "optimize", "--strategy", "trend",
        "--symbol", "EURUSD",
        "--start", "2024-01-02", "--end", "2024-01-05",
        "--fast", "2,3", "--slow", "5,8", "--trend", "13",
        "--exit", "none",
        "--n-splits", "2", "--bar-minutes", "5",
        "--data-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "mean OOS" in result.output


def test_backtest_trend_runs_end_to_end(tmp_path):
    _seed_store(tmp_path)
    result = runner.invoke(app, [
        "backtest",
        "--strategy", "trend",
        "--symbol", "EURUSD",
        "--start", "2024-01-02",
        "--end", "2024-01-03",
        "--fast", "2",
        "--slow", "3",
        "--trend", "5",
        "--exit", "fixed",
        "--stop-pips", "15",
        "--tp-pips", "30",
        "--bar-minutes", "1",
        "--data-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "trend" in result.output.lower()
