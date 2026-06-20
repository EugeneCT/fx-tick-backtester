"""CLI test: backtest --strategy trend end-to-end."""
from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from fxtrader.cli import app
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
