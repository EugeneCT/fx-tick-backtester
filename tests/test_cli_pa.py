from datetime import datetime, timezone, timedelta

from fxtrader.cli import run_backtest
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore


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
