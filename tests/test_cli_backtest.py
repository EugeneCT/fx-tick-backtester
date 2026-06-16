from datetime import datetime, timezone

from fxtrader.cli import run_backtest
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore


def _price(i: int) -> float:
    # Minute closes form a small dip then a clear uptrend, so the fast MA
    # starts at/below the slow MA and crosses *above* it on 1-min bars. A
    # strictly-monotonic series cannot do this: the fast MA would already sit
    # above the slow MA on the first comparison, so no upward cross ever fires.
    if i < 60:
        return 1.100
    if i < 120:
        return 1.099
    if i < 180:
        return 1.100 + (i - 120) * 0.00001
    return 1.100 + (i - 180) * 0.001


def _seed_store(tmp_path):
    store = TickStore(tmp_path)
    # 200 one-second ticks (dip then trending up) so a fast/slow MA cross fires
    # on 1-min bars.
    start = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    ticks = []
    from datetime import timedelta
    for i in range(200):
        p = _price(i)
        ticks.append(Tick("EURUSD", start + timedelta(seconds=i), p, p + 0.00002))
    store.write_ticks("EURUSD", ticks)
    return store


def test_run_backtest_returns_result_and_metrics(tmp_path):
    _seed_store(tmp_path)
    result, metrics = run_backtest(
        symbol="EURUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 3, tzinfo=timezone.utc),
        data_dir=tmp_path,
        fast=2, slow=3, size=10_000,
        bar_minutes=1,
    )
    assert metrics["num_fills"] >= 1
    assert "sharpe" in metrics and "max_drawdown" in metrics
    assert result.final_equity > 0
