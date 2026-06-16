from datetime import datetime, timedelta, timezone

from fxtrader.broker.simulated import SimulatedBroker
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Side, Tick
from fxtrader.engine.backtest import BacktestEngine
from fxtrader.metrics.performance import compute_metrics
from fxtrader.risk.manager import RiskLimits, RiskManager
from fxtrader.strategy.base import Strategy


class _OneRoundTrip(Strategy):
    """Deterministic: buy 10k on tick 2, sell 10k on tick 8."""

    def __init__(self):
        self.n = 0

    def on_tick(self, tick):
        self.n += 1
        if self.n == 2:
            self.order("EURUSD", Side.BUY, 10_000)
        elif self.n == 8:
            self.order("EURUSD", Side.SELL, 10_000)


def test_full_stack_golden_path():
    start = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    # zero-spread, +0.0001 per tick: buy at tick2 price, sell at tick8 price
    prices = [1.1000 + i * 0.0001 for i in range(10)]
    ticks = [Tick("EURUSD", start + timedelta(seconds=i), p, p) for i, p in enumerate(prices)]

    engine = BacktestEngine(
        feed=iter(ticks),
        strategy=_OneRoundTrip(),
        broker=SimulatedBroker(commission_rate=0.0, min_commission=0.0),
        portfolio=Portfolio(100_000),
        risk=RiskManager(RiskLimits()),
        equity_interval=timedelta(seconds=1),
    )
    result = engine.run()
    metrics = compute_metrics(result.equity_curve, result.fills, periods_per_year=8760)

    # buy at prices[1]=1.1001 (tick2), sell at prices[7]=1.1007 (tick8) -> +0.0006 * 10_000 = 6.0
    assert len(result.fills) == 2
    assert round(result.final_equity - 100_000, 4) == 6.0
    assert metrics["num_trades"] == 1
    assert metrics["win_rate"] == 1.0
    assert result.portfolio.position("EURUSD").quantity == 0
