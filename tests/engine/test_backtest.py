from datetime import datetime, timedelta, timezone

from fxtrader.broker.simulated import SimulatedBroker
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Side, Tick
from fxtrader.engine.backtest import BacktestEngine, BacktestResult
from fxtrader.risk.manager import RiskLimits, RiskManager
from fxtrader.strategy.base import Strategy


def _ticks(prices, start=datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc), step_s=1):
    # bid == ask for simple arithmetic (zero spread)
    return [Tick("EURUSD", start + timedelta(seconds=i * step_s), p, p)
            for i, p in enumerate(prices)]


class _BuyOnceThenSell(Strategy):
    """Buys on the 1st tick, sells on the 5th, via market orders."""

    def __init__(self):
        self.n = 0

    def on_tick(self, tick):
        self.n += 1
        if self.n == 1:
            self.order("EURUSD", Side.BUY, 10_000)
        elif self.n == 5:
            self.order("EURUSD", Side.SELL, 10_000)


def _engine(ticks, strategy, **kw):
    return BacktestEngine(
        feed=iter(ticks),
        strategy=strategy,
        broker=SimulatedBroker(commission_rate=0.0, min_commission=0.0),
        portfolio=Portfolio(100_000),
        risk=RiskManager(RiskLimits()),
        equity_interval=timedelta(seconds=1),
        **kw,
    )


def test_engine_runs_and_returns_result():
    ticks = _ticks([1.10, 1.11, 1.12, 1.13, 1.14, 1.15])
    result = _engine(ticks, _BuyOnceThenSell()).run()
    assert isinstance(result, BacktestResult)
    assert result.equity_curve  # non-empty
    assert len(result.fills) == 2  # one buy, one sell


def test_engine_profit_on_up_move():
    # buy at 1.10 (tick 1), sell at 1.14 (tick 5) -> +0.04 * 10_000 = +400
    ticks = _ticks([1.10, 1.11, 1.12, 1.13, 1.14, 1.15])
    result = _engine(ticks, _BuyOnceThenSell()).run()
    assert round(result.final_equity - 100_000, 2) == 400.00
    assert result.portfolio.position("EURUSD").quantity == 0


def test_engine_feeds_bars_to_strategy():
    seen_bars = []

    class BarWatcher(Strategy):
        def on_bar(self, bar):
            seen_bars.append(bar)

    # 1-second ticks across >2 minutes so at least 2 one-minute bars close
    ticks = _ticks([1.10 + i * 0.0001 for i in range(150)], step_s=1)
    eng = BacktestEngine(
        feed=iter(ticks), strategy=BarWatcher(),
        broker=SimulatedBroker(), portfolio=Portfolio(100_000),
        risk=RiskManager(RiskLimits()),
        bar_interval=timedelta(minutes=1), equity_interval=timedelta(minutes=1),
    )
    eng.run()
    assert len(seen_bars) >= 2
    assert all(b.symbol == "EURUSD" for b in seen_bars)


def test_kill_switch_flattens_and_halts():
    # Strategy keeps buying; price crashes; daily-loss kill switch must flatten and stop.
    class Buyer(Strategy):
        def on_tick(self, tick):
            self.order("EURUSD", Side.BUY, 10_000)

    prices = [1.10, 1.10, 0.50, 0.50, 0.50]  # massive drop after building a position
    ticks = _ticks(prices)
    eng = BacktestEngine(
        feed=iter(ticks), strategy=Buyer(),
        broker=SimulatedBroker(commission_rate=0.0, min_commission=0.0),
        portfolio=Portfolio(100_000),
        risk=RiskManager(RiskLimits(daily_loss_limit=1_000)),
        equity_interval=timedelta(seconds=1),
    )
    result = eng.run()
    # After the kill switch trips, the position is flattened to zero and stays there.
    assert result.portfolio.position("EURUSD").quantity == 0
    assert eng.risk.halted is True
