import json
from datetime import datetime, timezone

from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Fill, Side, Tick
from fxtrader.engine.live import LiveEngine, TradingMode
from fxtrader.risk.manager import RiskLimits, RiskManager
from fxtrader.strategy.base import Strategy


class _FakeBroker:
    def __init__(self):
        self.submitted = []
        self.connected = False
        self._fill_cb = None
        self._tick_cb = None
    def set_fill_callback(self, cb): self._fill_cb = cb
    def set_tick_callback(self, cb): self._tick_cb = cb
    def connect(self): self.connected = True
    def disconnect(self): self.connected = False
    def update_market(self, tick): pass
    def submit(self, order): self.submitted.append(order)
    def cancel(self, order): pass
    def positions(self): return {}


class _BuyEveryTick(Strategy):
    def on_tick(self, tick):
        self.order("EURUSD", Side.BUY, 10_000)


def _tick(price, second=0):
    return Tick("EURUSD", datetime(2024, 1, 2, 9, 0, second, tzinfo=timezone.utc), price, price)


def _engine(strategy, mode, broker=None, risk=None, tmp_path=None, **kw):
    return LiveEngine(
        broker=broker or _FakeBroker(),
        strategy=strategy,
        portfolio=Portfolio(100_000),
        risk=risk or RiskManager(RiskLimits()),
        mode=mode,
        state_path=(tmp_path / "state.json") if tmp_path else None,
        **kw,
    )


def test_paper_mode_submits_orders():
    broker = _FakeBroker()
    eng = _engine(_BuyEveryTick(), TradingMode.PAPER, broker=broker)
    eng.start()
    eng._on_tick(_tick(1.10))
    assert len(broker.submitted) == 1


def test_dry_run_suppresses_orders():
    broker = _FakeBroker()
    eng = _engine(_BuyEveryTick(), TradingMode.DRY_RUN, broker=broker)
    eng.start()
    eng._on_tick(_tick(1.10))
    assert broker.submitted == []


def test_fill_updates_portfolio():
    eng = _engine(Strategy(), TradingMode.PAPER)
    eng.start()
    eng._on_fill(Fill("o", "EURUSD", Side.BUY, 10_000, 1.10,
                      datetime(2024, 1, 2, tzinfo=timezone.utc)))
    assert eng.portfolio.position("EURUSD").quantity == 10_000


def test_kill_switch_flattens_via_broker():
    broker = _FakeBroker()
    risk = RiskManager(RiskLimits(daily_loss_limit=1_000))
    eng = _engine(Strategy(), TradingMode.PAPER, broker=broker, risk=risk)
    eng.start()
    eng._on_fill(Fill("o", "EURUSD", Side.BUY, 100_000, 1.10,
                      datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)))
    eng._on_tick(_tick(1.0, second=1))  # -10_000 loss >> 1_000 limit
    assert eng.risk.halted is True
    assert any(o.side is Side.SELL for o in broker.submitted)


def test_state_persisted_on_tick(tmp_path):
    eng = _engine(Strategy(), TradingMode.PAPER, tmp_path=tmp_path)
    eng.start()
    eng._on_fill(Fill("o", "EURUSD", Side.BUY, 10_000, 1.10,
                      datetime(2024, 1, 2, tzinfo=timezone.utc)))
    eng._on_tick(_tick(1.11))
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["positions"]["EURUSD"] == 10_000
    assert "equity" in state and "halted" in state
