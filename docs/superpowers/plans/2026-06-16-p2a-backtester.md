# P2a — Backtester Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run a strategy over stored ticks through an event-driven backtest engine with realistic simulated fills, risk controls, and performance metrics — and expose it as `fxtrader backtest`.

**Architecture:** One `Strategy` base class emits `Order`s; a `RiskManager` vets them; a `Broker` fills them. `SimulatedBroker` fills market orders at ask/bid (+slippage) with an IB-like commission. The `BacktestEngine` streams `Tick`s from a `HistoricalTickFeed`, advances a `SimClock`, resamples ticks to bars (for bar-based strategies), routes orders order→risk→broker→fill→portfolio, samples the equity curve, and returns a `BacktestResult`. `compute_metrics` turns the equity curve + fills into Sharpe/Sortino/drawdown/trade stats. The reference `MaCrossover` strategy and a `backtest` CLI tie it together. This is the SAME `Strategy`/`Broker`/`RiskManager` surface that P3 (live/IB) and P4 (research agent) will reuse.

**Tech Stack:** Python 3.12 (`.venv`), stdlib only for the engine/metrics (math, collections, dataclasses), typer for the CLI, polars only via the existing feed/store, pytest.

**Conventions:**
- All timestamps tz-aware UTC. Use `.venv/bin/pytest` / `.venv/bin/python` (shell `source` doesn't persist).
- Reuse existing types: `Tick`, `Bar`, `Order`, `Fill`, `Position`, `Side`, `OrderType` from `fxtrader.core.types`; `Portfolio` from `fxtrader.core.portfolio`; `SimClock` from `fxtrader.core.clock`; `HistoricalTickFeed`/`Resampler` from `fxtrader.data.feed`; `TickStore` from `fxtrader.data.store`.
- Position quantities are signed base units (e.g. +10_000 EURUSD = long 10k EUR). Commission is in account currency.
- Commit after each task with the message in its final step. No placeholders — every code/test step is complete.

---

## File structure

```
src/fxtrader/strategy/__init__.py          # Task 1
src/fxtrader/strategy/base.py              # Task 1  — Strategy
src/fxtrader/broker/__init__.py            # Task 2
src/fxtrader/broker/base.py               # Task 2  — Broker ABC
src/fxtrader/broker/simulated.py          # Task 3  — SimulatedBroker
src/fxtrader/risk/__init__.py              # Task 4
src/fxtrader/risk/manager.py              # Task 4  — RiskLimits, RiskManager
src/fxtrader/metrics/__init__.py           # Task 5
src/fxtrader/metrics/performance.py       # Task 5  — compute_metrics
src/fxtrader/engine/__init__.py            # Task 6
src/fxtrader/engine/backtest.py           # Task 6  — BacktestEngine, BacktestResult
src/fxtrader/strategy/examples/__init__.py # Task 7
src/fxtrader/strategy/examples/ma_crossover.py  # Task 7 — MaCrossover
src/fxtrader/cli.py                         # Task 8  — backtest command (MODIFY)
tests/...                                   # one test module per source module
```

---

## Task 1: Strategy base class

**Files:**
- Create: `src/fxtrader/strategy/__init__.py`
- Create: `src/fxtrader/strategy/base.py`
- Create: `tests/strategy/test_base.py`

A `Strategy` holds no I/O. The engine calls `bind(clock, portfolio, submit)` to wire it, then drives lifecycle hooks. `order(...)` builds an `Order` and hands it to the engine's `submit` sink. All hooks default to no-ops.

- [ ] **Step 1: Write the failing test `tests/strategy/test_base.py`**

```python
from datetime import datetime, timezone

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Order, OrderType, Side, Tick
from fxtrader.strategy.base import Strategy


class _Spy(Strategy):
    def __init__(self):
        self.ticks = []
        self.started = False

    def on_start(self):
        self.started = True

    def on_tick(self, tick):
        self.ticks.append(tick)
        self.order(tick.symbol, Side.BUY, 10_000)


def _bind(strategy):
    clock = SimClock()
    portfolio = Portfolio(100_000)
    sent = []
    strategy.bind(clock, portfolio, sent.append)
    return clock, portfolio, sent


def test_bind_exposes_clock_and_portfolio():
    s = _Spy()
    clock, portfolio, _ = _bind(s)
    assert s.clock is clock
    assert s.portfolio is portfolio


def test_order_helper_builds_and_submits_order():
    s = _Spy()
    _, _, sent = _bind(s)
    tick = Tick("EURUSD", datetime(2024, 1, 2, tzinfo=timezone.utc), 1.1, 1.1002)
    s.on_tick(tick)
    assert len(sent) == 1
    order = sent[0]
    assert order.symbol == "EURUSD"
    assert order.side is Side.BUY
    assert order.quantity == 10_000
    assert order.order_type is OrderType.MARKET


def test_default_hooks_are_noops():
    s = Strategy()
    _bind(s)
    # None of these should raise
    s.on_start()
    s.on_bar(None)
    s.on_fill(None)
    s.on_stop()


def test_on_start_hook_runs():
    s = _Spy()
    _bind(s)
    s.on_start()
    assert s.started is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/strategy/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.strategy'`

- [ ] **Step 3: Create `src/fxtrader/strategy/__init__.py`**

```python
"""Trading strategies and the Strategy base class."""
```

- [ ] **Step 4: Write `src/fxtrader/strategy/base.py`**

```python
"""Strategy base class — pure logic, no I/O. The engine binds it to a context."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from fxtrader.core.clock import Clock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Fill, Order, OrderType, Side, Tick

OrderSink = Callable[[Order], None]


class Strategy:
    """Subclass and override the on_* hooks. Use self.order(...) to trade.

    The engine calls bind() before any hook. self.clock / self.portfolio are
    read-only views; never mutate the portfolio directly.
    """

    clock: Clock
    portfolio: Portfolio
    _submit: OrderSink

    def bind(self, clock: Clock, portfolio: Portfolio, submit: OrderSink) -> None:
        self.clock = clock
        self.portfolio = portfolio
        self._submit = submit

    def order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        *,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> None:
        self._submit(
            Order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
            )
        )

    # Lifecycle hooks — override as needed. Defaults are no-ops.
    def on_start(self) -> None: ...
    def on_tick(self, tick: Tick) -> None: ...
    def on_bar(self, bar: Bar) -> None: ...
    def on_fill(self, fill: Fill) -> None: ...
    def on_stop(self) -> None: ...
```

(Note: the imports of `datetime` is unused — remove it. Keep only what the file uses: `Callable`, `Clock`, `Portfolio`, `Bar`, `Fill`, `Order`, `OrderType`, `Side`, `Tick`. Run `.venv/bin/ruff check src/fxtrader/strategy/base.py` and remove any unused import it flags.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/strategy/test_base.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add src/fxtrader/strategy/__init__.py src/fxtrader/strategy/base.py tests/strategy/test_base.py
git commit -m "feat: Strategy base class (bind + order helper + lifecycle hooks)"
```

---

## Task 2: Broker interface

**Files:**
- Create: `src/fxtrader/broker/__init__.py`
- Create: `src/fxtrader/broker/base.py`
- Create: `tests/broker/test_base.py`

`Broker` is the abstraction both `SimulatedBroker` (backtest) and the future `IBBroker` (live) implement. It receives market updates, accepts orders, and delivers fills through a callback.

- [ ] **Step 1: Write the failing test `tests/broker/test_base.py`**

```python
import pytest

from fxtrader.broker.base import Broker


def test_broker_is_abstract():
    with pytest.raises(TypeError):
        Broker()  # abstract methods not implemented


def test_concrete_subclass_can_register_and_invoke_fill_callback():
    class Dummy(Broker):
        def update_market(self, tick):
            pass

        def submit(self, order):
            self._emit_fill("a-fill")

    got = []
    b = Dummy()
    b.set_fill_callback(got.append)
    b.submit(object())
    assert got == ["a-fill"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/broker/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.broker'`

- [ ] **Step 3: Create `src/fxtrader/broker/__init__.py`**

```python
"""Broker abstractions: simulated (backtest) and IB (live)."""
```

- [ ] **Step 4: Write `src/fxtrader/broker/base.py`**

```python
"""Broker interface shared by the backtest and live engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from fxtrader.core.types import Fill, Order, Tick

FillCallback = Callable[[Fill], None]


class Broker(ABC):
    """Receives market data, accepts orders, emits fills via a callback."""

    _on_fill: FillCallback | None = None

    def set_fill_callback(self, callback: FillCallback) -> None:
        self._on_fill = callback

    def _emit_fill(self, fill: Fill) -> None:
        if self._on_fill is not None:
            self._on_fill(fill)

    @abstractmethod
    def update_market(self, tick: Tick) -> None:
        """Inform the broker of the latest market for a symbol."""

    @abstractmethod
    def submit(self, order: Order) -> None:
        """Accept an order; fills are delivered asynchronously via the callback."""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/broker/test_base.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/fxtrader/broker/__init__.py src/fxtrader/broker/base.py tests/broker/test_base.py
git commit -m "feat: Broker interface (market update, submit, fill callback)"
```

---

## Task 3: SimulatedBroker

**Files:**
- Create: `src/fxtrader/broker/simulated.py`
- Create: `tests/broker/test_simulated.py`

Fills MARKET orders immediately against the symbol's last tick: BUY at `ask + slippage`, SELL at `bid - slippage`. Commission = `max(min_commission, notional * commission_rate)`, notional = `quantity * fill_price`. Fill timestamp = the symbol's last tick time. Limit/stop orders are out of scope for P2a (raise `NotImplementedError`).

- [ ] **Step 1: Write the failing test `tests/broker/test_simulated.py`**

```python
from datetime import datetime, timezone

import pytest

from fxtrader.broker.simulated import SimulatedBroker
from fxtrader.core.types import Order, OrderType, Side, Tick


def _tick(bid, ask, minute=0):
    return Tick("EURUSD", datetime(2024, 1, 2, 9, minute, tzinfo=timezone.utc), bid, ask)


def test_buy_fills_at_ask_plus_slippage():
    b = SimulatedBroker(commission_rate=0.0, min_commission=0.0, slippage=0.00001)
    fills = []
    b.set_fill_callback(fills.append)
    b.update_market(_tick(1.10000, 1.10020))
    b.submit(Order("EURUSD", Side.BUY, 10_000))
    assert len(fills) == 1
    assert round(fills[0].price, 5) == round(1.10020 + 0.00001, 5)
    assert fills[0].quantity == 10_000
    assert fills[0].ts == datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)


def test_sell_fills_at_bid_minus_slippage():
    b = SimulatedBroker(commission_rate=0.0, min_commission=0.0, slippage=0.00001)
    fills = []
    b.set_fill_callback(fills.append)
    b.update_market(_tick(1.10000, 1.10020))
    b.submit(Order("EURUSD", Side.SELL, 10_000))
    assert round(fills[0].price, 5) == round(1.10000 - 0.00001, 5)


def test_commission_is_bps_of_notional_with_floor():
    b = SimulatedBroker(commission_rate=0.00002, min_commission=2.0)
    fills = []
    b.set_fill_callback(fills.append)
    b.update_market(_tick(1.10000, 1.10000))
    # notional = 1_000_000 * 1.10 = 1_100_000; comm = 1_100_000 * 0.00002 = 22.0 (> floor 2.0)
    b.submit(Order("EURUSD", Side.BUY, 1_000_000))
    assert round(fills[0].commission, 4) == 22.0
    # small order hits the floor: notional = 1000*1.1=1100; *0.00002=0.022 -> floor 2.0
    b.submit(Order("EURUSD", Side.BUY, 1_000))
    assert fills[1].commission == 2.0


def test_market_order_without_quote_raises():
    b = SimulatedBroker()
    b.set_fill_callback(lambda f: None)
    with pytest.raises(KeyError):
        b.submit(Order("EURUSD", Side.BUY, 10_000))


def test_limit_order_not_supported():
    b = SimulatedBroker()
    b.set_fill_callback(lambda f: None)
    b.update_market(_tick(1.1, 1.1002))
    with pytest.raises(NotImplementedError):
        b.submit(Order("EURUSD", Side.BUY, 10_000, order_type=OrderType.LIMIT, limit_price=1.1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/broker/test_simulated.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.broker.simulated'`

- [ ] **Step 3: Write `src/fxtrader/broker/simulated.py`**

```python
"""SimulatedBroker — immediate market-order fills with slippage + commission."""

from __future__ import annotations

from fxtrader.broker.base import Broker
from fxtrader.core.types import Fill, Order, OrderType, Side, Tick


class SimulatedBroker(Broker):
    def __init__(
        self,
        *,
        commission_rate: float = 0.00002,  # 0.2 bps of notional
        min_commission: float = 2.0,
        slippage: float = 0.0,  # price units added to the touch
    ) -> None:
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self._market: dict[str, Tick] = {}
        self._next_id = 0

    def update_market(self, tick: Tick) -> None:
        self._market[tick.symbol] = tick

    def submit(self, order: Order) -> None:
        if order.order_type is not OrderType.MARKET:
            raise NotImplementedError("SimulatedBroker supports MARKET orders only (P2a)")
        tick = self._market[order.symbol]  # KeyError if no quote yet — intentional
        if order.side is Side.BUY:
            price = tick.ask + self.slippage
        else:
            price = tick.bid - self.slippage
        notional = order.quantity * price
        commission = max(self.min_commission, notional * self.commission_rate)
        self._next_id += 1
        self._emit_fill(
            Fill(
                order_id=order.order_id or f"sim-{self._next_id}",
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=price,
                ts=tick.ts,
                commission=commission,
            )
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/broker/test_simulated.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/broker/simulated.py tests/broker/test_simulated.py
git commit -m "feat: SimulatedBroker (market fills at ask/bid + slippage + commission)"
```

---

## Task 4: RiskManager

**Files:**
- Create: `src/fxtrader/risk/__init__.py`
- Create: `src/fxtrader/risk/manager.py`
- Create: `tests/risk/test_manager.py`

`check(order, portfolio, prices, clock)` approves/rejects an order against limits: max absolute position per symbol, max gross notional across symbols, and max orders per trailing 60 seconds. `update_equity(equity, clock)` tracks day-start equity and trips `halted` when the daily loss limit is breached (kill switch). A halted manager rejects everything.

- [ ] **Step 1: Write the failing test `tests/risk/test_manager.py`**

```python
from datetime import datetime, timedelta, timezone

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Fill, Order, Side
from fxtrader.risk.manager import RiskLimits, RiskManager


def _clock(ts):
    c = SimClock()
    c.set_time(ts)
    return c


def _portfolio_long(qty, price=1.10):
    p = Portfolio(100_000)
    p.apply_fill(Fill("o", "EURUSD", Side.BUY, qty, price,
                      datetime(2024, 1, 2, tzinfo=timezone.utc)))
    return p


T0 = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)


def test_allows_order_within_limits():
    rm = RiskManager(RiskLimits(max_position=50_000))
    p = Portfolio(100_000)
    assert rm.check(Order("EURUSD", Side.BUY, 10_000), p, {"EURUSD": 1.10}, _clock(T0)) is True


def test_rejects_order_exceeding_max_position():
    rm = RiskManager(RiskLimits(max_position=15_000))
    p = _portfolio_long(10_000)
    # +10_000 already; another +10_000 -> 20_000 > 15_000
    assert rm.check(Order("EURUSD", Side.BUY, 10_000), p, {"EURUSD": 1.10}, _clock(T0)) is False


def test_rejects_order_exceeding_max_gross_notional():
    rm = RiskManager(RiskLimits(max_gross_notional=20_000))
    p = Portfolio(100_000)
    # 10_000 * 1.10 = 11_000 ok; 20_000 * 1.10 = 22_000 > 20_000
    assert rm.check(Order("EURUSD", Side.BUY, 10_000), p, {"EURUSD": 1.10}, _clock(T0)) is True
    assert rm.check(Order("EURUSD", Side.BUY, 20_000), p, {"EURUSD": 1.10}, _clock(T0)) is False


def test_rejects_when_orders_per_minute_exceeded():
    rm = RiskManager(RiskLimits(max_orders_per_minute=2))
    p = Portfolio(100_000)
    prices = {"EURUSD": 1.10}
    assert rm.check(Order("EURUSD", Side.BUY, 1_000), p, prices, _clock(T0)) is True
    assert rm.check(Order("EURUSD", Side.BUY, 1_000), p, prices, _clock(T0)) is True
    assert rm.check(Order("EURUSD", Side.BUY, 1_000), p, prices, _clock(T0)) is False
    # 61s later the window has cleared
    later = _clock(T0 + timedelta(seconds=61))
    assert rm.check(Order("EURUSD", Side.BUY, 1_000), p, prices, later) is True


def test_daily_loss_limit_trips_kill_switch():
    rm = RiskManager(RiskLimits(daily_loss_limit=500))
    c = _clock(T0)
    rm.update_equity(100_000, c)  # day start
    assert rm.halted is False
    rm.update_equity(99_600, c)  # -400, still ok
    assert rm.halted is False
    rm.update_equity(99_400, c)  # -600 -> breach
    assert rm.halted is True
    # halted rejects everything
    assert rm.check(Order("EURUSD", Side.SELL, 1_000), Portfolio(100_000), {"EURUSD": 1.1}, c) is False


def test_day_rollover_resets_day_start_equity():
    rm = RiskManager(RiskLimits(daily_loss_limit=500))
    rm.update_equity(100_000, _clock(T0))
    # next day, lower equity becomes the new baseline
    next_day = _clock(T0 + timedelta(days=1))
    rm.update_equity(99_000, next_day)
    assert rm.halted is False
    rm.update_equity(98_600, next_day)  # -400 from new baseline
    assert rm.halted is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/risk/test_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.risk'`

- [ ] **Step 3: Create `src/fxtrader/risk/__init__.py`**

```python
"""Risk management: limits and the kill switch, enforced in backtest and live."""
```

- [ ] **Step 4: Write `src/fxtrader/risk/manager.py`**

```python
"""RiskManager — order vetting + daily-loss kill switch, identical in backtest and live."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from fxtrader.core.clock import Clock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Order, Side


@dataclass
class RiskLimits:
    max_position: float = float("inf")          # abs base units per symbol
    max_gross_notional: float = float("inf")    # sum over symbols of |qty| * price
    daily_loss_limit: float = float("inf")      # positive; halt when day P&L <= -limit
    max_orders_per_minute: int = 1_000_000


class RiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self.halted = False
        self._order_times: deque = deque()
        self._day = None
        self._day_start_equity: float | None = None

    def update_equity(self, equity: float, clock: Clock) -> None:
        day = clock.now().date()
        if day != self._day:
            self._day = day
            self._day_start_equity = equity
        if self._day_start_equity is not None:
            if equity - self._day_start_equity <= -self.limits.daily_loss_limit:
                self.halted = True

    def check(self, order: Order, portfolio: Portfolio, prices: dict, clock: Clock) -> bool:
        if self.halted:
            return False

        now = clock.now()
        while self._order_times and (now - self._order_times[0]).total_seconds() > 60:
            self._order_times.popleft()
        if len(self._order_times) >= self.limits.max_orders_per_minute:
            return False

        signed = order.quantity if order.side is Side.BUY else -order.quantity

        projected = {s: p.quantity for s, p in portfolio.positions.items()}
        projected[order.symbol] = projected.get(order.symbol, 0.0) + signed

        if abs(projected[order.symbol]) > self.limits.max_position:
            return False

        gross = sum(abs(q) * prices[s] for s, q in projected.items() if s in prices)
        if gross > self.limits.max_gross_notional:
            return False

        self._order_times.append(now)
        return True
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/risk/test_manager.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add src/fxtrader/risk/__init__.py src/fxtrader/risk/manager.py tests/risk/test_manager.py
git commit -m "feat: RiskManager (position/gross/rate limits + daily-loss kill switch)"
```

---

## Task 5: Performance metrics

**Files:**
- Create: `src/fxtrader/metrics/__init__.py`
- Create: `src/fxtrader/metrics/performance.py`
- Create: `tests/metrics/test_performance.py`

`compute_metrics(equity_curve, fills, *, periods_per_year)` returns a dict of metrics. `equity_curve` is `list[tuple[datetime, float]]` sampled at a fixed interval. Trade outcomes are reconstructed from fills (each position-reducing fill realizes a trade PnL chunk).

- [ ] **Step 1: Write the failing test `tests/metrics/test_performance.py`**

```python
import math
from datetime import datetime, timedelta, timezone

from fxtrader.core.types import Fill, Side
from fxtrader.metrics.performance import compute_metrics, trade_pnls


def _curve(values, start=datetime(2024, 1, 2, tzinfo=timezone.utc), step_hours=1):
    return [(start + timedelta(hours=i * step_hours), v) for i, v in enumerate(values)]


def test_total_return_and_final_equity():
    m = compute_metrics(_curve([100_000, 101_000, 102_000]), [], periods_per_year=8760)
    assert round(m["total_return"], 5) == round(102_000 / 100_000 - 1, 5)
    assert m["final_equity"] == 102_000


def test_max_drawdown():
    # peak 110, trough 99 -> dd = (99-110)/110 = -0.1
    m = compute_metrics(_curve([100, 110, 99, 105]), [], periods_per_year=252)
    assert round(m["max_drawdown"], 5) == round(-11 / 110, 5)


def test_sharpe_positive_for_steady_gains():
    m = compute_metrics(_curve([100, 101, 102, 103, 104]), [], periods_per_year=252)
    assert m["sharpe"] > 0
    assert math.isfinite(m["sharpe"])


def test_flat_curve_has_zero_sharpe_and_drawdown():
    m = compute_metrics(_curve([100, 100, 100]), [], periods_per_year=252)
    assert m["sharpe"] == 0.0
    assert m["max_drawdown"] == 0.0


def test_trade_pnls_from_round_trip_fills():
    ts = datetime(2024, 1, 2, tzinfo=timezone.utc)
    fills = [
        Fill("1", "EURUSD", Side.BUY, 10_000, 1.1000, ts),
        Fill("2", "EURUSD", Side.SELL, 10_000, 1.1010, ts),   # +10 win
        Fill("3", "EURUSD", Side.BUY, 10_000, 1.2000, ts),
        Fill("4", "EURUSD", Side.SELL, 10_000, 1.1990, ts),   # -10 loss
    ]
    pnls = trade_pnls(fills)
    assert [round(p, 4) for p in pnls] == [10.0, -10.0]


def test_trade_stats_win_rate_and_profit_factor():
    ts = datetime(2024, 1, 2, tzinfo=timezone.utc)
    fills = [
        Fill("1", "EURUSD", Side.BUY, 10_000, 1.1000, ts),
        Fill("2", "EURUSD", Side.SELL, 10_000, 1.1010, ts),   # +10
        Fill("3", "EURUSD", Side.BUY, 10_000, 1.2000, ts),
        Fill("4", "EURUSD", Side.SELL, 10_000, 1.1990, ts),   # -10
        Fill("5", "EURUSD", Side.BUY, 10_000, 1.3000, ts),
        Fill("6", "EURUSD", Side.SELL, 10_000, 1.3030, ts),   # +30
    ]
    m = compute_metrics(_curve([100_000, 100_030]), fills, periods_per_year=8760)
    assert m["num_trades"] == 3
    assert round(m["win_rate"], 4) == round(2 / 3, 4)
    # profit factor = (10+30) / 10 = 4.0
    assert round(m["profit_factor"], 4) == 4.0


def test_empty_curve_is_safe():
    m = compute_metrics([], [], periods_per_year=252)
    assert m["num_trades"] == 0
    assert m["sharpe"] == 0.0
    assert m["total_return"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/metrics/test_performance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.metrics'`

- [ ] **Step 3: Create `src/fxtrader/metrics/__init__.py`**

```python
"""Performance metrics and reporting."""
```

- [ ] **Step 4: Write `src/fxtrader/metrics/performance.py`**

```python
"""Backtest performance metrics derived from an equity curve and fills."""

from __future__ import annotations

import math
from datetime import datetime

from fxtrader.core.types import Fill, Side

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def trade_pnls(fills: list[Fill]) -> list[float]:
    """Realized price-PnL per position-reducing fill (one entry per closing event)."""
    qty: dict[str, float] = {}
    avg: dict[str, float] = {}
    out: list[float] = []
    for f in fills:
        signed = f.quantity if f.side is Side.BUY else -f.quantity
        q = qty.get(f.symbol, 0.0)
        a = avg.get(f.symbol, 0.0)
        if q != 0 and (q > 0) != (signed > 0):
            closing = min(abs(signed), abs(q))
            direction = 1.0 if q > 0 else -1.0
            out.append((f.price - a) * closing * direction)
        new_q = q + signed
        if new_q == 0:
            a = 0.0
        elif q == 0 or (q > 0) == (signed > 0):
            a = (a * abs(q) + f.price * abs(signed)) / abs(new_q)
        elif abs(signed) > abs(q):
            a = f.price
        qty[f.symbol] = new_q
        avg[f.symbol] = a
    return out


def _zeroed() -> dict:
    return {
        "total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "max_drawdown": 0.0, "num_trades": 0, "win_rate": 0.0,
        "profit_factor": 0.0, "final_equity": 0.0, "num_fills": 0,
    }


def compute_metrics(
    equity_curve: list[tuple[datetime, float]],
    fills: list[Fill],
    *,
    periods_per_year: float,
) -> dict:
    metrics = _zeroed()
    metrics["num_fills"] = len(fills)

    pnls = trade_pnls(fills)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    metrics["num_trades"] = len(pnls)
    metrics["win_rate"] = len(wins) / len(pnls) if pnls else 0.0
    if losses:
        metrics["profit_factor"] = sum(wins) / abs(sum(losses))
    elif wins:
        metrics["profit_factor"] = float("inf")

    if len(equity_curve) < 2:
        if equity_curve:
            metrics["final_equity"] = equity_curve[-1][1]
        return metrics

    eqs = [e for _, e in equity_curve]
    metrics["final_equity"] = eqs[-1]
    if eqs[0] > 0:
        metrics["total_return"] = eqs[-1] / eqs[0] - 1

    rets = [eqs[i] / eqs[i - 1] - 1 for i in range(1, len(eqs)) if eqs[i - 1] != 0]
    if rets:
        n = len(rets)
        mean = sum(rets) / n
        std = (sum((r - mean) ** 2 for r in rets) / n) ** 0.5
        if std > 0:
            metrics["sharpe"] = mean / std * math.sqrt(periods_per_year)
        downside = (sum(min(r, 0.0) ** 2 for r in rets) / n) ** 0.5
        if downside > 0:
            metrics["sortino"] = mean / downside * math.sqrt(periods_per_year)

    peak = eqs[0]
    max_dd = 0.0
    for e in eqs:
        peak = max(peak, e)
        if peak > 0:
            max_dd = min(max_dd, (e - peak) / peak)
    metrics["max_drawdown"] = max_dd

    span = (equity_curve[-1][0] - equity_curve[0][0]).total_seconds()
    years = max(span / _SECONDS_PER_YEAR, 1e-9)
    if eqs[0] > 0 and eqs[-1] > 0:
        metrics["cagr"] = (eqs[-1] / eqs[0]) ** (1 / years) - 1

    return metrics
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/metrics/test_performance.py -v`
Expected: PASS (8 passed)

- [ ] **Step 6: Commit**

```bash
git add src/fxtrader/metrics/__init__.py src/fxtrader/metrics/performance.py tests/metrics/test_performance.py
git commit -m "feat: performance metrics (return/Sharpe/Sortino/drawdown/trade stats)"
```

---

## Task 6: BacktestEngine

**Files:**
- Create: `src/fxtrader/engine/__init__.py`
- Create: `src/fxtrader/engine/backtest.py`
- Create: `tests/engine/test_backtest.py`

The engine streams ticks, advances the clock, resamples to bars, routes orders through risk→broker, applies fills to the portfolio, samples the equity curve, and runs the kill-switch flatten. It returns a `BacktestResult`.

- [ ] **Step 1: Write the failing test `tests/engine/test_backtest.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/engine/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.engine'`

- [ ] **Step 3: Create `src/fxtrader/engine/__init__.py`**

```python
"""Backtest and live execution engines."""
```

- [ ] **Step 4: Write `src/fxtrader/engine/backtest.py`**

```python
"""Event-driven backtest engine: ticks -> strategy -> risk -> broker -> portfolio."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from fxtrader.broker.base import Broker
from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Order, Side, Tick
from fxtrader.data.feed import Resampler
from fxtrader.risk.manager import RiskManager
from fxtrader.strategy.base import Strategy


@dataclass
class BacktestResult:
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    fills: list = field(default_factory=list)
    final_equity: float = 0.0
    portfolio: Portfolio | None = None


class BacktestEngine:
    def __init__(
        self,
        feed: Iterable[Tick],
        strategy: Strategy,
        broker: Broker,
        portfolio: Portfolio,
        risk: RiskManager,
        *,
        equity_interval: timedelta = timedelta(hours=1),
        bar_interval: timedelta | None = None,
    ) -> None:
        self.feed = feed
        self.strategy = strategy
        self.broker = broker
        self.portfolio = portfolio
        self.risk = risk
        self.equity_interval = equity_interval
        self.bar_interval = bar_interval
        self.clock = SimClock()
        self.equity_curve: list[tuple[datetime, float]] = []
        self._last_price: dict[str, float] = {}
        self._resamplers: dict[str, Resampler] = {}
        self._flattened = False
        self._eq_bucket: datetime | None = None

    def _resampler(self, symbol: str) -> Resampler:
        rs = self._resamplers.get(symbol)
        if rs is None:
            rs = Resampler(self.bar_interval)
            self._resamplers[symbol] = rs
        return rs

    def _on_order(self, order: Order) -> None:
        if self.risk.halted:
            return
        if order.symbol not in self._last_price:
            return
        if self.risk.check(order, self.portfolio, self._last_price, self.clock):
            self.broker.submit(order)

    def _on_fill(self, fill) -> None:
        self.portfolio.apply_fill(fill)
        self.strategy.on_fill(fill)

    def _flatten(self) -> None:
        for symbol, pos in list(self.portfolio.positions.items()):
            if pos.quantity != 0 and symbol in self._last_price:
                side = Side.SELL if pos.quantity > 0 else Side.BUY
                self.broker.submit(Order(symbol, side, abs(pos.quantity)))
        self._flattened = True

    def _floor(self, ts: datetime) -> datetime:
        step = self.equity_interval.total_seconds()
        epoch = ts.timestamp()
        return datetime.fromtimestamp(epoch - (epoch % step), tz=timezone.utc)

    def run(self) -> BacktestResult:
        self.strategy.bind(self.clock, self.portfolio, self._on_order)
        self.broker.set_fill_callback(self._on_fill)
        self.strategy.on_start()

        for tick in self.feed:
            self.clock.set_time(tick.ts)
            self._last_price[tick.symbol] = tick.mid
            self.broker.update_market(tick)

            equity = self.portfolio.equity(self._last_price)
            self.risk.update_equity(equity, self.clock)
            if self.risk.halted:
                if not self._flattened:
                    self._flatten()
            else:
                # Not halted (incl. after the daily kill switch re-arms next session):
                # clear the flatten latch so a future breach can flatten again.
                self._flattened = False
                self.strategy.on_tick(tick)
                if self.bar_interval is not None:
                    bar = self._resampler(tick.symbol).update(tick)
                    if bar is not None:
                        self.strategy.on_bar(bar)

            bucket = self._floor(tick.ts)
            if bucket != self._eq_bucket:
                self.equity_curve.append((bucket, equity))
                self._eq_bucket = bucket

        if self.bar_interval is not None and not self.risk.halted:
            for rs in self._resamplers.values():
                bar = rs.flush()
                if bar is not None:
                    self.strategy.on_bar(bar)

        self.strategy.on_stop()
        final_equity = self.portfolio.equity(self._last_price)
        self.equity_curve.append((self.clock.now(), final_equity))
        return BacktestResult(
            equity_curve=self.equity_curve,
            fills=list(self.portfolio.fills),
            final_equity=final_equity,
            portfolio=self.portfolio,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/engine/test_backtest.py -v`
Expected: PASS (4 passed)

If `test_kill_switch_flattens_and_halts` is flaky on the exact halt point, confirm the logic: a long 10k position built at 1.10 then marked at 0.50 loses 10_000*(1.10-0.50)=6_000 >> 1_000 limit, so `update_equity` halts and `_flatten` closes the position on that same tick. Do not weaken the test; fix the engine if needed.

- [ ] **Step 6: Run full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add src/fxtrader/engine/__init__.py src/fxtrader/engine/backtest.py tests/engine/test_backtest.py
git commit -m "feat: BacktestEngine (event loop, bars, risk/kill-switch, equity sampling)"
```

---

## Task 7: MaCrossover reference strategy

**Files:**
- Create: `src/fxtrader/strategy/examples/__init__.py`
- Create: `src/fxtrader/strategy/examples/ma_crossover.py`
- Create: `tests/strategy/test_ma_crossover.py`

A bar-based moving-average crossover: maintains fast/slow simple moving averages of bar closes; on a fast-above-slow cross goes long `size`, on fast-below-slow goes short `size` (flat→long→short flips). Single symbol.

- [ ] **Step 1: Write the failing test `tests/strategy/test_ma_crossover.py`**

```python
from datetime import datetime, timedelta, timezone

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Side
from fxtrader.strategy.examples.ma_crossover import MaCrossover


def _bar(close, minute):
    ts = datetime(2024, 1, 2, 9, minute, tzinfo=timezone.utc)
    return Bar("EURUSD", ts, close, close, close, close, 1.0)


def _bound_strategy(fast=2, slow=4, size=10_000):
    s = MaCrossover(symbol="EURUSD", fast=fast, slow=slow, size=size)
    orders = []
    s.bind(SimClock(), Portfolio(100_000), orders.append)
    return s, orders


def test_no_order_before_slow_window_filled():
    s, orders = _bound_strategy()
    for i, c in enumerate([1.10, 1.11, 1.12]):  # only 3 bars, slow=4
        s.on_bar(_bar(c, i))
    assert orders == []


def test_golden_cross_goes_long():
    s, orders = _bound_strategy(fast=2, slow=4, size=10_000)
    # rising series so fast MA crosses above slow MA
    for i, c in enumerate([1.10, 1.10, 1.10, 1.10, 1.20, 1.30]):
        s.on_bar(_bar(c, i))
    assert orders, "expected at least one order"
    first = orders[0]
    assert first.side is Side.BUY
    assert first.quantity == 10_000


def test_ignores_other_symbol_bars():
    s, orders = _bound_strategy()
    other = Bar("GBPUSD", datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc),
                1.27, 1.27, 1.27, 1.27, 1.0)
    for _ in range(10):
        s.on_bar(other)
    assert orders == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/strategy/test_ma_crossover.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.strategy.examples'`

- [ ] **Step 3: Create `src/fxtrader/strategy/examples/__init__.py`**

```python
"""Example/reference strategies."""
```

- [ ] **Step 4: Write `src/fxtrader/strategy/examples/ma_crossover.py`**

```python
"""Moving-average crossover reference strategy (bar-based, single symbol)."""

from __future__ import annotations

from collections import deque

from fxtrader.core.types import Bar, Side
from fxtrader.strategy.base import Strategy


class MaCrossover(Strategy):
    def __init__(self, symbol: str, fast: int, slow: int, size: float) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self.size = size
        self._closes: deque[float] = deque(maxlen=slow)
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol != self.symbol:
            return
        self._closes.append(bar.close)
        if len(self._closes) < self.slow:
            return

        closes = list(self._closes)
        fast = sum(closes[-self.fast:]) / self.fast
        slow = sum(closes) / self.slow

        if self._prev_fast is not None and self._prev_slow is not None:
            crossed_up = self._prev_fast <= self._prev_slow and fast > slow
            crossed_down = self._prev_fast >= self._prev_slow and fast < slow
            signal = 1 if crossed_up else (-1 if crossed_down else 0)
            if signal != 0:
                target = signal * self.size
                position = self.portfolio.position(self.symbol).quantity
                delta = target - position
                if delta != 0:
                    side = Side.BUY if delta > 0 else Side.SELL
                    self.order(self.symbol, side, abs(delta))

        self._prev_fast = fast
        self._prev_slow = slow
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/strategy/test_ma_crossover.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/fxtrader/strategy/examples/__init__.py src/fxtrader/strategy/examples/ma_crossover.py tests/strategy/test_ma_crossover.py
git commit -m "feat: MaCrossover reference strategy"
```

---

## Task 8: `fxtrader backtest` CLI + golden integration test

**Files:**
- Modify: `src/fxtrader/cli.py` (add `run_backtest` + `backtest` command)
- Create: `tests/test_cli_backtest.py`
- Create: `tests/integration/test_backtest_golden.py`

`run_backtest` wires a `HistoricalTickFeed` + `MaCrossover` + `SimulatedBroker` + `RiskManager` + `BacktestEngine`, computes metrics, and returns `(result, metrics)`. The command prints a metrics summary. A golden integration test drives the full stack on synthetic ticks with a hand-verifiable outcome.

- [ ] **Step 1: Write the failing test `tests/test_cli_backtest.py`**

```python
from datetime import datetime, timezone

from fxtrader.cli import run_backtest
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore


def _seed_store(tmp_path):
    store = TickStore(tmp_path)
    # 200 one-second ticks trending up, so a fast/slow MA cross fires on 1-min bars
    start = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    ticks = []
    from datetime import timedelta
    for i in range(200):
        p = 1.10 + i * 0.0005
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_backtest.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_backtest'`

- [ ] **Step 3: Modify `src/fxtrader/cli.py`**

Add to the imports at the top (merge with existing):

```python
from datetime import timedelta

from fxtrader.broker.simulated import SimulatedBroker
from fxtrader.core.portfolio import Portfolio
from fxtrader.data.feed import HistoricalTickFeed
from fxtrader.engine.backtest import BacktestEngine
from fxtrader.metrics.performance import compute_metrics
from fxtrader.risk.manager import RiskLimits, RiskManager
from fxtrader.strategy.examples.ma_crossover import MaCrossover
```

Add this helper (after `run_backfill`):

```python
def run_backtest(
    symbol: str,
    start: datetime,
    end: datetime,
    data_dir: str | Path,
    *,
    fast: int = 20,
    slow: int = 50,
    size: float = 10_000,
    bar_minutes: int = 1,
    starting_cash: float = 100_000.0,
    commission_rate: float = 0.00002,
    min_commission: float = 2.0,
    slippage: float = 0.0,
    daily_loss_limit: float = float("inf"),
    max_position: float = float("inf"),
):
    """Run a MaCrossover backtest over stored ticks. Returns (BacktestResult, metrics)."""
    store = TickStore(data_dir)
    feed = HistoricalTickFeed(store, [symbol], start, end)
    strategy = MaCrossover(symbol=symbol, fast=fast, slow=slow, size=size)
    broker = SimulatedBroker(
        commission_rate=commission_rate, min_commission=min_commission, slippage=slippage
    )
    portfolio = Portfolio(starting_cash)
    risk = RiskManager(RiskLimits(max_position=max_position, daily_loss_limit=daily_loss_limit))
    bar_interval = timedelta(minutes=bar_minutes)
    engine = BacktestEngine(
        feed=feed, strategy=strategy, broker=broker, portfolio=portfolio, risk=risk,
        bar_interval=bar_interval, equity_interval=bar_interval,
    )
    result = engine.run()
    periods_per_year = (365.25 * 24 * 3600) / bar_interval.total_seconds()
    metrics = compute_metrics(result.equity_curve, result.fills, periods_per_year=periods_per_year)
    return result, metrics
```

Add this command (after the `backfill` command):

```python
@app.command()
def backtest(
    symbol: str = typer.Option(..., help="Single symbol, e.g. EURUSD"),
    start: str = typer.Option(..., help="Inclusive start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Exclusive end date YYYY-MM-DD"),
    data_dir: str = typer.Option("data", help="Parquet store root"),
    fast: int = typer.Option(20, help="Fast MA window (bars)"),
    slow: int = typer.Option(50, help="Slow MA window (bars)"),
    size: float = typer.Option(10_000, help="Position size (base units)"),
    bar_minutes: int = typer.Option(1, help="Bar interval in minutes"),
) -> None:
    """Backtest the MA-crossover strategy over stored ticks."""
    _result, metrics = run_backtest(
        symbol=symbol, start=_parse_day(start), end=_parse_day(end),
        data_dir=data_dir, fast=fast, slow=slow, size=size, bar_minutes=bar_minutes,
    )
    typer.echo(f"symbol={symbol}  bars={bar_minutes}m  MA({fast}/{slow})")
    typer.echo(f"  trades        : {metrics['num_trades']}")
    typer.echo(f"  win rate      : {metrics['win_rate']:.1%}")
    typer.echo(f"  total return  : {metrics['total_return']:.2%}")
    typer.echo(f"  CAGR          : {metrics['cagr']:.2%}")
    typer.echo(f"  Sharpe        : {metrics['sharpe']:.2f}")
    typer.echo(f"  Sortino       : {metrics['sortino']:.2f}")
    typer.echo(f"  max drawdown  : {metrics['max_drawdown']:.2%}")
    typer.echo(f"  profit factor : {metrics['profit_factor']:.2f}")
    typer.echo(f"  final equity  : {metrics['final_equity']:,.2f}")
```

- [ ] **Step 4: Run the CLI test**

Run: `.venv/bin/pytest tests/test_cli_backtest.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Write the golden integration test `tests/integration/test_backtest_golden.py`**

```python
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
```

- [ ] **Step 6: Run the golden test + full suite + lint**

Run: `.venv/bin/pytest tests/integration/test_backtest_golden.py -v` → 1 passed
Run: `.venv/bin/pytest -q` → all pass
Run: `.venv/bin/ruff check src/fxtrader` → clean (fix any lint in new code)

- [ ] **Step 7: Verify the CLI end-to-end on REAL data (network-free; uses the downloaded store)**

Run: `.venv/bin/fxtrader backtest --symbol EURUSD --start 2024-01-02 --end 2024-02-01 --data-dir data --fast 20 --slow 50 --bar-minutes 5`
Expected: prints a metrics summary (trades, Sharpe, drawdown, final equity) over the real January EUR/USD data already in `data/`. Report the actual numbers.

- [ ] **Step 8: Commit**

```bash
git add src/fxtrader/cli.py tests/test_cli_backtest.py tests/integration/test_backtest_golden.py
git commit -m "feat: fxtrader backtest CLI + golden full-stack integration test"
```

---

## Self-review

**Spec coverage (P2 backtester portion):** §4.2 `Strategy`/`Broker`/`RiskManager`/`Clock`/`Portfolio` abstractions (Tasks 1,2,4; reuses existing Clock/Portfolio); §5.2 `SimulatedBroker` fills+slippage+commission + `BacktestEngine` (Tasks 3,6); §5.3 `Strategy` ABC + `MaCrossover` (Tasks 1,7); §5.4 `RiskManager` limits + daily-loss kill switch (Task 4, enforced in engine Task 6); §5.6 metrics (Task 5); §5.7 `fxtrader backtest` (Task 8). ✓ Deferred to P2b (noted): HTML/plot report + `results.json`, pydantic YAML config, limit/stop orders, walk-forward.

**Placeholder scan:** none — all code/tests are complete. The one cleanup note (remove unused `datetime` import in Task 1) is explicit.

**Type/signature consistency:** `Strategy.bind(clock, portfolio, submit)` / `order(symbol, side, quantity, *, order_type, limit_price, stop_price)`; `Broker.update_market(tick)` / `submit(order)` / `set_fill_callback(cb)` / `_emit_fill(fill)`; `SimulatedBroker(*, commission_rate, min_commission, slippage)`; `RiskLimits(max_position, max_gross_notional, daily_loss_limit, max_orders_per_minute)` + `RiskManager.check(order, portfolio, prices, clock)` / `update_equity(equity, clock)` / `.halted`; `compute_metrics(equity_curve, fills, *, periods_per_year)` + `trade_pnls(fills)`; `BacktestEngine(feed, strategy, broker, portfolio, risk, *, equity_interval, bar_interval)` + `.run() -> BacktestResult(equity_curve, fills, final_equity, portfolio)`; `MaCrossover(symbol, fast, slow, size)`; `run_backtest(symbol, start, end, data_dir, *, fast, slow, size, bar_minutes, ...)`. Used consistently across tasks.

**Scope:** Stops at a runnable, tested backtester with core metrics and a CLI summary. Reporting/config/limit orders/walk-forward are P2b.
