# Trend-Filtered Crossover + MAE-Driven Exits — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trend-filtered MA-crossover strategy with synthetic, MAE-informed stop/TP exits, plus the MAE/MFE diagnostic tooling and CLI selection to research it — without touching the engine or live path.

**Architecture:** New single-symbol bar-based `Strategy` (trend gate + flat-in-chop + auto-flatten) that delegates exits to a pluggable `ExitPolicy` (`NoExit`/`FixedPips`/`AtrMultiple`). Exits are market orders triggered at bar granularity inside `on_bar`. A new `metrics/excursions.py` reconstructs per-trade MAE/MFE from fills + bars (reusing the existing fill-pairing logic). A `strategy/registry.py` factory makes the CLI's new `--strategy`/exit options generic. Spec: `docs/superpowers/specs/2026-06-19-trend-filtered-crossover-design.md`.

**Tech Stack:** Python 3.12 (`.venv/bin/...`), polars, typer, pytest. No new dependencies.

## Global Constraints

- Python pinned to **3.12**; always invoke tooling as `.venv/bin/python` / `.venv/bin/pytest` / `.venv/bin/fxtrader` (shell `source` does not persist across tool calls).
- **No engine or live-path changes.** Do not edit `engine/backtest.py`, `engine/live.py`, `broker/*`, or `risk/*`. Exits are strategy-internal market orders; MAE tooling is post-processing.
- **DRY / YAGNI / TDD.** Test-first for every code unit. No trailing stops, no tick-level exit fills, no live wiring of the new strategy, no USDJPY, no feed-perf refactor.
- All work on branch `feat/trend-filtered-crossover` (already created; spec committed at `fc28288`).
- Full suite must stay green (baseline: **140 passed, 2 skipped**). Run `.venv/bin/pytest -q` before each commit.
- **Every commit message ends with this trailer** (one blank line before it):
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **Crossover seed-data gotcha (seen 3× already):** a cross fires only on an actual crossing, so test close series must dip-then-rise (or rise-then-dip) across *multiple* bars; a monotonic or single-bar series produces zero trades.
- Pip convention: `0.01` for `*JPY` pairs, else `0.0001`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/fxtrader/core/instruments.py` (Create) | `pip_size(symbol)` helper |
| `src/fxtrader/strategy/exits.py` (Create) | `ExitPolicy` protocol + `NoExit`, `FixedPips`, `AtrMultiple` |
| `src/fxtrader/strategy/examples/trend_filtered_crossover.py` (Create) | `TrendFilteredCrossover` strategy |
| `src/fxtrader/metrics/excursions.py` (Create) | `TradeRecord`, `trade_excursions`, `summarize_excursions` |
| `src/fxtrader/strategy/registry.py` (Create) | `make_strategy`, `make_exit_policy`, `STRATEGY_PARAMS` |
| `src/fxtrader/cli.py` (Modify) | strategy-generic `run_backtest`/`run_optimize`; `--strategy`+exit options on `backtest`/`optimize` |
| `research/diagnose_excursions.py` (Create) | glue script: trend run → MAE/MFE summary |

---

## Task 1: `pip_size` helper

**Files:**
- Create: `src/fxtrader/core/instruments.py`
- Test: `tests/core/test_instruments.py`

**Interfaces:**
- Produces: `pip_size(symbol: str) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_instruments.py
from fxtrader.core.instruments import pip_size


def test_pip_size_default_is_one_ten_thousandth():
    assert pip_size("EURUSD") == 0.0001
    assert pip_size("gbpusd") == 0.0001


def test_pip_size_jpy_pairs_are_one_hundredth():
    assert pip_size("USDJPY") == 0.01
    assert pip_size("eurjpy") == 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/core/test_instruments.py -v`
Expected: FAIL — `ModuleNotFoundError: fxtrader.core.instruments`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fxtrader/core/instruments.py
"""Instrument conventions. Pure, no I/O."""

from __future__ import annotations


def pip_size(symbol: str) -> float:
    """One pip in price terms: 0.01 for JPY-quoted pairs, else 0.0001."""
    return 0.01 if symbol.upper().endswith("JPY") else 0.0001
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/core/test_instruments.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/core/instruments.py tests/core/test_instruments.py
git commit  # feat(core): pip_size instrument helper  (+ Co-Authored-By trailer)
```

---

## Task 2: `ExitPolicy` protocol + `NoExit` + `FixedPips`

**Files:**
- Create: `src/fxtrader/strategy/exits.py`
- Test: `tests/strategy/test_exits.py`

**Interfaces:**
- Consumes: `Bar` from `fxtrader.core.types`.
- Produces:
  - `ExitPolicy` (Protocol): `update(self, bar: Bar) -> None`; `distances(self) -> tuple[float | None, float | None]` returning `(stop_dist, tp_dist)` in price terms.
  - `NoExit()` → `distances()` always `(None, None)`.
  - `FixedPips(stop_pips: float, tp_pips: float, pip_size: float)` → `distances()` = `(stop_pips * pip_size, tp_pips * pip_size)`; `0` pips → that leg is `None`. Raises `ValueError` on negative inputs.

- [ ] **Step 1: Write the failing test**

```python
# tests/strategy/test_exits.py
from datetime import datetime, timezone

import pytest

from fxtrader.core.types import Bar
from fxtrader.strategy.exits import FixedPips, NoExit


def _bar(o=1.0, h=1.0, l=1.0, c=1.0):
    return Bar("EURUSD", datetime(2025, 1, 1, tzinfo=timezone.utc), o, h, l, c, 1.0)


def test_no_exit_returns_none_none():
    p = NoExit()
    p.update(_bar())
    assert p.distances() == (None, None)


def test_fixed_pips_converts_pips_to_price_distance():
    p = FixedPips(stop_pips=10, tp_pips=20, pip_size=0.0001)
    assert p.distances() == (pytest.approx(0.0010), pytest.approx(0.0020))


def test_fixed_pips_zero_leg_is_none():
    p = FixedPips(stop_pips=10, tp_pips=0, pip_size=0.0001)
    assert p.distances() == (pytest.approx(0.0010), None)


def test_fixed_pips_rejects_negative():
    with pytest.raises(ValueError):
        FixedPips(stop_pips=-1, tp_pips=10, pip_size=0.0001)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/strategy/test_exits.py -v`
Expected: FAIL — `ModuleNotFoundError: fxtrader.strategy.exits`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fxtrader/strategy/exits.py
"""Pluggable exit policies. Pure logic; distances are in price terms."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from fxtrader.core.types import Bar


@runtime_checkable
class ExitPolicy(Protocol):
    def update(self, bar: Bar) -> None: ...
    def distances(self) -> tuple[float | None, float | None]: ...


class NoExit:
    """No stop/TP — the pure trend-gate strategy."""

    def update(self, bar: Bar) -> None:  # noqa: D401
        return None

    def distances(self) -> tuple[float | None, float | None]:
        return (None, None)


class FixedPips:
    """Constant stop/TP distances expressed in pips."""

    def __init__(self, stop_pips: float, tp_pips: float, pip_size: float) -> None:
        if stop_pips < 0 or tp_pips < 0 or pip_size <= 0:
            raise ValueError("pips must be >= 0 and pip_size > 0")
        self._stop = stop_pips * pip_size if stop_pips > 0 else None
        self._tp = tp_pips * pip_size if tp_pips > 0 else None

    def update(self, bar: Bar) -> None:
        return None

    def distances(self) -> tuple[float | None, float | None]:
        return (self._stop, self._tp)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/strategy/test_exits.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/exits.py tests/strategy/test_exits.py
git commit  # feat(strategy): ExitPolicy protocol + NoExit + FixedPips
```

---

## Task 3: `AtrMultiple` exit policy

**Files:**
- Modify: `src/fxtrader/strategy/exits.py`
- Test: `tests/strategy/test_exits.py`

**Interfaces:**
- Produces: `AtrMultiple(period: int, stop_mult: float, tp_mult: float)`. Maintains ATR over OHLC bars (`TR = max(high-low, |high-prev_close|, |low-prev_close|)`, simple mean of the last `period` true ranges). `distances()` = `(stop_mult*ATR, tp_mult*ATR)`, or `(None, None)` until `period` bars have been seen. Raises `ValueError` on non-positive `period` or negative mults.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/strategy/test_exits.py
from fxtrader.strategy.exits import AtrMultiple


def test_atr_multiple_none_until_warm():
    p = AtrMultiple(period=3, stop_mult=2.0, tp_mult=3.0)
    p.update(_bar(o=1.0, h=1.0010, l=0.9990, c=1.0))   # TR = 0.0020
    assert p.distances() == (None, None)               # not enough bars yet


def test_atr_multiple_scales_by_atr():
    p = AtrMultiple(period=2, stop_mult=2.0, tp_mult=3.0)
    # two bars each with true range 0.0020 (no gaps) -> ATR = 0.0020
    p.update(_bar(o=1.0, h=1.0010, l=0.9990, c=1.0))
    p.update(_bar(o=1.0, h=1.0010, l=0.9990, c=1.0))
    stop, tp = p.distances()
    assert stop == pytest.approx(0.0040)   # 2.0 * 0.0020
    assert tp == pytest.approx(0.0060)     # 3.0 * 0.0020


def test_atr_multiple_rejects_bad_params():
    with pytest.raises(ValueError):
        AtrMultiple(period=0, stop_mult=2.0, tp_mult=3.0)
    with pytest.raises(ValueError):
        AtrMultiple(period=3, stop_mult=-1.0, tp_mult=3.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/strategy/test_exits.py -v`
Expected: FAIL — `ImportError: cannot import name 'AtrMultiple'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/fxtrader/strategy/exits.py
from collections import deque


class AtrMultiple:
    """Stop/TP as multiples of ATR (volatility-scaled), frozen at entry by the caller."""

    def __init__(self, period: int, stop_mult: float, tp_mult: float) -> None:
        if period <= 0 or stop_mult < 0 or tp_mult < 0:
            raise ValueError("period must be > 0 and mults >= 0")
        self._period = period
        self._stop_mult = stop_mult
        self._tp_mult = tp_mult
        self._trs: deque[float] = deque(maxlen=period)
        self._prev_close: float | None = None

    def update(self, bar: Bar) -> None:
        if self._prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - self._prev_close),
                abs(bar.low - self._prev_close),
            )
        self._trs.append(tr)
        self._prev_close = bar.close

    def distances(self) -> tuple[float | None, float | None]:
        if len(self._trs) < self._period:
            return (None, None)
        atr = sum(self._trs) / len(self._trs)
        stop = self._stop_mult * atr if self._stop_mult > 0 else None
        tp = self._tp_mult * atr if self._tp_mult > 0 else None
        return (stop, tp)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/strategy/test_exits.py -v`
Expected: PASS (7 tests total in file)

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/exits.py tests/strategy/test_exits.py
git commit  # feat(strategy): ATR-scaled exit policy
```

---

## Task 4: `TrendFilteredCrossover` — signal + trend gate (NoExit)

**Files:**
- Create: `src/fxtrader/strategy/examples/trend_filtered_crossover.py`
- Test: `tests/strategy/test_trend_filtered_crossover.py`

**Interfaces:**
- Consumes: `Strategy` base (`bind`, `order`, `self.portfolio.position(sym).quantity`, `self.portfolio.position(sym).avg_price`), `Bar`, `Side`; `ExitPolicy`, `NoExit`.
- Produces: `TrendFilteredCrossover(symbol: str, fast: int, slow: int, trend: int, size: float, exit_policy: ExitPolicy | None = None)`. Bar-based. Raises `ValueError` unless `fast < slow < trend`.

**Test harness note:** drive the strategy directly. Bind it with a real `Portfolio` and an order sink that forwards fills into the portfolio so `position()` stays correct. Helper:

```python
# tests/strategy/test_trend_filtered_crossover.py
from datetime import datetime, timedelta, timezone

import pytest

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Fill, Order, Side
from fxtrader.strategy.examples.trend_filtered_crossover import TrendFilteredCrossover

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _drive(strategy, closes, *, highs=None, lows=None, symbol="EURUSD"):
    """Feed a close series as bars; fills execute immediately at bar.close into the portfolio."""
    pf = Portfolio(100_000.0)
    clock = SimClock()
    orders: list[Order] = []

    def sink(order: Order) -> None:
        orders.append(order)
        pf.apply_fill(Fill("x", order.symbol, order.side, order.quantity,
                           _drive.price, clock.now(), 0.0))

    strategy.bind(clock, pf, sink)
    strategy.on_start()
    for i, c in enumerate(closes):
        ts = T0 + timedelta(minutes=i)
        clock.set_time(ts)  # SimClock advance (verified API)
        hi = highs[i] if highs else c
        lo = lows[i] if lows else c
        _drive.price = c
        strategy.on_bar(Bar(symbol, ts, c, hi, lo, c, 1.0))
    return pf, orders
```

> Verified APIs: `SimClock.set_time(ts)` + `SimClock.now()`; `Portfolio.apply_fill(fill)` + `Portfolio.position(symbol)` (`.quantity`, `.avg_price`). This mirrors the engine's real fill path (`engine/backtest.py:68` → `portfolio.apply_fill`). The behavioral assertions below are what matter.

- [ ] **Step 1: Write the failing tests**

```python
def test_rejects_unordered_windows():
    with pytest.raises(ValueError):
        TrendFilteredCrossover("EURUSD", fast=5, slow=5, trend=20, size=10_000)
    with pytest.raises(ValueError):
        TrendFilteredCrossover("EURUSD", fast=2, slow=20, trend=10, size=10_000)


def test_no_trades_before_warmup():
    s = TrendFilteredCrossover("EURUSD", fast=2, slow=3, trend=5, size=10_000)
    _pf, orders = _drive(s, [1.0, 1.1, 1.2, 1.3])  # < trend bars
    assert orders == []


def test_bullish_cross_in_uptrend_goes_long():
    # rising series: fast crosses above slow while price is above the trend MA
    s = TrendFilteredCrossover("EURUSD", fast=2, slow=4, trend=6, size=10_000)
    closes = [1.00, 0.99, 0.98, 0.99, 1.01, 1.03, 1.05, 1.07]
    pf, orders = _drive(s, closes)
    assert pf.position("EURUSD").quantity > 0


def test_bullish_cross_below_trend_stays_flat():
    # cross up happens but price is still below the long trend MA -> no long
    s = TrendFilteredCrossover("EURUSD", fast=2, slow=4, trend=6, size=10_000)
    closes = [1.10, 1.08, 1.05, 1.02, 0.99, 0.97, 0.98, 0.99]
    pf, _orders = _drive(s, closes)
    assert pf.position("EURUSD").quantity == 0


def test_trend_flip_against_open_position_flattens():
    # establish a long in an uptrend, then drive price below trend MA with no new cross
    s = TrendFilteredCrossover("EURUSD", fast=2, slow=4, trend=6, size=10_000)
    up = [1.00, 0.99, 0.98, 0.99, 1.01, 1.03, 1.05, 1.07]
    down = [1.00, 0.93, 0.88, 0.84, 0.80, 0.77]
    pf, _orders = _drive(s, up + down)
    assert pf.position("EURUSD").quantity == 0


def test_ignores_other_symbols():
    s = TrendFilteredCrossover("EURUSD", fast=2, slow=4, trend=6, size=10_000)
    _pf, orders = _drive(s, [1, 0.99, 0.98, 0.99, 1.01, 1.03, 1.05, 1.07], symbol="GBPUSD")
    assert orders == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/strategy/test_trend_filtered_crossover.py -v`
Expected: FAIL — module/class not found.

- [ ] **Step 3: Write minimal implementation**

```python
# src/fxtrader/strategy/examples/trend_filtered_crossover.py
"""Trend-filtered MA crossover: trade crosses only with the long trend; flat otherwise."""

from __future__ import annotations

from collections import deque

from fxtrader.core.types import Bar, Side
from fxtrader.strategy.base import Strategy
from fxtrader.strategy.exits import ExitPolicy, NoExit


class TrendFilteredCrossover(Strategy):
    def __init__(
        self,
        symbol: str,
        fast: int,
        slow: int,
        trend: int,
        size: float,
        exit_policy: ExitPolicy | None = None,
    ) -> None:
        if not (fast < slow < trend):
            raise ValueError("windows must satisfy fast < slow < trend")
        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self.trend = trend
        self.size = size
        self.exit_policy: ExitPolicy = exit_policy or NoExit()
        self._closes: deque[float] = deque(maxlen=trend)
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None
        self._bias = 0          # -1 / 0 / +1 from the last cross
        self._pos_dir = 0       # sign of the position we currently intend to hold
        self._entry_price: float | None = None
        self._stop_dist: float | None = None
        self._tp_dist: float | None = None

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol != self.symbol:
            return
        self.exit_policy.update(bar)
        self._closes.append(bar.close)
        if len(self._closes) < self.trend:
            return

        closes = list(self._closes)
        fast = sum(closes[-self.fast:]) / self.fast
        slow = sum(closes[-self.slow:]) / self.slow
        trend_ma = sum(closes) / self.trend
        trend_dir = 1 if bar.close > trend_ma else (-1 if bar.close < trend_ma else 0)

        # (exit-policy stop/TP check is added in Task 5, before bias update)

        if self._prev_fast is not None and self._prev_slow is not None:
            if self._prev_fast <= self._prev_slow and fast > slow:
                self._bias = 1
            elif self._prev_fast >= self._prev_slow and fast < slow:
                self._bias = -1
        self._prev_fast = fast
        self._prev_slow = slow

        desired = self._bias * self.size if (self._bias != 0 and self._bias == trend_dir) else 0
        self._reconcile(bar, desired)

    def _reconcile(self, bar: Bar, desired: float) -> None:
        position = self.portfolio.position(self.symbol).quantity
        delta = desired - position
        if delta != 0:
            side = Side.BUY if delta > 0 else Side.SELL
            self.order(self.symbol, side, abs(delta))
        new_dir = 0 if desired == 0 else (1 if desired > 0 else -1)
        if new_dir == 0:
            self._entry_price = None
            self._stop_dist = self._tp_dist = None
        elif new_dir != self._pos_dir:
            self._entry_price = bar.close
            self._stop_dist, self._tp_dist = self.exit_policy.distances()
        self._pos_dir = new_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/strategy/test_trend_filtered_crossover.py -v`
Expected: PASS (6 tests). If a behavioral test fails, adjust the *seed series* (respect the dip-then-rise gotcha) — not the gate logic.

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/examples/trend_filtered_crossover.py tests/strategy/test_trend_filtered_crossover.py
git commit  # feat(strategy): trend-filtered crossover (signal + gate)
```

---

## Task 5: `TrendFilteredCrossover` — stop/TP exit handling

**Files:**
- Modify: `src/fxtrader/strategy/examples/trend_filtered_crossover.py`
- Test: `tests/strategy/test_trend_filtered_crossover.py`

**Interfaces:**
- Adds an exit check at the top of the post-warmup `on_bar` body: against the open position, using `_entry_price` + `_stop_dist`/`_tp_dist` captured at entry. Long: stop if `bar.low <= entry - stop_dist`, TP if `bar.high >= entry + tp_dist`. Short: mirrored. **Same-bar both-touched → stop wins.** On a hit: flatten and set `_bias = 0` (disarm until the next cross).

- [ ] **Step 1: Write the failing tests**

```python
from fxtrader.strategy.exits import FixedPips


def test_long_stop_hit_flattens_and_disarms():
    s = TrendFilteredCrossover(
        "EURUSD", fast=2, slow=4, trend=6, size=10_000,
        exit_policy=FixedPips(stop_pips=20, tp_pips=200, pip_size=0.0001),  # stop 0.0020
    )
    up = [1.00, 0.99, 0.98, 0.99, 1.01, 1.03, 1.05, 1.07]   # enter long near 1.07
    # next bar dips 0.0030 below entry via its low -> stop touched
    pf, _ = _drive(s, up + [1.066], highs=[*([None] * len(up)), 1.066],
                   lows=[*([None] * len(up)), 1.066 - 0.0])  # see note
    # simpler: craft an explicit low below entry-stop
    assert pf.position("EURUSD").quantity == 0


def test_same_bar_stop_and_tp_stop_wins():
    s = TrendFilteredCrossover(
        "EURUSD", fast=2, slow=4, trend=6, size=10_000,
        exit_policy=FixedPips(stop_pips=20, tp_pips=20, pip_size=0.0001),
    )
    up = [1.00, 0.99, 0.98, 0.99, 1.01, 1.03, 1.05, 1.07]
    # a bar whose range touches BOTH entry+0.0020 (tp) and entry-0.0020 (stop)
    wide = Bar  # placeholder; constructed inside _drive via highs/lows
    pf, _ = _drive(s, up + [1.07], highs=[*([None] * len(up)), 1.075],
                   lows=[*([None] * len(up)), 1.065])
    assert pf.position("EURUSD").quantity == 0
```

> When writing these, set `highs`/`lows` explicitly on the trigger bar so the touch is unambiguous: for a long entered at `entry`, a stop of `0.0020` is hit when `low <= entry - 0.0020`; a TP when `high >= entry + 0.0020`. Replace the sketched arrays above with concrete numbers once you print the actual entry price (`_drive` sets fill price = bar close). Keep the dip-then-rise gotcha in mind for the entry leg.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/strategy/test_trend_filtered_crossover.py -k "stop or tp" -v`
Expected: FAIL — position is still non-zero (no exit logic yet).

- [ ] **Step 3: Write minimal implementation**

Insert the exit check into `on_bar`, replacing the `# (exit-policy stop/TP check ...)` comment, immediately after `trend_dir` is computed and before the cross-detection block:

```python
        # --- exit check FIRST (against the open position), bar-granular ---
        position = self.portfolio.position(self.symbol).quantity
        if position != 0 and self._entry_price is not None:
            entry = self._entry_price
            hit = False
            if position > 0:  # long
                if self._stop_dist is not None and bar.low <= entry - self._stop_dist:
                    hit = True
                elif self._tp_dist is not None and bar.high >= entry + self._tp_dist:
                    hit = True
            else:  # short
                if self._stop_dist is not None and bar.high >= entry + self._stop_dist:
                    hit = True
                elif self._tp_dist is not None and bar.low <= entry - self._tp_dist:
                    hit = True
            if hit:
                side = Side.SELL if position > 0 else Side.BUY
                self.order(self.symbol, side, abs(position))
                self._bias = 0
                self._pos_dir = 0
                self._entry_price = None
                self._stop_dist = self._tp_dist = None
                return  # done with this bar
```

(The same-bar stop-first rule is encoded by checking the stop leg before the TP leg in each branch.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/strategy/test_trend_filtered_crossover.py -v`
Expected: PASS (all tests, including Task 4's).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/examples/trend_filtered_crossover.py tests/strategy/test_trend_filtered_crossover.py
git commit  # feat(strategy): synthetic stop/TP exits with stop-first same-bar rule
```

---

## Task 6: `trade_excursions` + `TradeRecord`

**Files:**
- Create: `src/fxtrader/metrics/excursions.py`
- Test: `tests/metrics/test_excursions.py`

**Interfaces:**
- Consumes: `Fill`, `Bar`, `Side` from `fxtrader.core.types`.
- Produces:
  - `TradeRecord` dataclass: `entry_ts: datetime`, `exit_ts: datetime`, `direction: int` (+1 long / −1 short), `pnl: float`, `mae_pips: float`, `mfe_pips: float`.
  - `trade_excursions(fills: list[Fill], bars: list[Bar], pip_size: float) -> list[TradeRecord]`. Pairs fills into round-trip trades using the same open/close logic as `metrics.performance.trade_pnls` (position opens when qty leaves 0, closes when it returns to 0). For each trade, scan bars with `entry_ts <= bar.ts <= exit_ts`: long → `mae = max(entry - bar.low, 0)`, `mfe = max(bar.high - entry, 0)`; short mirrored. Convert to pips by dividing by `pip_size`.

- [ ] **Step 1: Write the failing test**

```python
# tests/metrics/test_excursions.py
from datetime import datetime, timedelta, timezone

import pytest

from fxtrader.core.types import Bar, Fill, Side
from fxtrader.metrics.excursions import trade_excursions

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _fill(side, price, minute, qty=10_000):
    return Fill("x", "EURUSD", side, qty, price, T0 + timedelta(minutes=minute), 0.0)


def _bar(minute, hi, lo):
    ts = T0 + timedelta(minutes=minute)
    return Bar("EURUSD", ts, lo, hi, lo, lo, 1.0)


def test_long_trade_mae_mfe_in_pips():
    fills = [_fill(Side.BUY, 1.1000, 0), _fill(Side.SELL, 1.1010, 3)]
    bars = [_bar(0, 1.1000, 1.1000), _bar(1, 1.1005, 1.0990),  # adverse 10 pips, fav 5
            _bar(2, 1.1020, 1.1000), _bar(3, 1.1010, 1.1010)]  # fav 20 pips
    recs = trade_excursions(fills, bars, pip_size=0.0001)
    assert len(recs) == 1
    r = recs[0]
    assert r.direction == 1
    assert r.mae_pips == pytest.approx(10.0)   # 1.1000 - 1.0990
    assert r.mfe_pips == pytest.approx(20.0)   # 1.1020 - 1.1000
    assert r.pnl == pytest.approx((1.1010 - 1.1000) * 10_000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/metrics/test_excursions.py -v`
Expected: FAIL — `ModuleNotFoundError: fxtrader.metrics.excursions`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fxtrader/metrics/excursions.py
"""Per-trade Maximum Adverse / Favorable Excursion from fills + bars. Pure post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from fxtrader.core.types import Bar, Fill, Side


@dataclass(frozen=True, slots=True)
class TradeRecord:
    entry_ts: datetime
    exit_ts: datetime
    direction: int   # +1 long, -1 short
    pnl: float
    mae_pips: float
    mfe_pips: float


def _round_trips(fills: list[Fill]):
    """Yield (entry_ts, exit_ts, direction, entry_price) per closed round trip, single-symbol."""
    qty = 0.0
    entry_price = 0.0
    entry_ts: datetime | None = None
    direction = 0
    for f in fills:
        signed = f.quantity if f.side is Side.BUY else -f.quantity
        if qty == 0:
            qty = signed
            entry_price = f.price
            entry_ts = f.ts
            direction = 1 if signed > 0 else -1
            continue
        if (qty > 0) == (signed > 0):           # adding to position (rare here)
            entry_price = (entry_price * abs(qty) + f.price * abs(signed)) / abs(qty + signed)
            qty += signed
            continue
        # reducing / closing
        closing = min(abs(signed), abs(qty))
        yield (entry_ts, f.ts, direction, entry_price)
        qty += signed
        if qty != 0:                            # reversed straight through zero
            entry_price = f.price
            entry_ts = f.ts
            direction = 1 if qty > 0 else -1


def trade_excursions(fills: list[Fill], bars: list[Bar], pip_size: float) -> list[TradeRecord]:
    out: list[TradeRecord] = []
    for entry_ts, exit_ts, direction, entry_price in _round_trips(fills):
        mae = mfe = 0.0
        for b in bars:
            if b.ts < entry_ts or b.ts > exit_ts:
                continue
            if direction == 1:
                mae = max(mae, entry_price - b.low)
                mfe = max(mfe, b.high - entry_price)
            else:
                mae = max(mae, b.high - entry_price)
                mfe = max(mfe, entry_price - b.low)
        # pnl from entry to the closing bar's reference is approximate; use exit fill via bars' last close
        out.append(TradeRecord(
            entry_ts=entry_ts, exit_ts=exit_ts, direction=direction,
            pnl=_pnl(fills, entry_ts, exit_ts, direction, entry_price),
            mae_pips=mae / pip_size, mfe_pips=mfe / pip_size,
        ))
    return out


def _pnl(fills, entry_ts, exit_ts, direction, entry_price) -> float:
    exit_fill = next((f for f in fills if f.ts == exit_ts), None)
    if exit_fill is None:
        return 0.0
    return (exit_fill.price - entry_price) * exit_fill.quantity * direction
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/metrics/test_excursions.py -v`
Expected: PASS. If pairing differs from `trade_pnls`, cross-check against `src/fxtrader/metrics/performance.py:13` and align.

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/metrics/excursions.py tests/metrics/test_excursions.py
git commit  # feat(metrics): per-trade MAE/MFE excursions
```

---

## Task 7: `summarize_excursions`

**Files:**
- Modify: `src/fxtrader/metrics/excursions.py`
- Test: `tests/metrics/test_excursions.py`

**Interfaces:**
- Produces: `summarize_excursions(records: list[TradeRecord]) -> dict` with keys `n_trades`, `n_winners`, `n_losers`, and nested `winners`/`losers` dicts each holding `mae_p50/mae_p75/mae_p90` and `mfe_p50/mfe_p75/mfe_p90` (pips). A trade is a winner if `pnl > 0`. Empty input → zeroed summary.

- [ ] **Step 1: Write the failing test**

```python
from fxtrader.metrics.excursions import TradeRecord, summarize_excursions


def _rec(pnl, mae, mfe):
    return TradeRecord(T0, T0, 1, pnl, mae, mfe)


def test_summary_splits_winners_and_losers():
    recs = [_rec(5, 4, 30), _rec(8, 6, 40), _rec(-3, 25, 5), _rec(-9, 35, 3)]
    s = summarize_excursions(recs)
    assert s["n_trades"] == 4
    assert s["n_winners"] == 2 and s["n_losers"] == 2
    assert s["winners"]["mae_p50"] <= s["losers"]["mae_p50"]  # winners bleed less


def test_summary_empty_is_zeroed():
    s = summarize_excursions([])
    assert s["n_trades"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/metrics/test_excursions.py -k summary -v`
Expected: FAIL — `ImportError: cannot import name 'summarize_excursions'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/fxtrader/metrics/excursions.py
def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[idx]


def _leg(records: list[TradeRecord]) -> dict:
    mae = [r.mae_pips for r in records]
    mfe = [r.mfe_pips for r in records]
    return {
        "mae_p50": _pct(mae, 0.50), "mae_p75": _pct(mae, 0.75), "mae_p90": _pct(mae, 0.90),
        "mfe_p50": _pct(mfe, 0.50), "mfe_p75": _pct(mfe, 0.75), "mfe_p90": _pct(mfe, 0.90),
    }


def summarize_excursions(records: list[TradeRecord]) -> dict:
    winners = [r for r in records if r.pnl > 0]
    losers = [r for r in records if r.pnl <= 0]
    return {
        "n_trades": len(records),
        "n_winners": len(winners),
        "n_losers": len(losers),
        "winners": _leg(winners),
        "losers": _leg(losers),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/metrics/test_excursions.py -v`
Expected: PASS (all excursion tests).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/metrics/excursions.py tests/metrics/test_excursions.py
git commit  # feat(metrics): winner/loser MAE/MFE percentile summary
```

---

## Task 8: Strategy + exit registry

**Files:**
- Create: `src/fxtrader/strategy/registry.py`
- Test: `tests/strategy/test_registry.py`

**Interfaces:**
- Consumes: `MaCrossover`, `TrendFilteredCrossover`, `NoExit`/`FixedPips`/`AtrMultiple`, `pip_size`.
- Produces:
  - `STRATEGY_PARAMS: dict[str, tuple[str, ...]]` = `{"ma": ("fast","slow"), "trend": ("fast","slow","trend")}`.
  - `make_exit_policy(kind: str, *, params: dict, symbol: str) -> ExitPolicy`. `kind` ∈ `{"none","fixed","atr"}`. Unknown → `ValueError`.
  - `make_strategy(name: str, *, symbol: str, params: dict, exit_policy=None) -> Strategy`. `name` ∈ `{"ma","trend"}`. Unknown → `ValueError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/strategy/test_registry.py
import pytest

from fxtrader.strategy.examples.ma_crossover import MaCrossover
from fxtrader.strategy.examples.trend_filtered_crossover import TrendFilteredCrossover
from fxtrader.strategy.exits import AtrMultiple, FixedPips, NoExit
from fxtrader.strategy.registry import make_exit_policy, make_strategy


def test_make_strategy_ma():
    s = make_strategy("ma", symbol="EURUSD", params={"fast": 10, "slow": 30, "size": 10_000})
    assert isinstance(s, MaCrossover)


def test_make_strategy_trend_with_exit():
    s = make_strategy(
        "trend", symbol="EURUSD",
        params={"fast": 10, "slow": 30, "trend": 100, "size": 10_000},
        exit_policy=FixedPips(10, 20, 0.0001),
    )
    assert isinstance(s, TrendFilteredCrossover)


def test_make_strategy_unknown_raises():
    with pytest.raises(ValueError):
        make_strategy("nope", symbol="EURUSD", params={})


def test_make_exit_policy_variants():
    assert isinstance(make_exit_policy("none", params={}, symbol="EURUSD"), NoExit)
    assert isinstance(make_exit_policy(
        "fixed", params={"stop_pips": 10, "tp_pips": 20}, symbol="EURUSD"), FixedPips)
    assert isinstance(make_exit_policy(
        "atr", params={"atr_period": 14, "stop_atr": 2.0, "tp_atr": 3.0}, symbol="EURUSD"),
        AtrMultiple)
    with pytest.raises(ValueError):
        make_exit_policy("bad", params={}, symbol="EURUSD")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/strategy/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: fxtrader.strategy.registry`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fxtrader/strategy/registry.py
"""Factory mapping CLI strategy/exit names to objects. Keeps cli.py generic."""

from __future__ import annotations

from fxtrader.core.instruments import pip_size
from fxtrader.strategy.base import Strategy
from fxtrader.strategy.examples.ma_crossover import MaCrossover
from fxtrader.strategy.examples.trend_filtered_crossover import TrendFilteredCrossover
from fxtrader.strategy.exits import AtrMultiple, ExitPolicy, FixedPips, NoExit

STRATEGY_PARAMS: dict[str, tuple[str, ...]] = {
    "ma": ("fast", "slow"),
    "trend": ("fast", "slow", "trend"),
}


def make_exit_policy(kind: str, *, params: dict, symbol: str) -> ExitPolicy:
    if kind == "none":
        return NoExit()
    if kind == "fixed":
        return FixedPips(params["stop_pips"], params["tp_pips"], pip_size(symbol))
    if kind == "atr":
        return AtrMultiple(params["atr_period"], params["stop_atr"], params["tp_atr"])
    raise ValueError(f"unknown exit kind {kind!r}")


def make_strategy(name: str, *, symbol: str, params: dict, exit_policy: ExitPolicy | None = None) -> Strategy:
    if name == "ma":
        return MaCrossover(symbol=symbol, fast=params["fast"], slow=params["slow"], size=params["size"])
    if name == "trend":
        return TrendFilteredCrossover(
            symbol=symbol, fast=params["fast"], slow=params["slow"], trend=params["trend"],
            size=params["size"], exit_policy=exit_policy,
        )
    raise ValueError(f"unknown strategy {name!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/strategy/test_registry.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/registry.py tests/strategy/test_registry.py
git commit  # feat(strategy): strategy + exit registry
```

---

## Task 9: CLI — strategy-generic `run_backtest` + `backtest --strategy`

**Files:**
- Modify: `src/fxtrader/cli.py` (`run_backtest` ~L88-116; `backtest` command ~L241-289; imports ~L25)
- Test: `tests/cli/test_cli_trend.py` (new), reusing existing CLI test fixtures (read `tests/` for the small-data fixture the current `backtest` test uses).

**Interfaces:**
- `run_backtest(...)` gains keyword params (all defaulted to preserve MA behavior): `strategy_name: str = "ma"`, `trend: int = 0`, `exit_kind: str = "none"`, `stop_pips: float = 0.0`, `tp_pips: float = 0.0`, `atr_period: int = 0`, `stop_atr: float = 0.0`, `tp_atr: float = 0.0`. Builds the strategy via `make_strategy` + `make_exit_policy` instead of constructing `MaCrossover` directly.
- `backtest` command gains options: `--strategy`, `--trend`, `--exit`, `--stop-pips`, `--tp-pips`, `--atr-period`, `--stop-atr`, `--tp-atr`. For `--strategy trend`, route directly through `run_backtest` (not `BacktestConfig`) and print a strategy-aware summary. `--json`/`--html`/`--config` remain MA-only in v1 (documented YAGNI).

- [ ] **Step 1: Write the failing test**

```python
# tests/cli/test_cli_trend.py
from typer.testing import CliRunner

from fxtrader.cli import app

runner = CliRunner()
# Reuse the data fixture pattern from the existing backtest CLI test (read it first).
DATA_DIR = "tests/fixtures/ministore"   # <- replace with the real fixture path used by the repo


def test_backtest_trend_runs_end_to_end():
    result = runner.invoke(app, [
        "backtest", "--strategy", "trend", "--symbol", "EURUSD",
        "--start", "2025-01-01", "--end", "2025-01-08",
        "--fast", "5", "--slow", "20", "--trend", "50",
        "--exit", "fixed", "--stop-pips", "15", "--tp-pips", "30",
        "--bar-minutes", "15", "--data-dir", DATA_DIR,
    ])
    assert result.exit_code == 0, result.output
    assert "trend" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/cli/test_cli_trend.py -v`
Expected: FAIL — unknown option `--strategy` (exit code 2).

- [ ] **Step 3: Write minimal implementation**

In `run_backtest`, replace the `strategy = MaCrossover(...)` line (cli.py:102) and add params:

```python
    # new keyword params in the signature (after `size`, before `bar_minutes`), all defaulted:
    #   strategy_name: str = "ma", trend: int = 0, exit_kind: str = "none",
    #   stop_pips: float = 0.0, tp_pips: float = 0.0,
    #   atr_period: int = 0, stop_atr: float = 0.0, tp_atr: float = 0.0,
    from fxtrader.strategy.registry import make_exit_policy, make_strategy
    exit_policy = make_exit_policy(
        exit_kind,
        params={"stop_pips": stop_pips, "tp_pips": tp_pips,
                "atr_period": atr_period, "stop_atr": stop_atr, "tp_atr": tp_atr},
        symbol=symbol,
    )
    strategy = make_strategy(
        strategy_name, symbol=symbol,
        params={"fast": fast, "slow": slow, "trend": trend, "size": size},
        exit_policy=exit_policy,
    )
```

Add the options + trend branch to the `backtest` command (after the existing MA/config branch):

```python
    strategy: str = typer.Option("ma", help="ma | trend"),
    trend: int = typer.Option(200, help="Trend-filter MA window (trend strategy)"),
    exit_: str = typer.Option("none", "--exit", help="none | fixed | atr"),
    stop_pips: float = typer.Option(0.0, help="Stop in pips (exit=fixed)"),
    tp_pips: float = typer.Option(0.0, help="Take-profit in pips (exit=fixed)"),
    atr_period: int = typer.Option(14, help="ATR period (exit=atr)"),
    stop_atr: float = typer.Option(0.0, help="Stop = mult x ATR (exit=atr)"),
    tp_atr: float = typer.Option(0.0, help="TP = mult x ATR (exit=atr)"),
    # ... in the body, BEFORE the BacktestConfig path:
    if strategy == "trend":
        if not (symbol and start and end):
            raise typer.BadParameter("trend strategy needs --symbol/--start/--end")
        result, metrics = run_backtest(
            symbol=symbol, start=_parse_day(start), end=_parse_day(end), data_dir=data_dir,
            fast=fast, slow=slow, size=size, bar_minutes=bar_minutes,
            strategy_name="trend", trend=trend, exit_kind=exit_,
            stop_pips=stop_pips, tp_pips=tp_pips,
            atr_period=atr_period, stop_atr=stop_atr, tp_atr=tp_atr,
            daily_loss_limit=daily_loss_limit or float("inf"),
            max_position=max_position or float("inf"),
        )
        typer.echo(f"symbol={symbol}  bars={bar_minutes}m  trend MA({fast}/{slow}/{trend})  exit={exit_}")
        typer.echo(f"  trades        : {metrics['num_trades']}")
        typer.echo(f"  total return  : {metrics['total_return']:.2%}")
        typer.echo(f"  Sharpe        : {metrics['sharpe']:.2f}")
        typer.echo(f"  profit factor : {metrics['profit_factor']:.2f}")
        return
    if strategy not in ("ma", "trend"):
        raise typer.BadParameter("strategy must be ma | trend")
```

> Note: `run_backtest`'s `start`/`end` are `datetime` (the existing MA path passes parsed days via the config → `to_run_kwargs`). In the trend branch pass `_parse_day(...)`. Verify by reading the current `run_backtest` signature head (cli.py:84-98).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/cli/test_cli_trend.py -v && .venv/bin/pytest -q`
Expected: new test PASS; full suite still green (no regressions in the MA path).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/cli.py tests/cli/test_cli_trend.py
git commit  # feat(cli): --strategy trend backtest with exit options
```

---

## Task 10: CLI — `optimize --strategy` with exit grids

**Files:**
- Modify: `src/fxtrader/cli.py` (`_ma_run_fn` → generic; `run_optimize`; `optimize` command)
- Test: `tests/cli/test_cli_trend.py`

**Interfaces:**
- Generic `run_fn` builder `_strategy_run_fn(strategy_name, symbol, data_dir, bar_minutes, base_params, exit_kind)` returning `run_fn(params, start, end)` that merges `params` over `base_params`, validates windows (`fast < slow` and, for trend, `< trend`; return `None` if violated), and calls `run_backtest(..., strategy_name=..., exit_kind=..., **exit/signal kwargs)`.
- `run_optimize(...)` gains `strategy: str = "ma"`, `trend_values: list[int] | None = None`, `exit_kind: str = "none"`, and fixed exit params (`stop_pips`/`tp_pips`/`atr_period`/`stop_atr`/`tp_atr`). The param grid = the selected strategy's signal params present as lists (`STRATEGY_PARAMS`), built from `*_values`.
- `optimize` command gains `--strategy`, `--trend` (comma-list), `--exit`, and the fixed exit params. (Staged research: optimize signal with `--exit none`; later optimize exits by fixing signal and sweeping — for v1 the exit sweep is driven by re-running `optimize` per candidate or via the diagnose script; full exit-grid sweep is OPTIONAL and may be done by looping in `research/`.)

- [ ] **Step 1: Write the failing test**

```python
def test_optimize_trend_signal_runs():
    result = runner.invoke(app, [
        "optimize", "--strategy", "trend", "--symbol", "EURUSD",
        "--start", "2025-01-01", "--end", "2025-01-15",
        "--fast", "5,10", "--slow", "20,30", "--trend", "50,100",
        "--exit", "none", "--n-splits", "2", "--bar-minutes", "30",
        "--data-dir", DATA_DIR,
    ])
    assert result.exit_code == 0, result.output
    assert "mean OOS" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/cli/test_cli_trend.py -k optimize_trend -v`
Expected: FAIL — unknown option `--strategy`.

- [ ] **Step 3: Write minimal implementation**

```python
# replace _ma_run_fn with a generic builder
def _strategy_run_fn(strategy_name, symbol, data_dir, bar_minutes, base_params, exit_kind):
    def run_fn(params, start, end):
        p = {**base_params, **params}
        if p["fast"] >= p["slow"]:
            return None
        if strategy_name == "trend" and not (p["fast"] < p["slow"] < p["trend"]):
            return None
        _result, metrics = run_backtest(
            symbol=symbol, start=start, end=end, data_dir=data_dir,
            fast=p["fast"], slow=p["slow"], size=p.get("size", 10_000),
            bar_minutes=bar_minutes, strategy_name=strategy_name,
            trend=p.get("trend", 0), exit_kind=exit_kind,
            stop_pips=p.get("stop_pips", 0.0), tp_pips=p.get("tp_pips", 0.0),
            atr_period=p.get("atr_period", 14), stop_atr=p.get("stop_atr", 0.0),
            tp_atr=p.get("tp_atr", 0.0),
        )
        return metrics
    return run_fn
```

In `run_optimize`, build the grid from `STRATEGY_PARAMS[strategy]` (e.g. `{"fast": fast_values, "slow": slow_values}` for ma; add `"trend": trend_values` for trend), set `base_params={"size": size, "trend": trend_values[0] if ...}` only for the non-gridded fixed values, and call `walk_forward_optimize(run_fn, start, end, grid, ...)`. Add `--strategy/--trend/--exit` and exit params to the `optimize` command; default `--strategy ma` preserves current behavior.

> Keep the MA path byte-for-byte equivalent: when `strategy == "ma"` the grid is `{"fast":..., "slow":...}` and `exit_kind="none"`, so existing optimize tests must still pass unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/cli/test_cli_trend.py -v && .venv/bin/pytest -q`
Expected: new test PASS; full suite green (existing `optimize` tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/cli.py tests/cli/test_cli_trend.py
git commit  # feat(cli): --strategy trend optimize (signal grid)
```

---

## Task 11: Research execution — staged optimize + MAE diagnostic + findings

**Files:**
- Create: `research/diagnose_excursions.py` (glue, not unit-tested — mirrors existing `research/summarize.py` style)
- Create: `research/findings-2026-06-19.md`

This task runs the platform; it produces a report, not library code. Use coarse bars and the established IS/OOS split. **Compute discipline:** serial runs only (the feed OOM caveat), bars 15–30m, EURUSD primary.

- [ ] **Step 1: Stage-1 signal walk-forward (in-sample, no exits)**

```bash
.venv/bin/fxtrader optimize --strategy trend --symbol EURUSD \
  --start 2023-06-01 --end 2025-06-01 \
  --fast 10,20 --slow 50,100 --trend 200 --exit none \
  --n-splits 3 --bar-minutes 30 --objective sharpe \
  --json research/json/trend_EURUSD_signal_30m.json
```
Record the robust signal params (modal `best_params` across folds).

- [ ] **Step 2: MAE/MFE diagnostic on the chosen signal**

Write `research/diagnose_excursions.py`:

```python
"""Run the trend strategy (no exits) on in-sample EURUSD and print the MAE/MFE summary."""
from datetime import datetime, timezone

from fxtrader.cli import run_backtest
from fxtrader.data.feed import HistoricalTickFeed, Resampler
from fxtrader.data.store import TickStore
from fxtrader.metrics.excursions import summarize_excursions, trade_excursions
from fxtrader.core.instruments import pip_size

SYMBOL, BAR_MIN = "EURUSD", 30
START = datetime(2023, 6, 1, tzinfo=timezone.utc)
END = datetime(2025, 6, 1, tzinfo=timezone.utc)
FAST, SLOW, TREND = 20, 100, 200   # <- from Step 1

result, _metrics = run_backtest(
    symbol=SYMBOL, start=START, end=END, data_dir="data",
    fast=FAST, slow=SLOW, size=10_000, bar_minutes=BAR_MIN,
    strategy_name="trend", trend=TREND, exit_kind="none",
)

# rebuild the same bars the engine saw, for excursion measurement
from datetime import timedelta
res = Resampler(timedelta(minutes=BAR_MIN))
bars = []
for tick in HistoricalTickFeed(TickStore("data"), [SYMBOL], START, END):
    bar = res.update(tick)
    if bar:
        bars.append(bar)
last = res.flush()
if last:
    bars.append(last)

recs = trade_excursions(result.fills, bars, pip_size(SYMBOL))
import json
print(json.dumps(summarize_excursions(recs), indent=2))
```

Run: `.venv/bin/python research/diagnose_excursions.py | tee research/json/trend_EURUSD_mae.json`
From the winner/loser percentiles, derive a small candidate set: e.g. fixed `stop_pips ≈ winners.mae_p90`, `tp_pips ≈ winners.mfe_p50..p75`; ATR mults `stop_atr ∈ {1.5, 2.0}`, `tp_atr ∈ {2.0, 3.0}`.

- [ ] **Step 3: Stage-3 exit candidates — backtest each on in-sample, then validate finalists on OOS**

For each candidate (fixed + ATR), run an in-sample `backtest --strategy trend` with the fixed signal; keep the best 1–2 by Sharpe/PF; then run those **once** on the frozen OOS window:

```bash
.venv/bin/fxtrader backtest --strategy trend --symbol EURUSD \
  --start 2025-06-01 --end 2026-05-01 \
  --fast 20 --slow 100 --trend 200 \
  --exit fixed --stop-pips <S> --tp-pips <T> --bar-minutes 30 --data-dir data
```

- [ ] **Step 4: Write `research/findings-2026-06-19.md`**

Include: signal params chosen, the MAE/MFE summary table (winner vs loser percentiles), exit candidates tested, the in-sample vs frozen-OOS results table, and an honest verdict (including "no edge survived costs" if that is the outcome) + recommended next step (mean-reversion strategy B if trend has no edge).

- [ ] **Step 5: Commit**

```bash
git add research/diagnose_excursions.py research/findings-2026-06-19.md research/json/
git commit  # research: trend-filtered crossover MAE study + findings
```

---

## Self-Review

**Spec coverage:**
- §3 strategy logic → Tasks 4–5. §4 ExitPolicy (NoExit/FixedPips/AtrMultiple) → Tasks 2–3. §5 MAE/MFE tooling → Tasks 6–7. §6 registry + CLI → Tasks 8–10. `pip_size` helper → Task 1. §7 staged research process → Task 11. §11 DoD (suite green, end-to-end `--strategy trend`, findings report) → Tasks 9–11. All spec sections map to a task.
- **Coverage note:** the spec mentioned `--exit atr` end-to-end via `optimize`; v1 sweeps ATR/exit candidates via repeated `backtest` runs in Task 11 rather than a single joint optimize grid — a deliberate scope reduction stated in Task 10 and consistent with the spec's "staged optimization" (§7).

**Placeholder scan:** Tasks 5 and 9 contain sketch arrays / fixture paths the engineer must fill with concrete numbers (entry price, real fixture dir) — these are flagged inline with explicit instructions on how to compute them, not silent TBDs. No "add error handling"-style placeholders remain.

**Type consistency:** `make_strategy`/`make_exit_policy` signatures match between Task 8 (definition) and Tasks 9–10 (use). `run_backtest` new kwargs (`strategy_name`, `trend`, `exit_kind`, `stop_pips`, `tp_pips`, `atr_period`, `stop_atr`, `tp_atr`) are identical in Tasks 9 and 10. `TradeRecord` fields match between Tasks 6 and 7. `ExitPolicy.distances()` return type `(float|None, float|None)` consistent across Tasks 2, 3, 4, 5.

**Known executor caveats (flagged, not placeholders):** the Task 4 harness APIs (`SimClock.set_time`, `Portfolio.apply_fill`) are verified against the source. The remaining open item is the CLI fixture path in Tasks 9–10 (`DATA_DIR`): set it to the repo's existing small-data test fixture by reading the current `backtest` CLI test first.
