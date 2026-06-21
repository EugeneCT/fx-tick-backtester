# Mean-Reversion Fader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a z-score mean-reversion fader with a Kaufman Efficiency-Ratio ranging gate and the existing stop/TP exit overlay, wired into the `--strategy` CLI — without touching the engine or live path.

**Architecture:** New single-symbol bar-based `Strategy` that fades stretched moves (`z = (close − SMA)/std`) back to the mean, only when ranging (`efficiency_ratio < er_threshold`), exiting on reversion (`|z| ≤ exit_z`) or the `ExitPolicy` stop/TP (risk-first). Two pure indicators (`zscore`, `efficiency_ratio`) live in a new `strategy/indicators.py`. The registry and CLI accrete a `meanrev` case alongside `ma`/`trend`. Spec: `docs/superpowers/specs/2026-06-21-mean-reversion-fader-design.md`.

**Tech Stack:** Python 3.12 (`.venv/bin/...`), polars, typer, pytest. No new dependencies.

## Global Constraints

- Python pinned to **3.12**; invoke tooling as `.venv/bin/python` / `.venv/bin/pytest` / `.venv/bin/fxtrader` (shell `source` does not persist across tool calls).
- **No engine or live-path changes.** Do not edit `engine/*`, `broker/*`, `risk/*`, or `build_live_engine`. Exits are strategy-internal market orders.
- **DRY / YAGNI / TDD.** Test-first for every code unit. No separate ER window (the z `lookback` is the ER window), no max-hold exit, no trailing stops, no live wiring, no USDJPY, no generalizing the CLI param plumbing (accrete `meanrev` flags like `--trend` was added).
- **Reuse, don't reinvent:** `ExitPolicy` (`strategy/exits.py`), `pip_size`, the `_entry_price`/`_stop_dist`/`_tp_dist`/`_pos_dir`/`_last_exit_reason` exit-overlay pattern and stop-first same-bar rule (mirror `strategy/examples/trend_filtered_crossover.py`).
- **MA/trend paths must stay byte-identical.** The existing `optimize` MA grid `{"fast","slow"}` and trend grid `{"fast","slow","trend"}` and all their tests must be unchanged. Run the FULL suite before each commit (baseline: **171 passed, 2 skipped**).
- All work on branch `feat/mean-reversion-fader` (already created; spec committed at `035ba2d`). Do NOT create other branches.
- **Every commit message ends with this trailer** (one blank line before it):
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Pip convention: `0.01` for `*JPY`, else `0.0001` (handled by `pip_size`; the fader validates on EURUSD/GBPUSD).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/fxtrader/strategy/indicators.py` (Create) | pure `zscore` + `efficiency_ratio` |
| `src/fxtrader/strategy/examples/mean_reversion_fader.py` (Create) | `MeanReversionFader` strategy |
| `src/fxtrader/strategy/registry.py` (Modify) | `meanrev` in `STRATEGY_PARAMS` + `make_strategy` |
| `src/fxtrader/cli.py` (Modify) | `run_backtest` meanrev kwargs; `backtest`/`optimize` meanrev options; `_strategy_run_fn` + grid; `_float_list` |
| `tests/strategy/test_indicators.py` (Create) | indicator unit tests |
| `tests/strategy/test_mean_reversion_fader.py` (Create) | strategy unit tests |
| `tests/strategy/test_registry.py` (Modify) | meanrev factory case |
| `tests/test_cli_meanrev.py` (Create) | backtest + optimize end-to-end |

The staged research RUN (signal WFO → MAE → exit WFO → OOS on EURUSD+GBPUSD → `research/findings-<date>.md`) is a **separate background job**, not part of this plan.

---

## Task 1: Indicators — `zscore` + `efficiency_ratio`

**Files:**
- Create: `src/fxtrader/strategy/indicators.py`
- Test: `tests/strategy/test_indicators.py`

**Interfaces:**
- Produces: `zscore(closes: Sequence[float]) -> float` = `(closes[-1] − mean) / pstdev`, returns `0.0` when population std is 0. `efficiency_ratio(closes: Sequence[float]) -> float` = `abs(closes[-1] − closes[0]) / sum(abs(c_i − c_{i-1}))`, returns `0.0` when the denominator is 0; result in `[0, 1]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/strategy/test_indicators.py
import pytest

from fxtrader.strategy.indicators import efficiency_ratio, zscore


def test_zscore_known_series():
    # mean=3, pstdev=sqrt(2)~1.41421, last=5 -> z = 2/1.41421
    assert zscore([1, 2, 3, 4, 5]) == pytest.approx(2 / (2 ** 0.5))


def test_zscore_flat_window_is_zero():
    assert zscore([2.0, 2.0, 2.0]) == 0.0


def test_efficiency_ratio_monotonic_is_one():
    assert efficiency_ratio([1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_efficiency_ratio_oscillating_is_zero():
    # net move 0 over the window, lots of gross movement
    assert efficiency_ratio([1, 2, 1, 2, 1]) == pytest.approx(0.0)


def test_efficiency_ratio_flat_is_zero():
    assert efficiency_ratio([1.0, 1.0, 1.0]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/strategy/test_indicators.py -v`
Expected: FAIL — `ModuleNotFoundError: fxtrader.strategy.indicators`

- [ ] **Step 3: Write minimal implementation**

```python
# src/fxtrader/strategy/indicators.py
"""Pure technical indicators used by strategies. No I/O, no state."""

from __future__ import annotations

import statistics
from collections.abc import Sequence


def zscore(closes: Sequence[float]) -> float:
    """(last − mean) / population stdev over the window; 0.0 when the window is flat."""
    std = statistics.pstdev(closes)
    if std == 0:
        return 0.0
    return (closes[-1] - statistics.fmean(closes)) / std


def efficiency_ratio(closes: Sequence[float]) -> float:
    """Kaufman Efficiency Ratio: |net move| / sum(|bar-to-bar moves|), in [0, 1]; 0.0 if flat."""
    gross = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
    if gross == 0:
        return 0.0
    return abs(closes[-1] - closes[0]) / gross
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/strategy/test_indicators.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/indicators.py tests/strategy/test_indicators.py
git commit  # feat(strategy): zscore + efficiency_ratio indicators   (+ trailer)
```

---

## Task 2: `MeanReversionFader` — entry + regime gate + signal exit (NoExit)

**Files:**
- Create: `src/fxtrader/strategy/examples/mean_reversion_fader.py`
- Test: `tests/strategy/test_mean_reversion_fader.py`

**Interfaces:**
- Consumes: `Strategy` base, `Bar`/`Side`, `zscore`/`efficiency_ratio`, `ExitPolicy`/`NoExit`.
- Produces: `MeanReversionFader(symbol, lookback, entry_z, exit_z, er_threshold, size, exit_policy=None)`. Raises `ValueError` unless `lookback ≥ 2`, `0 ≤ exit_z < entry_z`, `0 < er_threshold ≤ 1`. Captures `_entry_price`/`_stop_dist`/`_tp_dist` at entry and exposes `_last_exit_reason` (Task 3 uses these for stop/TP).

**Test harness** (mirrors the trend strategy's `_drive`; include it in the new test file):

```python
# tests/strategy/test_mean_reversion_fader.py
from datetime import datetime, timedelta, timezone

import pytest

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Fill, Order, Side
from fxtrader.strategy.exits import FixedPips
from fxtrader.strategy.examples.mean_reversion_fader import MeanReversionFader

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _drive(strategy, closes, *, highs=None, lows=None, symbol="EURUSD"):
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
        clock.set_time(ts)
        hi = highs[i] if highs else c
        lo = lows[i] if lows else c
        _drive.price = c
        strategy.on_bar(Bar(symbol, ts, c, hi, lo, c, 1.0))
    return pf, orders
```

- [ ] **Step 1: Write the failing tests**

```python
def test_rejects_bad_params():
    with pytest.raises(ValueError):
        MeanReversionFader("EURUSD", lookback=1, entry_z=2, exit_z=0.5, er_threshold=0.3, size=10_000)
    with pytest.raises(ValueError):
        MeanReversionFader("EURUSD", lookback=5, entry_z=2, exit_z=2, er_threshold=0.3, size=10_000)
    with pytest.raises(ValueError):
        MeanReversionFader("EURUSD", lookback=5, entry_z=2, exit_z=0.5, er_threshold=0, size=10_000)


def test_no_trades_before_warmup():
    s = MeanReversionFader("EURUSD", lookback=5, entry_z=1.0, exit_z=0.3, er_threshold=0.5, size=10_000)
    _pf, orders = _drive(s, [1.0, 1.0, 1.0, 1.0])  # < lookback+1 bars
    assert orders == []


def test_fades_short_when_stretched_high_and_ranging():
    # an oscillating (ranging -> low ER) series that ends with a high spike -> z high
    s = MeanReversionFader("EURUSD", lookback=5, entry_z=1.0, exit_z=0.3, er_threshold=0.6, size=10_000)
    closes = [1.0000, 1.0010, 1.0000, 1.0010, 1.0000, 1.0010, 1.0040]
    pf, _ = _drive(s, closes)
    assert pf.position("EURUSD").quantity < 0   # short the stretched-high price


def test_no_entry_when_stretched_but_trending():
    # a monotonic ramp: ER ~ 1 (trending) even though z is large -> regime gate blocks entry
    s = MeanReversionFader("EURUSD", lookback=5, entry_z=1.0, exit_z=0.3, er_threshold=0.5, size=10_000)
    closes = [1.0000, 1.0010, 1.0020, 1.0030, 1.0040, 1.0050, 1.0060]
    pf, _ = _drive(s, closes)
    assert pf.position("EURUSD").quantity == 0


def test_signal_exit_on_reversion_to_mean():
    s = MeanReversionFader("EURUSD", lookback=5, entry_z=1.0, exit_z=0.3, er_threshold=0.9, size=10_000)
    # ranging, spike high (enter short), then revert toward the mean (|z| <= exit_z) -> flat
    closes = [1.0000, 1.0010, 1.0000, 1.0010, 1.0000, 1.0010, 1.0040, 1.0008]
    pf, _ = _drive(s, closes)
    assert pf.position("EURUSD").quantity == 0
    assert s._last_exit_reason == "signal"


def test_ignores_other_symbols():
    s = MeanReversionFader("EURUSD", lookback=5, entry_z=1.0, exit_z=0.3, er_threshold=0.9, size=10_000)
    _pf, orders = _drive(s, [1.0, 1.001, 1.0, 1.001, 1.0, 1.001, 1.004], symbol="GBPUSD")
    assert orders == []
```

> When a behavioral test doesn't fire as intended, adjust the SEED SERIES (z and ER are coupled — an oscillating series gives low ER + a spike gives high z; a ramp gives high ER), not the strategy logic. Print `zscore(window)` / `efficiency_ratio(closes)` for the trigger bar to pick numbers. Do NOT weaken assertions.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/strategy/test_mean_reversion_fader.py -v`
Expected: FAIL — module/class not found.

- [ ] **Step 3: Write minimal implementation**

```python
# src/fxtrader/strategy/examples/mean_reversion_fader.py
"""Z-score mean-reversion fader, gated to ranging regimes by Kaufman Efficiency Ratio."""

from __future__ import annotations

import statistics
from collections import deque

from fxtrader.core.types import Bar, Side
from fxtrader.strategy.base import Strategy
from fxtrader.strategy.exits import ExitPolicy, NoExit
from fxtrader.strategy.indicators import efficiency_ratio, zscore


class MeanReversionFader(Strategy):
    def __init__(
        self,
        symbol: str,
        lookback: int,
        entry_z: float,
        exit_z: float,
        er_threshold: float,
        size: float,
        exit_policy: ExitPolicy | None = None,
    ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if not (0 <= exit_z < entry_z):
            raise ValueError("require 0 <= exit_z < entry_z")
        if not (0 < er_threshold <= 1):
            raise ValueError("er_threshold must be in (0, 1]")
        self.symbol = symbol
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.er_threshold = er_threshold
        self.size = size
        self.exit_policy: ExitPolicy = exit_policy or NoExit()
        self._closes: deque[float] = deque(maxlen=lookback + 1)
        self._pos_dir = 0
        self._entry_price: float | None = None
        self._stop_dist: float | None = None
        self._tp_dist: float | None = None
        self._last_exit_reason: str | None = None

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol != self.symbol:
            return
        self.exit_policy.update(bar)
        self._closes.append(bar.close)
        if len(self._closes) < self.lookback + 1:
            return
        closes = list(self._closes)
        window = closes[-self.lookback:]
        if statistics.pstdev(window) == 0.0:
            return
        z = zscore(window)
        er = efficiency_ratio(closes)

        position = self.portfolio.position(self.symbol).quantity

        # (Task 3 inserts the stop/TP exit check HERE, before the signal exit.)

        # signal exit: reverted to the mean
        if position != 0 and abs(z) <= self.exit_z:
            self._flatten(position, "signal")
            return

        # entry: only when flat AND ranging
        desired = 0.0
        if position == 0 and er < self.er_threshold:
            if z > self.entry_z:
                desired = -self.size
            elif z < -self.entry_z:
                desired = self.size
        self._reconcile(bar, desired)

    def _flatten(self, position: float, reason: str) -> None:
        side = Side.SELL if position > 0 else Side.BUY
        self.order(self.symbol, side, abs(position))
        self._last_exit_reason = reason
        self._pos_dir = 0
        self._entry_price = None
        self._stop_dist = self._tp_dist = None

    def _reconcile(self, bar: Bar, desired: float) -> None:
        position = self.portfolio.position(self.symbol).quantity
        delta = desired - position
        if delta != 0:
            side = Side.BUY if delta > 0 else Side.SELL
            self.order(self.symbol, side, abs(delta))
        new_dir = 0 if desired == 0 else (1 if desired > 0 else -1)
        if new_dir != 0 and new_dir != self._pos_dir:
            self._entry_price = bar.close
            self._stop_dist, self._tp_dist = self.exit_policy.distances()
        self._pos_dir = new_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/strategy/test_mean_reversion_fader.py -v`
Expected: PASS (6 tests). Tune seed series (not logic) if a behavioral case doesn't fire.

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/examples/mean_reversion_fader.py tests/strategy/test_mean_reversion_fader.py
git commit  # feat(strategy): mean-reversion fader (entry + ER gate + signal exit)
```

---

## Task 3: Fader stop/TP exit overlay

**Files:**
- Modify: `src/fxtrader/strategy/examples/mean_reversion_fader.py`
- Test: `tests/strategy/test_mean_reversion_fader.py`

**Interfaces:**
- Inserts the stop/TP check at the marked spot in `on_bar`, **before** the signal-exit. Long: stop if `bar.low ≤ entry − stop_dist`, TP if `bar.high ≥ entry + tp_dist`; short mirrored; **same-bar both touched → stop wins** (stop leg checked before TP). On a hit: `_flatten(position, "stop"|"tp")` and `return`. `None` legs don't trigger.

- [ ] **Step 1: Write the failing test**

```python
def test_stop_hit_flattens_and_disarms():
    s = MeanReversionFader(
        "EURUSD", lookback=5, entry_z=1.0, exit_z=0.3, er_threshold=0.9, size=10_000,
        exit_policy=FixedPips(stop_pips=20, tp_pips=500, pip_size=0.0001),  # stop 0.0020
    )
    # ranging then spike high -> enter SHORT near 1.0040; then a bar whose HIGH runs to the stop
    closes = [1.0000, 1.0010, 1.0000, 1.0010, 1.0000, 1.0010, 1.0040, 1.0045]
    highs = [None] * 7 + [1.0065]      # short stop = entry(1.0040) + 0.0020 = 1.0060 -> high 1.0065 hits
    lows = [None] * 7 + [1.0045]
    pf, _ = _drive(s, closes, highs=highs, lows=lows)
    assert pf.position("EURUSD").quantity == 0
    assert s._last_exit_reason == "stop"
```

> Derive the real entry price first by printing it (the fader enters at the spike bar's close). Set the trigger bar's `high`/`low` so it unambiguously breaches the short stop (`entry + stop_dist`) and not the TP. Keep assertions strict.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/strategy/test_mean_reversion_fader.py -k stop_hit -v`
Expected: FAIL — position still open (no stop logic yet).

- [ ] **Step 3: Write minimal implementation**

Replace the `# (Task 3 inserts the stop/TP exit check HERE ...)` comment with:

```python
        # stop/TP overlay (risk-first), bar-granular, stop-first same-bar rule
        if position != 0 and self._entry_price is not None:
            entry = self._entry_price
            reason = None
            if position > 0:  # long
                if self._stop_dist is not None and bar.low <= entry - self._stop_dist:
                    reason = "stop"
                elif self._tp_dist is not None and bar.high >= entry + self._tp_dist:
                    reason = "tp"
            else:  # short
                if self._stop_dist is not None and bar.high >= entry + self._stop_dist:
                    reason = "stop"
                elif self._tp_dist is not None and bar.low <= entry - self._tp_dist:
                    reason = "tp"
            if reason:
                self._flatten(position, reason)
                return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/strategy/test_mean_reversion_fader.py -v`
Expected: PASS (all fader tests).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/examples/mean_reversion_fader.py tests/strategy/test_mean_reversion_fader.py
git commit  # feat(strategy): fader stop/TP exit overlay (risk-first)
```

---

## Task 4: Registry — `meanrev` factory case

**Files:**
- Modify: `src/fxtrader/strategy/registry.py`
- Test: `tests/strategy/test_registry.py`

**Interfaces:**
- `STRATEGY_PARAMS["meanrev"] = ("lookback", "entry_z", "exit_z", "er_threshold")`.
- `make_strategy("meanrev", symbol=..., params={"lookback","entry_z","exit_z","er_threshold","size"}, exit_policy=...) -> MeanReversionFader`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/strategy/test_registry.py
from fxtrader.strategy.examples.mean_reversion_fader import MeanReversionFader
from fxtrader.strategy.exits import FixedPips
from fxtrader.strategy.registry import STRATEGY_PARAMS, make_strategy


def test_make_strategy_meanrev():
    s = make_strategy(
        "meanrev", symbol="EURUSD",
        params={"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "er_threshold": 0.3, "size": 10_000},
        exit_policy=FixedPips(10, 20, 0.0001),
    )
    assert isinstance(s, MeanReversionFader)


def test_strategy_params_has_meanrev():
    assert STRATEGY_PARAMS["meanrev"] == ("lookback", "entry_z", "exit_z", "er_threshold")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/strategy/test_registry.py -k meanrev -v`
Expected: FAIL — `KeyError: 'meanrev'` / `ValueError: unknown strategy 'meanrev'`.

- [ ] **Step 3: Write minimal implementation**

In `src/fxtrader/strategy/registry.py`: add the import, the `STRATEGY_PARAMS` entry, and the `make_strategy` branch.

```python
from fxtrader.strategy.examples.mean_reversion_fader import MeanReversionFader
```
```python
STRATEGY_PARAMS: dict[str, tuple[str, ...]] = {
    "ma": ("fast", "slow"),
    "trend": ("fast", "slow", "trend"),
    "meanrev": ("lookback", "entry_z", "exit_z", "er_threshold"),
}
```
```python
    # inside make_strategy, before the final `raise ValueError`:
    if name == "meanrev":
        return MeanReversionFader(
            symbol=symbol, lookback=params["lookback"], entry_z=params["entry_z"],
            exit_z=params["exit_z"], er_threshold=params["er_threshold"],
            size=params["size"], exit_policy=exit_policy,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/strategy/test_registry.py -v`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/registry.py tests/strategy/test_registry.py
git commit  # feat(strategy): register meanrev in the strategy factory
```

---

## Task 5: CLI — `run_backtest` meanrev kwargs + `backtest --strategy meanrev`

**Files:**
- Modify: `src/fxtrader/cli.py` (`run_backtest` signature + its `make_strategy` params dict; `backtest` command options + a `meanrev` branch)
- Test: `tests/test_cli_meanrev.py` (new)

**Interfaces:**
- `run_backtest(...)` gains kwargs (defaulted, MA/trend-preserving): `lookback: int = 0`, `entry_z: float = 0.0`, `exit_z: float = 0.0`, `er_threshold: float = 0.0`. Its `make_strategy` call passes a params dict that includes these four keys (plus the existing `fast/slow/trend/size`), so each strategy reads what it needs.
- `backtest` command: `--lookback` (int), `--entry-z`/`--exit-z`/`--er-threshold` (float). Accept `strategy == "meanrev"`; add a branch (mirroring the trend branch) that calls `run_backtest(..., strategy_name="meanrev", lookback=..., entry_z=..., exit_z=..., er_threshold=..., exit_kind=exit_, ...)` and prints a meanrev summary containing `"meanrev"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_meanrev.py
from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from fxtrader.cli import app
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore

runner = CliRunner()


def _seed(tmp_path):
    store = TickStore(tmp_path)
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    ticks = []
    # oscillating 30s ticks (ranging) so the fader has stretched+reverting moves
    for i in range(2000):
        base = 1.10 + 0.0010 * ((i // 5) % 2)  # flip every 5 ticks -> mean-reverting
        ticks.append(Tick("EURUSD", start + timedelta(seconds=i * 30), base, base + 0.00002))
    store.write_ticks("EURUSD", ticks)


def test_backtest_meanrev_runs_end_to_end(tmp_path):
    _seed(tmp_path)
    result = runner.invoke(app, [
        "backtest", "--strategy", "meanrev", "--symbol", "EURUSD",
        "--start", "2024-01-02", "--end", "2024-01-03",
        "--lookback", "20", "--entry-z", "1.5", "--exit-z", "0.3", "--er-threshold", "0.5",
        "--exit", "fixed", "--stop-pips", "15", "--tp-pips", "30",
        "--bar-minutes", "5", "--data-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "meanrev" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_meanrev.py -v`
Expected: FAIL — `No such option: --strategy ... meanrev` rejected / unknown option `--lookback` (exit code 2).

- [ ] **Step 3: Write minimal implementation**

In `run_backtest` add the four kwargs after `tp_atr: float = 0.0,`:
```python
    lookback: int = 0,
    entry_z: float = 0.0,
    exit_z: float = 0.0,
    er_threshold: float = 0.0,
```
and ensure the `make_strategy` params dict in `run_backtest` includes them, e.g.:
```python
    params={
        "fast": fast, "slow": slow, "trend": trend, "size": size,
        "lookback": lookback, "entry_z": entry_z, "exit_z": exit_z, "er_threshold": er_threshold,
    },
```

In the `backtest` command, add options:
```python
    lookback: int = typer.Option(20, help="Z/ER window (meanrev)"),
    entry_z: float = typer.Option(2.0, "--entry-z", help="Entry z-threshold (meanrev)"),
    exit_z: float = typer.Option(0.5, "--exit-z", help="Exit z-threshold (meanrev)"),
    er_threshold: float = typer.Option(0.3, "--er-threshold", help="Max efficiency ratio to fade (meanrev)"),
```
Change the guard and add the branch (after the trend branch, before the config path):
```python
    if strategy not in ("ma", "trend", "meanrev"):
        raise typer.BadParameter("strategy must be ma | trend | meanrev")
    if strategy == "meanrev":
        if not (symbol and start and end):
            raise typer.BadParameter("meanrev strategy needs --symbol/--start/--end")
        result, metrics = run_backtest(
            symbol=symbol, start=_parse_day(start), end=_parse_day(end), data_dir=data_dir,
            size=size, bar_minutes=bar_minutes, strategy_name="meanrev",
            lookback=lookback, entry_z=entry_z, exit_z=exit_z, er_threshold=er_threshold,
            exit_kind=exit_, stop_pips=stop_pips, tp_pips=tp_pips,
            atr_period=atr_period, stop_atr=stop_atr, tp_atr=tp_atr,
            daily_loss_limit=daily_loss_limit or float("inf"),
            max_position=max_position or float("inf"),
        )
        typer.echo(f"symbol={symbol}  bars={bar_minutes}m  meanrev(lookback={lookback} entry_z={entry_z} exit_z={exit_z} er<{er_threshold})  exit={exit_}")
        typer.echo(f"  trades        : {metrics['num_trades']}")
        typer.echo(f"  total return  : {metrics['total_return']:.2%}")
        typer.echo(f"  Sharpe        : {metrics['sharpe']:.2f}")
        typer.echo(f"  profit factor : {metrics['profit_factor']:.2f}")
        return
```

> Verify the existing trend branch still precedes this and the MA/config path is unchanged. Confirm `run_backtest`'s current `make_strategy` call site (read it) and extend the exact params dict it builds.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cli_meanrev.py -v && .venv/bin/pytest -q`
Expected: new test PASS; full suite green (MA/trend unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/cli.py tests/test_cli_meanrev.py
git commit  # feat(cli): --strategy meanrev backtest with exit options
```

---

## Task 6: CLI — `optimize --strategy meanrev` (float grids)

**Files:**
- Modify: `src/fxtrader/cli.py` (`_strategy_run_fn` meanrev validation + run_backtest call; `run_optimize` grid + new `*_values` params; `optimize` command options + `_float_list`)
- Test: `tests/test_cli_meanrev.py`

**Interfaces:**
- `_float_list(value: str) -> list[float]` (mirrors `_int_list`).
- `run_optimize(...)` gains `lookback_values: list[int] | None = None`, `entry_z_values/exit_z_values/er_threshold_values: list[float] | None = None`. For `strategy == "meanrev"` the grid is `{"lookback":…, "entry_z":…, "exit_z":…, "er_threshold":…}` (NOT fast/slow). For ma/trend the grid is unchanged.
- `_strategy_run_fn` validates per strategy: ma/trend keep the `fast/slow[/trend]` checks; `meanrev` returns `None` unless `lookback ≥ 2 and 0 ≤ exit_z < entry_z and 0 < er_threshold ≤ 1`. The `run_backtest` call passes `fast/slow` via `.get(...)` defaults and forwards the meanrev params.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_cli_meanrev.py
from fxtrader.cli import run_optimize


def test_optimize_meanrev_param_grid(tmp_path):
    _seed(tmp_path)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 3, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy="meanrev",
        lookback_values=[20], entry_z_values=[1.5, 2.0],
        exit_z_values=[0.3], er_threshold_values=[0.5],
        n_splits=2, bar_minutes=5, objective="sharpe",
    )
    assert out["param_grid"] == {
        "lookback": [20], "entry_z": [1.5, 2.0], "exit_z": [0.3], "er_threshold": [0.5],
    }
    assert "folds" in out and "mean_oos_objective" in out


def test_optimize_ma_grid_unchanged(tmp_path):
    _seed(tmp_path)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 3, tzinfo=timezone.utc),
        data_dir=tmp_path, fast_values=[2, 3], slow_values=[5, 8],
        n_splits=2, bar_minutes=5,
    )
    assert out["param_grid"] == {"fast": [2, 3], "slow": [5, 8]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_meanrev.py -k optimize -v`
Expected: FAIL — `run_optimize() got an unexpected keyword argument 'lookback_values'`.

- [ ] **Step 3: Write minimal implementation**

Add `_float_list` next to `_int_list`:
```python
def _float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]
```

In `run_optimize` add the params and the grid branch:
```python
    lookback_values: list[int] | None = None,
    entry_z_values: list[float] | None = None,
    exit_z_values: list[float] | None = None,
    er_threshold_values: list[float] | None = None,
```
```python
    if strategy == "meanrev":
        grid = {
            "lookback": lookback_values or [], "entry_z": entry_z_values or [],
            "exit_z": exit_z_values or [], "er_threshold": er_threshold_values or [],
        }
    else:
        grid = {"fast": fast_values, "slow": slow_values}
        if strategy == "trend":
            grid["trend"] = trend_values or []
```

In `_strategy_run_fn`, make validation strategy-aware and the `run_backtest` call robust to missing `fast/slow`:
```python
    def run_fn(params: dict, start: datetime, end: datetime):
        p = {**base_params, **params}
        if strategy_name in ("ma", "trend"):
            if p["fast"] >= p["slow"]:
                return None
            if strategy_name == "trend" and not (p["fast"] < p["slow"] < p["trend"]):
                return None
        elif strategy_name == "meanrev":
            if not (p["lookback"] >= 2 and 0 <= p["exit_z"] < p["entry_z"] and 0 < p["er_threshold"] <= 1):
                return None
        _result, metrics = run_backtest(
            symbol=symbol, start=start, end=end, data_dir=data_dir,
            fast=p.get("fast", 20), slow=p.get("slow", 50), size=p.get("size", 10_000),
            bar_minutes=bar_minutes, strategy_name=strategy_name,
            trend=p.get("trend", 0), exit_kind=exit_kind,
            stop_pips=p.get("stop_pips", 0.0), tp_pips=p.get("tp_pips", 0.0),
            atr_period=p.get("atr_period", 14), stop_atr=p.get("stop_atr", 0.0),
            tp_atr=p.get("tp_atr", 0.0),
            lookback=p.get("lookback", 0), entry_z=p.get("entry_z", 0.0),
            exit_z=p.get("exit_z", 0.0), er_threshold=p.get("er_threshold", 0.0),
        )
        return metrics
```

In the `optimize` command, add options + accept meanrev + pass `*_values`:
```python
    lookback: str = typer.Option(None, "--lookback", help="Comma-separated z/ER windows (meanrev)"),
    entry_z: str = typer.Option(None, "--entry-z", help="Comma-separated entry z (meanrev)"),
    exit_z: str = typer.Option(None, "--exit-z", help="Comma-separated exit z (meanrev)"),
    er_threshold: str = typer.Option(None, "--er-threshold", help="Comma-separated ER thresholds (meanrev)"),
```
```python
    # in the run_optimize(...) call:
        lookback_values=_int_list(lookback) if lookback else None,
        entry_z_values=_float_list(entry_z) if entry_z else None,
        exit_z_values=_float_list(exit_z) if exit_z else None,
        er_threshold_values=_float_list(er_threshold) if er_threshold else None,
```

> The `optimize` command currently has `*` `--fast/--slow` with defaults; for meanrev those go unused. Do not change their defaults. Confirm the MA grid path (`test_optimize_ma_grid_unchanged` and the existing `tests/test_cli_optimize.py`) stays byte-identical.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cli_meanrev.py -v && .venv/bin/pytest -q`
Expected: both new optimize tests PASS; full suite green (existing optimize tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/cli.py tests/test_cli_meanrev.py
git commit  # feat(cli): --strategy meanrev optimize (float param grids)
```

---

## Self-Review

**Spec coverage:** §3 fader logic → Tasks 2–3. §4 indicators → Task 1. §5 reuse (exits/registry/CLI) → Tasks 3/4/5/6. §7 repo layout → all tasks. §8 testing → each task's tests + the regime-gate test (Task 2 `test_no_entry_when_stretched_but_trending`) and stop-first (Task 3). §10 DoD (suite green, end-to-end meanrev, MA/trend unchanged) → Tasks 5–6. The staged research run (§6) is the intentionally separate job, not a code task.

**Placeholder scan:** Tasks 2/3/5 ask the implementer to tune seed series / derive the entry price with explicit instructions on how (print `zscore`/`efficiency_ratio`, set trigger bar high/low) — these are real procedures, not silent TBDs. No "add validation"-style placeholders.

**Type consistency:** `MeanReversionFader(symbol, lookback, entry_z, exit_z, er_threshold, size, exit_policy)` is identical across Tasks 2, 4. The four `run_backtest` kwargs (`lookback`, `entry_z`, `exit_z`, `er_threshold`) match across Tasks 5 and 6. `STRATEGY_PARAMS["meanrev"]` tuple matches Task 4's definition and Task 6's grid keys. `_float_list` defined once (Task 6). `zscore`/`efficiency_ratio` signatures consistent across Tasks 1 and 2. `_last_exit_reason` values (`"signal"`/`"stop"`/`"tp"`) consistent across Tasks 2–3.

**Known executor caveats (flagged, not placeholders):** the `_drive` harness duplicates the trend test's (acceptable for an isolated test fixture). The Task 5 implementer must read `run_backtest`'s current `make_strategy` call site and extend the exact params dict it builds (the dict is shown). Seed series in Tasks 2/3/5 may need numeric tuning to fire the intended z/ER condition — flagged with the method.
