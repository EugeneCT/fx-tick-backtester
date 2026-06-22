# Cross-Pair Spread Reversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a market-neutral pairs strategy that fades the z-score of `log(EURUSD)−log(GBPUSD)` (two-leg, β=1), wired into the `--strategy` CLI with multi-symbol (`--symbol2`) support — without engine or live-path changes.

**Architecture:** New two-symbol bar-based `Strategy` (`primary`=EURUSD clock, `secondary`=GBPUSD) that samples the log-spread on each primary bar, z-scores it over a rolling window, opens both legs when stretched and closes on reversion or a z-divergence stop. The engine is already multi-symbol (per-symbol `Resampler` dict, symbol-keyed `SimulatedBroker._market`); the wiring threads a second symbol through `run_backtest`/`make_strategy`/`run_optimize`. Spec: `docs/superpowers/specs/2026-06-22-cross-pair-spread-reversion-design.md`.

**Tech Stack:** Python 3.12 (`.venv/bin/...`), polars, typer, pytest. No new dependencies.

## Global Constraints

- Python pinned to **3.12**; invoke tooling as `.venv/bin/python` / `.venv/bin/pytest` / `.venv/bin/fxtrader`.
- **No engine or live-path changes.** Do not edit `engine/*`, `broker/*`, `risk/*`, or `build_live_engine`. The strategy uses the existing multi-symbol engine.
- **DRY / YAGNI / TDD.** Test-first for every code unit. No rolling-β (β=1), no max-hold exit, no `ExitPolicy` for xpair, no live wiring, no USDJPY, no CLI-plumbing generalization (accrete `xpair` flags).
- **Reuse:** the `zscore` indicator (`strategy/indicators.py`); the `--lookback`/`--entry-z`/`--exit-z` CLI options (from meanrev); `_int_list`/`_float_list`.
- **ma/trend/meanrev/orb paths must stay byte-identical.** Single-symbol callers pass `symbol2=None`; new `run_backtest`/`run_optimize` params default harmlessly. The existing optimize grids and their tests are unchanged. Run the FULL suite before each commit (baseline: **203 passed, 2 skipped**).
- All work on branch `feat/cross-pair-spread-reversion` (already created; spec committed at `acd872b`). Do NOT create other branches.
- **Every commit message ends with** (one blank line before it): `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Validation: `lookback ≥ 2`, `0 ≤ exit_z < entry_z < stop_z`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/fxtrader/strategy/examples/pairs_spread_reversion.py` (Create) | `PairsSpreadReversion` strategy |
| `src/fxtrader/strategy/registry.py` (Modify) | `xpair` in `STRATEGY_PARAMS`; `make_strategy` gains `symbol2` + an `xpair` branch |
| `src/fxtrader/cli.py` (Modify) | `run_backtest` `symbol2`/`stop_z` + feed; `--symbol2`/`--stop-z`; optimize `symbol2`/`stop_z_values` + xpair grid/validation |
| `tests/strategy/test_pairs_spread_reversion.py` (Create) | strategy tests + `_drive_pairs` multi-symbol harness |
| `tests/strategy/test_registry.py` (Modify) | xpair factory case |
| `tests/test_cli_xpair.py` (Create) | backtest + optimize end-to-end (2-symbol seed) |

The staged research RUN (WF + OOS, no cross-pair confirmation) is a **separate background job**, not part of this plan.

---

## Task 1: `PairsSpreadReversion` strategy

**Files:**
- Create: `src/fxtrader/strategy/examples/pairs_spread_reversion.py`
- Test: `tests/strategy/test_pairs_spread_reversion.py`

**Interfaces:**
- Consumes: `Strategy` base, `Side`, `zscore`. `self.portfolio.position(sym).quantity`.
- Produces: `PairsSpreadReversion(primary, secondary, lookback, entry_z, exit_z, stop_z, size)`. Raises `ValueError` unless `lookback ≥ 2` and `0 ≤ exit_z < entry_z < stop_z`. Exposes `_last_exit_reason` (`"signal"`/`"stop"`). Trades both legs.

**Test harness** (multi-symbol; include in the new test file):

```python
# tests/strategy/test_pairs_spread_reversion.py
import math
from datetime import datetime, timedelta, timezone

import pytest

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Fill, Order, Side
from fxtrader.strategy.examples.pairs_spread_reversion import PairsSpreadReversion

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _bar(symbol, i, close):
    ts = T0 + timedelta(minutes=i)
    return Bar(symbol, ts, close, close, close, close, 1.0)


def _drive_pairs(strategy, bars):
    """Feed interleaved primary/secondary Bars (time order); fills at bar.close per symbol."""
    pf = Portfolio(100_000.0)
    clock = SimClock()
    orders: list[Order] = []
    px: dict[str, float] = {}

    def sink(order: Order) -> None:
        orders.append(order)
        pf.apply_fill(Fill("x", order.symbol, order.side, order.quantity,
                           px[order.symbol], clock.now(), 0.0))

    strategy.bind(clock, pf, sink)
    strategy.on_start()
    for b in bars:
        clock.set_time(b.ts)
        px[b.symbol] = b.close
        strategy.on_bar(b)
    return pf, orders


def _seq(eur_closes, gbp=1.2500):
    """Interleave (GBP const, EUR varying) bars per period so each EUR bar samples a spread."""
    bars = []
    for i, e in enumerate(eur_closes):
        bars.append(_bar("GBPUSD", 2 * i, gbp))      # secondary first -> sets _last[GBP]
        bars.append(_bar("EURUSD", 2 * i + 1, e))    # primary -> samples spread, may trade
    return bars
```

- [ ] **Step 1: Write the failing tests**

```python
def _x(**kw):
    base = dict(primary="EURUSD", secondary="GBPUSD", lookback=5,
                entry_z=1.0, exit_z=0.3, stop_z=3.0, size=10_000)
    base.update(kw)
    return PairsSpreadReversion(**base)


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        _x(lookback=1)
    with pytest.raises(ValueError):
        _x(entry_z=1.0, exit_z=1.0)        # exit_z !< entry_z
    with pytest.raises(ValueError):
        _x(entry_z=3.0, stop_z=3.0)        # entry_z !< stop_z


def test_no_trade_before_warmup():
    s = _x()
    pf, orders = _drive_pairs(s, _seq([1.10, 1.10, 1.10]))   # < lookback EUR bars
    assert orders == []


def test_secondary_bar_alone_does_not_trade():
    s = _x()
    pf, orders = _drive_pairs(s, [_bar("GBPUSD", 0, 1.25)])
    assert orders == []


def test_enter_short_spread_when_z_high():
    # EUR oscillates (ranging spread) then spikes UP -> spread z high -> SELL EUR + BUY GBP
    s = _x(lookback=5, entry_z=1.0)
    eur = [1.1000, 1.1010, 1.1000, 1.1010, 1.1000, 1.1010, 1.1100]
    pf, orders = _drive_pairs(s, _seq(eur))
    assert pf.position("EURUSD").quantity < 0     # short the rich leg
    assert pf.position("GBPUSD").quantity > 0     # long the cheap leg


def test_enter_long_spread_when_z_low():
    s = _x(lookback=5, entry_z=1.0)
    eur = [1.1000, 1.1010, 1.1000, 1.1010, 1.1000, 1.1010, 1.0900]   # spike DOWN
    pf, _ = _drive_pairs(s, _seq(eur))
    assert pf.position("EURUSD").quantity > 0
    assert pf.position("GBPUSD").quantity < 0


def test_signal_exit_on_reversion():
    s = _x(lookback=5, entry_z=1.0, exit_z=0.3, stop_z=5.0)
    eur = [1.1000, 1.1010, 1.1000, 1.1010, 1.1000, 1.1010, 1.1100, 1.1012]  # spike then revert
    pf, _ = _drive_pairs(s, _seq(eur))
    assert pf.position("EURUSD").quantity == 0
    assert pf.position("GBPUSD").quantity == 0
    assert s._last_exit_reason == "signal"


def test_divergence_stop():
    s = _x(lookback=5, entry_z=1.0, exit_z=0.3, stop_z=2.0)
    # enter then the spread diverges FURTHER (z grows past stop_z) -> cut
    eur = [1.1000, 1.1010, 1.1000, 1.1010, 1.1000, 1.1010, 1.1080, 1.1200]
    pf, _ = _drive_pairs(s, _seq(eur))
    assert pf.position("EURUSD").quantity == 0
    assert s._last_exit_reason == "stop"


def test_ignores_third_symbol():
    s = _x()
    pf, orders = _drive_pairs(s, [_bar("USDJPY", i, 150.0) for i in range(10)])
    assert orders == []
```

> z and the spread are coupled to the EUR series (GBP held constant, so `spread = log(EUR) − log(1.25)`). If a behavioral test doesn't fire, print `zscore([...])` of the spread window at the trigger bar and adjust the EUR seed (spike magnitude / oscillation), NOT the logic. Don't weaken assertions. For `test_divergence_stop`, the second spike must push `|z| ≥ stop_z` while the position is open (it entered on the first spike).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/strategy/test_pairs_spread_reversion.py -v`
Expected: FAIL — module/class not found.

- [ ] **Step 3: Write minimal implementation**

```python
# src/fxtrader/strategy/examples/pairs_spread_reversion.py
"""Market-neutral cross-pair spread reversion: fade z-score of log(primary)-log(secondary)."""

from __future__ import annotations

import math
import statistics
from collections import deque

from fxtrader.core.types import Bar, Side
from fxtrader.strategy.base import Strategy
from fxtrader.strategy.indicators import zscore


class PairsSpreadReversion(Strategy):
    def __init__(
        self,
        primary: str,
        secondary: str,
        lookback: int,
        entry_z: float,
        exit_z: float,
        stop_z: float,
        size: float,
    ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if not (0 <= exit_z < entry_z < stop_z):
            raise ValueError("require 0 <= exit_z < entry_z < stop_z")
        self.primary = primary
        self.secondary = secondary
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.size = size
        self._last: dict[str, float] = {}
        self._spreads: deque[float] = deque(maxlen=lookback)
        self._pos_dir = 0
        self._last_exit_reason: str | None = None

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol == self.secondary:
            self._last[self.secondary] = bar.close
            return
        if bar.symbol != self.primary:
            return
        self._last[self.primary] = bar.close
        if self.secondary not in self._last:
            return
        s = math.log(self._last[self.primary]) - math.log(self._last[self.secondary])
        self._spreads.append(s)
        if len(self._spreads) < self.lookback:
            return
        window = list(self._spreads)
        if statistics.pstdev(window) == 0.0:
            return
        z = zscore(window)

        pos = self.portfolio.position(self.primary).quantity
        if pos != 0:
            if abs(z) <= self.exit_z:
                self._close("signal")
                return
            if abs(z) >= self.stop_z:
                self._close("stop")
                return
        if pos == 0:
            if z > self.entry_z:
                self._open(-1)   # short the spread: SELL primary + BUY secondary
            elif z < -self.entry_z:
                self._open(1)    # long the spread:  BUY primary + SELL secondary

    def _open(self, direction: int) -> None:
        prim_side = Side.BUY if direction > 0 else Side.SELL
        sec_side = Side.SELL if direction > 0 else Side.BUY
        self.order(self.primary, prim_side, self.size)
        self.order(self.secondary, sec_side, self.size)
        self._pos_dir = direction

    def _close(self, reason: str) -> None:
        for sym in (self.primary, self.secondary):
            q = self.portfolio.position(sym).quantity
            if q != 0:
                side = Side.SELL if q > 0 else Side.BUY
                self.order(sym, side, abs(q))
        self._last_exit_reason = reason
        self._pos_dir = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/strategy/test_pairs_spread_reversion.py -v`
Expected: PASS (8 tests). Tune EUR seeds (not logic) if a behavioral case misfires.

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/examples/pairs_spread_reversion.py tests/strategy/test_pairs_spread_reversion.py
git commit  # feat(strategy): cross-pair spread reversion (two-leg, z-reversion + divergence stop)
```

---

## Task 2: Registry — `xpair` factory case + `symbol2`

**Files:**
- Modify: `src/fxtrader/strategy/registry.py`
- Test: `tests/strategy/test_registry.py`

**Interfaces:**
- `make_strategy(name, *, symbol, params, exit_policy=None, symbol2=None)` — adds `symbol2`.
- `STRATEGY_PARAMS["xpair"] = ("lookback", "entry_z", "exit_z", "stop_z")`.
- `make_strategy("xpair", symbol=…, symbol2=…, params={lookback,entry_z,exit_z,stop_z,size}) -> PairsSpreadReversion`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/strategy/test_registry.py
from fxtrader.strategy.examples.pairs_spread_reversion import PairsSpreadReversion


def test_make_strategy_xpair():
    s = make_strategy(
        "xpair", symbol="EURUSD", symbol2="GBPUSD",
        params={"lookback": 20, "entry_z": 2.0, "exit_z": 0.5, "stop_z": 4.0, "size": 10_000},
    )
    assert isinstance(s, PairsSpreadReversion)


def test_strategy_params_has_xpair():
    assert STRATEGY_PARAMS["xpair"] == ("lookback", "entry_z", "exit_z", "stop_z")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/strategy/test_registry.py -k xpair -v`
Expected: FAIL — `KeyError: 'xpair'` / `make_strategy() got an unexpected keyword argument 'symbol2'`.

- [ ] **Step 3: Write minimal implementation**

```python
# import at top of registry.py
from fxtrader.strategy.examples.pairs_spread_reversion import PairsSpreadReversion
```
```python
# add to STRATEGY_PARAMS
    "xpair": ("lookback", "entry_z", "exit_z", "stop_z"),
```
```python
# change make_strategy signature to add symbol2, and add the branch before the final raise:
def make_strategy(name: str, *, symbol: str, params: dict,
                  exit_policy: ExitPolicy | None = None, symbol2: str | None = None) -> Strategy:
    ...
    if name == "xpair":
        return PairsSpreadReversion(
            primary=symbol, secondary=symbol2, lookback=params["lookback"],
            entry_z=params["entry_z"], exit_z=params["exit_z"], stop_z=params["stop_z"],
            size=params["size"],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/strategy/test_registry.py -v`
Expected: PASS (existing + 2 new). Existing ma/trend/meanrev/orb factory tests still pass (they don't pass `symbol2`; default `None` is harmless).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/registry.py tests/strategy/test_registry.py
git commit  # feat(strategy): register xpair (+ symbol2) in the factory
```

---

## Task 3: CLI — `run_backtest` symbol2/stop_z + `backtest --strategy xpair`

**Files:**
- Modify: `src/fxtrader/cli.py` (`run_backtest`: `symbol2`/`stop_z` kwargs, feed, make_strategy call; `backtest` command: `--symbol2`/`--stop-z`, guard, `xpair` branch)
- Test: `tests/test_cli_xpair.py` (new)

**Interfaces:**
- `run_backtest(...)` gains `symbol2: str | None = None`, `stop_z: float = 0.0`. The feed becomes `HistoricalTickFeed(store, [symbol] + ([symbol2] if symbol2 else []), start, end)`; the `make_strategy` call passes `symbol2=symbol2` and the params dict gains `"stop_z": stop_z`.
- `backtest` command: `--symbol2` (str), `--stop-z` (float); guard accepts `"xpair"`; an `xpair` branch routes through `run_backtest(strategy_name="xpair", symbol2=…, lookback=…, entry_z=…, exit_z=…, stop_z=…, exit_kind="none", …)` and prints a summary naming both legs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_xpair.py
from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from fxtrader.cli import app
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore

runner = CliRunner()


def _seed(tmp_path, n=2000):
    """Two-symbol store: GBPUSD ~flat, EURUSD oscillating -> a mean-reverting spread."""
    store = TickStore(tmp_path)
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    eur, gbp = [], []
    for i in range(n):
        ts = start + timedelta(seconds=i * 30)
        e = 1.10 + 0.0010 * ((i // 5) % 2)   # oscillates -> spread mean-reverts
        eur.append(Tick("EURUSD", ts, e, e + 0.00002))
        gbp.append(Tick("GBPUSD", ts, 1.2500, 1.25002))
    store.write_ticks("EURUSD", eur)
    store.write_ticks("GBPUSD", gbp)


def test_backtest_xpair_runs_end_to_end(tmp_path):
    _seed(tmp_path)
    result = runner.invoke(app, [
        "backtest", "--strategy", "xpair", "--symbol", "EURUSD", "--symbol2", "GBPUSD",
        "--start", "2024-01-02", "--end", "2024-01-03",
        "--lookback", "20", "--entry-z", "1.5", "--exit-z", "0.3", "--stop-z", "4.0",
        "--bar-minutes", "5", "--data-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "xpair" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_xpair.py -v`
Expected: FAIL — strategy `"xpair"` rejected / unknown option `--symbol2` (exit 2).

- [ ] **Step 3: Write minimal implementation**

In `run_backtest` add the kwargs (after `exit_hour: int = 0,`):
```python
    symbol2: str | None = None,
    stop_z: float = 0.0,
```
change the feed line:
```python
    feed = HistoricalTickFeed(store, [symbol] + ([symbol2] if symbol2 else []), start, end)
```
pass `symbol2` and `stop_z` to make_strategy:
```python
    strategy = make_strategy(
        strategy_name, symbol=symbol, symbol2=symbol2,
        params={... existing keys ..., "stop_z": stop_z},
        exit_policy=exit_policy,
    )
```

In the `backtest` command, add options:
```python
    symbol2: str = typer.Option(None, "--symbol2", help="Second leg symbol (xpair)"),
    stop_z: float = typer.Option(4.0, "--stop-z", help="Divergence stop z (xpair)"),
```
update the guard and add the branch:
```python
    if strategy not in ("ma", "trend", "meanrev", "orb", "xpair"):
        raise typer.BadParameter("strategy must be ma | trend | meanrev | orb | xpair")
    if strategy == "xpair":
        if not (symbol and symbol2 and start and end):
            raise typer.BadParameter("xpair needs --symbol/--symbol2/--start/--end")
        result, metrics = run_backtest(
            symbol=symbol, symbol2=symbol2, start=_parse_day(start), end=_parse_day(end),
            data_dir=data_dir, size=size, bar_minutes=bar_minutes, strategy_name="xpair",
            lookback=lookback, entry_z=entry_z, exit_z=exit_z, stop_z=stop_z, exit_kind="none",
            daily_loss_limit=daily_loss_limit or float("inf"),
            max_position=max_position or float("inf"),
        )
        typer.echo(f"symbol={symbol}/{symbol2}  bars={bar_minutes}m  xpair(lookback={lookback} entry_z={entry_z} exit_z={exit_z} stop_z={stop_z})")
        typer.echo(f"  trades        : {metrics['num_trades']}")
        typer.echo(f"  total return  : {metrics['total_return']:.2%}")
        typer.echo(f"  Sharpe        : {metrics['sharpe']:.2f}")
        typer.echo(f"  profit factor : {metrics['profit_factor']:.2f}")
        return
```

> `--lookback`/`--entry-z`/`--exit-z` already exist on the `backtest` command (from meanrev). Read the command to confirm and reuse them. Do not alter the ma/config path.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cli_xpair.py -v && .venv/bin/pytest -q`
Expected: new test PASS; full suite green (single-symbol strategies unaffected — they pass `symbol2=None`).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/cli.py tests/test_cli_xpair.py
git commit  # feat(cli): --strategy xpair backtest with --symbol2
```

---

## Task 4: CLI — `optimize --strategy xpair`

**Files:**
- Modify: `src/fxtrader/cli.py` (`run_optimize` symbol2/stop_z_values + xpair grid; `_strategy_run_fn` symbol2/xpair validation + run_backtest call; `optimize` command options)
- Test: `tests/test_cli_xpair.py`

**Interfaces:**
- `run_optimize(...)` gains `symbol2: str | None = None`, `stop_z_values: list[float] | None = None`. For `strategy == "xpair"` the grid is `{"lookback","entry_z","exit_z","stop_z"}`; other grids unchanged. `symbol2` is threaded into `_strategy_run_fn`.
- `_strategy_run_fn(strategy_name, symbol, data_dir, bar_minutes, base_params, exit_kind, symbol2=None)` — add `symbol2`; add an `xpair` validation branch (`return None` unless `lookback≥2 and 0≤exit_z<entry_z<stop_z`); the `run_backtest` call forwards `symbol2=symbol2` and `stop_z=p.get("stop_z", 0.0)`.
- `optimize` command: `--symbol2` (str), `--stop-z` (comma floats); accept `"xpair"`; pass `symbol2` + `stop_z_values`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_cli_xpair.py
from fxtrader.cli import run_optimize


def test_optimize_xpair_param_grid(tmp_path):
    _seed(tmp_path, n=4000)
    out = run_optimize(
        symbol="EURUSD", symbol2="GBPUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 4, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy="xpair",
        lookback_values=[20], entry_z_values=[1.5, 2.0], exit_z_values=[0.3], stop_z_values=[4.0],
        n_splits=2, bar_minutes=5, objective="sharpe",
    )
    assert out["param_grid"] == {
        "lookback": [20], "entry_z": [1.5, 2.0], "exit_z": [0.3], "stop_z": [4.0],
    }
    assert "folds" in out and "mean_oos_objective" in out


def test_optimize_orb_grid_still_works(tmp_path):
    _seed(tmp_path, n=4000)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 4, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy="orb",
        range_start_values=[0], range_end_values=[7], entry_end_values=[16], exit_hour_values=[21],
        n_splits=2, bar_minutes=60,
    )
    assert out["param_grid"] == {"range_start_hour": [0], "range_end_hour": [7],
                                 "entry_end_hour": [16], "exit_hour": [21]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_xpair.py -k optimize -v`
Expected: FAIL — `run_optimize() got an unexpected keyword argument 'stop_z_values'`.

- [ ] **Step 3: Write minimal implementation**

In `run_optimize` add params:
```python
    symbol2: str | None = None,
    stop_z_values: list[float] | None = None,
```
Add the `xpair` grid branch (as the first branch, before `orb`):
```python
    if strategy == "xpair":
        grid = {
            "lookback": lookback_values or [], "entry_z": entry_z_values or [],
            "exit_z": exit_z_values or [], "stop_z": stop_z_values or [],
        }
    elif strategy == "orb":
        grid = { ... }   # unchanged
    elif strategy == "meanrev":
        grid = { ... }   # unchanged
    else:
        grid = {"fast": fast_values, "slow": slow_values}
        if strategy == "trend":
            grid["trend"] = trend_values or []
```
Pass `symbol2` to the run_fn builder:
```python
    run_fn = _strategy_run_fn(strategy, symbol, data_dir, bar_minutes, base_params, exit_kind, symbol2=symbol2)
```

In `_strategy_run_fn` add the `symbol2=None` param, the xpair validation branch, and forward to run_backtest:
```python
def _strategy_run_fn(strategy_name, symbol, data_dir, bar_minutes, base_params, exit_kind, symbol2=None):
    def run_fn(params, start, end):
        p = {**base_params, **params}
        if strategy_name in ("ma", "trend"):
            ...  # unchanged
        elif strategy_name == "meanrev":
            ...  # unchanged
        elif strategy_name == "orb":
            ...  # unchanged
        elif strategy_name == "xpair":
            if not (p["lookback"] >= 2 and 0 <= p["exit_z"] < p["entry_z"] < p["stop_z"]):
                return None
        _result, metrics = run_backtest(
            symbol=symbol, symbol2=symbol2, start=start, end=end, data_dir=data_dir,
            ...  # all existing p.get(...) kwargs unchanged ...
            stop_z=p.get("stop_z", 0.0),
        )
        return metrics
    return run_fn
```

In the `optimize` command, add options + accept xpair + pass through:
```python
    symbol2: str = typer.Option(None, "--symbol2", help="Second leg symbol (xpair)"),
    stop_z: str = typer.Option(None, "--stop-z", help="Comma divergence-stop z (xpair)"),
```
```python
    # in the run_optimize(...) call:
        symbol2=symbol2,
        stop_z_values=_float_list(stop_z) if stop_z else None,
```

> Keep the ma/trend/meanrev/orb grids and validation byte-identical — the xpair branches are additive. `tests/test_cli_optimize.py`, `tests/test_cli_trend.py`, `tests/test_cli_meanrev.py`, `tests/test_cli_orb.py` must stay green.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cli_xpair.py -v && .venv/bin/pytest -q`
Expected: both new tests PASS; full suite green (all prior optimize tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/cli.py tests/test_cli_xpair.py
git commit  # feat(cli): --strategy xpair optimize (two-symbol, stop_z grid)
```

---

## Self-Review

**Spec coverage:** §3 strategy logic (spread, z, two-leg entry, reversion + divergence-stop exits, validation) → Task 1. §4 multi-symbol wiring (run_backtest symbol2/feed, make_strategy symbol2/xpair, optimize threading) → Tasks 2–4. §6 layout → all tasks. §7 testing (warmup, both-leg entry, signal/stop exits, ignores third symbol, registry, CLI e2e + grid) → Tasks 1–4. §9 DoD (suite green, end-to-end xpair, other strategies unchanged) → Tasks 3–4. §5 staged research is the separate job.

**Placeholder scan:** Task 4 uses `# unchanged` / `# ... existing ...` markers for the ma/trend/meanrev/orb branches the implementer must NOT alter — these point at existing code to preserve, with the new additive code given in full. Not TBDs. Task 1 flags seed-tuning with the method (print `zscore`).

**Type consistency:** `PairsSpreadReversion(primary, secondary, lookback, entry_z, exit_z, stop_z, size)` identical across Tasks 1, 2. `make_strategy(..., symbol2=None)` consistent across Tasks 2, 3. `run_backtest(..., symbol2, stop_z)` consistent across Tasks 3, 4. `run_optimize(..., symbol2, stop_z_values)` and `_strategy_run_fn(..., symbol2)` consistent within Task 4. `STRATEGY_PARAMS["xpair"]` tuple matches Task 2's definition and Task 4's grid keys. `_last_exit_reason` values (`"signal"`/`"stop"`) consistent in Task 1.

**Known executor caveats (flagged):** the `_drive_pairs` harness is a new multi-symbol fixture (distinct from `_drive`/`_drive_bars`); it interleaves two symbols' bars. Task 1 seed series (EUR varying, GBP constant) may need numeric tuning to hit target z — flagged with the print-zscore method. Tasks 3–4 thread `symbol2` through several functions; the implementer must read the current `run_backtest`/`run_optimize`/`_strategy_run_fn` to place the additive code without touching the single-symbol branches.
