# Opening-Range Breakout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Opening-Range Breakout strategy — a daily, time-anchored session strategy (build the Asian-session range, break it during London, exit at a cutoff hour or via the reused stop/TP `ExitPolicy`) — wired into the `--strategy` CLI, with no engine or live-path changes.

**Architecture:** New single-symbol bar-based `Strategy` driven by a per-day state machine off `bar.ts` (UTC hour + date): accumulate the range over `[range_start,range_end)`, take one confirmed-close break during `[range_end,entry_end)`, exit on the reused price stop/TP (`ExitPolicy`) or at `exit_hour` (a time exit owned by the strategy). Registry and CLI accrete an `orb` case (all-integer hour params, reusing `_int_list`). Spec: `docs/superpowers/specs/2026-06-22-opening-range-breakout-design.md`.

**Tech Stack:** Python 3.12 (`.venv/bin/...`), polars, typer, pytest. No new dependencies.

## Global Constraints

- Python pinned to **3.12**; invoke tooling as `.venv/bin/python` / `.venv/bin/pytest` / `.venv/bin/fxtrader` (shell `source` does not persist).
- **No engine or live-path changes.** Do not edit `engine/*`, `broker/*`, `risk/*`, or `build_live_engine`. The time exit is strategy-internal; price stop/TP reuses `ExitPolicy`.
- **DRY / YAGNI / TDD.** Test-first for every code unit. No DST-aware boundaries (fixed UTC hours), no break buffer, no same-day re-entry (one trade/day), no live wiring, no USDJPY, no generalizing the CLI param plumbing (accrete `orb` int flags).
- **Reuse:** `ExitPolicy` (`strategy/exits.py`), `pip_size`, and the `_entry_price`/`_stop_dist`/`_tp_dist`/`_pos_dir`/`_last_exit_reason` + stop-first same-bar pattern from `strategy/examples/mean_reversion_fader.py`.
- **MA/trend/meanrev paths must stay byte-identical.** The existing optimize grids (`ma`={"fast","slow"}, `trend`=+"trend", `meanrev`={"lookback","entry_z","exit_z","er_threshold"}) and all their tests are unchanged. Run the FULL suite before each commit (baseline: **188 passed, 2 skipped**).
- All work on branch `feat/opening-range-breakout` (already created; spec committed at `23e458c`). Do NOT create other branches.
- **Every commit message ends with this trailer** (one blank line before it):
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Hour params are UTC ints in `[0,24)` with `range_start < range_end <= entry_end < exit_hour`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/fxtrader/strategy/examples/opening_range_breakout.py` (Create) | `OpeningRangeBreakout` strategy |
| `src/fxtrader/strategy/registry.py` (Modify) | `orb` in `STRATEGY_PARAMS` + `make_strategy` |
| `src/fxtrader/cli.py` (Modify) | `run_backtest` orb kwargs; `backtest`/`optimize` orb options + grid + `_strategy_run_fn` orb validation |
| `tests/strategy/test_opening_range_breakout.py` (Create) | strategy tests + `_drive_bars` time-aware harness |
| `tests/strategy/test_registry.py` (Modify) | orb factory case |
| `tests/test_cli_orb.py` (Create) | backtest + optimize end-to-end |

The staged research RUN (EURUSD+GBPUSD → `research/findings-<date>.md`) is a **separate background job**, not part of this plan.

---

## Task 1: `OpeningRangeBreakout` — daily state machine + time exit

**Files:**
- Create: `src/fxtrader/strategy/examples/opening_range_breakout.py`
- Test: `tests/strategy/test_opening_range_breakout.py`

**Interfaces:**
- Consumes: `Strategy` base, `Bar`/`Side`, `ExitPolicy`/`NoExit`.
- Produces: `OpeningRangeBreakout(symbol, range_start_hour, range_end_hour, entry_end_hour, exit_hour, size, exit_policy=None)`. Raises `ValueError` unless all hours ∈ `[0,24)` and `range_start_hour < range_end_hour <= entry_end_hour < exit_hour`. Builds the range over `[range_start,range_end)`, takes one confirmed-close break during `[range_end,entry_end)`, time-exits at `hour >= exit_hour`. Captures `_entry_price`/`_stop_dist`/`_tp_dist` at entry; exposes `_last_exit_reason` (Task 2 reuses these for stop/TP). A `# (Task 2 inserts the stop/TP exit check HERE ...)` comment marks the stop/TP slot (before the time exit).

**Test harness** (time-aware; include in the new test file):

```python
# tests/strategy/test_opening_range_breakout.py
from datetime import datetime, timezone

import pytest

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Fill, Order, Side
from fxtrader.strategy.exits import FixedPips
from fxtrader.strategy.examples.opening_range_breakout import OpeningRangeBreakout


def _bar(hour, o, hi, lo, c, *, day=2, symbol="EURUSD"):
    ts = datetime(2025, 1, day, hour, 0, tzinfo=timezone.utc)
    return Bar(symbol, ts, o, hi, lo, c, 1.0)


def _drive_bars(strategy, bars):
    """Feed explicit Bars (chosen UTC ts); fills execute at bar.close into the portfolio."""
    pf = Portfolio(100_000.0)
    clock = SimClock()
    orders: list[Order] = []
    px = {"v": 0.0}

    def sink(order: Order) -> None:
        orders.append(order)
        pf.apply_fill(Fill("x", order.symbol, order.side, order.quantity, px["v"], clock.now(), 0.0))

    strategy.bind(clock, pf, sink)
    strategy.on_start()
    for b in bars:
        clock.set_time(b.ts)
        px["v"] = b.close
        strategy.on_bar(b)
    return pf, orders
```

- [ ] **Step 1: Write the failing tests**

```python
def _orb(**kw):
    base = dict(symbol="EURUSD", range_start_hour=0, range_end_hour=7,
                entry_end_hour=16, exit_hour=21, size=10_000)
    base.update(kw)
    return OpeningRangeBreakout(**base)


def test_rejects_bad_hours():
    with pytest.raises(ValueError):
        _orb(range_start_hour=7, range_end_hour=7)         # range_start !< range_end
    with pytest.raises(ValueError):
        _orb(range_end_hour=17, entry_end_hour=16)         # range_end > entry_end
    with pytest.raises(ValueError):
        _orb(exit_hour=24)                                 # hour out of [0,24)


def test_no_trade_during_range_window():
    s = _orb()
    # bars only inside the Asian window 0..6
    pf, orders = _drive_bars(s, [_bar(h, 1.10, 1.10, 1.10, 1.10) for h in range(0, 7)])
    assert orders == []


def test_breakout_long_after_range():
    s = _orb()
    bars = [_bar(h, 1.10, 1.1010, 1.0990, 1.10) for h in range(0, 7)]  # range hi=1.1010 lo=1.0990
    bars.append(_bar(8, 1.10, 1.1030, 1.10, 1.1025))                    # close 1.1025 > hi -> LONG
    pf, _ = _drive_bars(s, bars)
    assert pf.position("EURUSD").quantity > 0


def test_breakout_short_after_range():
    s = _orb()
    bars = [_bar(h, 1.10, 1.1010, 1.0990, 1.10) for h in range(0, 7)]
    bars.append(_bar(8, 1.10, 1.10, 1.0970, 1.0975))                    # close 1.0975 < lo -> SHORT
    pf, _ = _drive_bars(s, bars)
    assert pf.position("EURUSD").quantity < 0


def test_one_trade_per_day():
    s = _orb()
    bars = [_bar(h, 1.10, 1.1010, 1.0990, 1.10) for h in range(0, 7)]
    bars.append(_bar(8, 1.10, 1.1030, 1.10, 1.1025))   # long entry
    bars.append(_bar(9, 1.10, 1.1040, 1.10, 1.1035))   # another break — must NOT add a 2nd entry
    pf, orders = _drive_bars(s, bars)
    assert len(orders) == 1


def test_no_entry_after_entry_window():
    s = _orb(entry_end_hour=9)
    bars = [_bar(h, 1.10, 1.1010, 1.0990, 1.10) for h in range(0, 7)]
    bars.append(_bar(10, 1.10, 1.1030, 1.10, 1.1025))  # hour 10 >= entry_end 9 -> no entry
    pf, orders = _drive_bars(s, bars)
    assert orders == []


def test_time_exit_flattens():
    s = _orb()
    bars = [_bar(h, 1.10, 1.1010, 1.0990, 1.10) for h in range(0, 7)]
    bars.append(_bar(8, 1.10, 1.1030, 1.10, 1.1025))   # long
    bars.append(_bar(21, 1.1025, 1.1025, 1.1025, 1.1025))  # hour 21 >= exit_hour -> flatten
    pf, _ = _drive_bars(s, bars)
    assert pf.position("EURUSD").quantity == 0
    assert s._last_exit_reason == "time"


def test_day_rollover_resets_and_allows_new_entry():
    s = _orb()
    day1 = [_bar(h, 1.10, 1.1010, 1.0990, 1.10) for h in range(0, 7)] + \
           [_bar(8, 1.10, 1.1030, 1.10, 1.1025), _bar(21, 1.1025, 1.1025, 1.1025, 1.1025)]
    day2 = [_bar(h, 1.10, 1.1010, 1.0990, 1.10, day=3) for h in range(0, 7)] + \
           [_bar(8, 1.10, 1.1030, 1.10, 1.1025, day=3)]
    pf, orders = _drive_bars(s, day1 + day2)
    # day1: entry + time-exit (2 orders); day2: a fresh entry (1 more)
    assert len(orders) == 3
    assert pf.position("EURUSD").quantity > 0


def test_ignores_other_symbols():
    s = _orb()
    bars = [_bar(h, 1.10, 1.1010, 1.0990, 1.10, symbol="GBPUSD") for h in range(0, 7)]
    bars.append(_bar(8, 1.10, 1.1030, 1.10, 1.1025, symbol="GBPUSD"))
    pf, orders = _drive_bars(s, bars)
    assert orders == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/strategy/test_opening_range_breakout.py -v`
Expected: FAIL — module/class not found.

- [ ] **Step 3: Write minimal implementation**

```python
# src/fxtrader/strategy/examples/opening_range_breakout.py
"""Opening-Range Breakout: build the Asian-session range, break it during London (one trade/day)."""

from __future__ import annotations

from fxtrader.core.types import Bar, Side
from fxtrader.strategy.base import Strategy
from fxtrader.strategy.exits import ExitPolicy, NoExit


class OpeningRangeBreakout(Strategy):
    def __init__(
        self,
        symbol: str,
        range_start_hour: int,
        range_end_hour: int,
        entry_end_hour: int,
        exit_hour: int,
        size: float,
        exit_policy: ExitPolicy | None = None,
    ) -> None:
        hours = (range_start_hour, range_end_hour, entry_end_hour, exit_hour)
        if not all(0 <= h < 24 for h in hours):
            raise ValueError("hours must be in [0, 24)")
        if not (range_start_hour < range_end_hour <= entry_end_hour < exit_hour):
            raise ValueError("require range_start < range_end <= entry_end < exit_hour")
        self.symbol = symbol
        self.range_start_hour = range_start_hour
        self.range_end_hour = range_end_hour
        self.entry_end_hour = entry_end_hour
        self.exit_hour = exit_hour
        self.size = size
        self.exit_policy: ExitPolicy = exit_policy or NoExit()
        self._day = None
        self._hi: float | None = None
        self._lo: float | None = None
        self._traded = False
        self._pos_dir = 0
        self._entry_price: float | None = None
        self._stop_dist: float | None = None
        self._tp_dist: float | None = None
        self._last_exit_reason: str | None = None

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol != self.symbol:
            return
        self.exit_policy.update(bar)
        day = bar.ts.date()
        hour = bar.ts.hour
        if day != self._day:
            self._day = day
            self._hi = self._lo = None
            self._traded = False
        position = self.portfolio.position(self.symbol).quantity

        # 1. EXIT FIRST
        # (Task 2 inserts the stop/TP exit check HERE, before the time exit.)
        if position != 0 and hour >= self.exit_hour:
            self._flatten(position, "time")
            return

        # 2. RANGE BUILD (Asian window) — never trade here
        if self.range_start_hour <= hour < self.range_end_hour:
            self._hi = bar.high if self._hi is None else max(self._hi, bar.high)
            self._lo = bar.low if self._lo is None else min(self._lo, bar.low)
            return

        # 3. BREAKOUT ENTRY (one per day, London window)
        if (self._hi is not None and not self._traded and position == 0
                and self.range_end_hour <= hour < self.entry_end_hour):
            desired = 0.0
            if bar.close > self._hi:
                desired = self.size
            elif bar.close < self._lo:
                desired = -self.size
            if desired != 0.0:
                self._traded = True
                self._enter(bar, desired)

    def _enter(self, bar: Bar, desired: float) -> None:
        side = Side.BUY if desired > 0 else Side.SELL
        self.order(self.symbol, side, abs(desired))   # entered from flat -> qty == |desired|
        self._pos_dir = 1 if desired > 0 else -1
        self._entry_price = bar.close
        self._stop_dist, self._tp_dist = self.exit_policy.distances()

    def _flatten(self, position: float, reason: str) -> None:
        side = Side.SELL if position > 0 else Side.BUY
        self.order(self.symbol, side, abs(position))
        self._last_exit_reason = reason
        self._pos_dir = 0
        self._entry_price = None
        self._stop_dist = self._tp_dist = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/strategy/test_opening_range_breakout.py -v`
Expected: PASS (9 tests). If a behavioral test misfires, adjust the seed *bars* (hours/prices), not the logic.

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/examples/opening_range_breakout.py tests/strategy/test_opening_range_breakout.py
git commit  # feat(strategy): opening-range breakout (daily state machine + time exit)
```

---

## Task 2: ORB stop/TP exit overlay

**Files:**
- Modify: `src/fxtrader/strategy/examples/opening_range_breakout.py`
- Test: `tests/strategy/test_opening_range_breakout.py`

**Interfaces:**
- Inserts the price stop/TP check at the marked slot, **before** the time exit. Long: stop if `bar.low <= entry - stop_dist`, TP if `bar.high >= entry + tp_dist`; short mirrored; **same-bar both touched → stop wins** (stop leg before TP). On a hit: `self._flatten(position, "stop"|"tp")` and `return`. `None` legs don't trigger.

- [ ] **Step 1: Write the failing test**

```python
def test_stop_hit_flattens_before_time_exit():
    s = _orb(exit_policy=FixedPips(stop_pips=20, tp_pips=500, pip_size=0.0001))  # stop 0.0020
    bars = [_bar(h, 1.10, 1.1010, 1.0990, 1.10) for h in range(0, 7)]
    bars.append(_bar(8, 1.10, 1.1030, 1.10, 1.1025))   # LONG entry at 1.1025
    # next bar dips: low 1.1000 <= entry(1.1025) - 0.0020 = 1.1005 -> stop; high far from TP
    bars.append(_bar(9, 1.1025, 1.1026, 1.1000, 1.1010))
    pf, _ = _drive_bars(s, bars)
    assert pf.position("EURUSD").quantity == 0
    assert s._last_exit_reason == "stop"
```

> Verify the entry close (1.1025) and that the trigger bar's `low` (1.1000) breaches `entry - stop_dist` (1.1005) while `high` stays well below the TP (1.1025 + 0.05). The test must fail if the stop inequality is wrong. Keep assertions strict.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/strategy/test_opening_range_breakout.py -k stop_hit -v`
Expected: FAIL — position still open / `_last_exit_reason` not "stop" (no stop logic yet; the bar-9 position would instead survive to be time-exited later or stay open).

- [ ] **Step 3: Write minimal implementation**

Replace the `# (Task 2 inserts the stop/TP exit check HERE ...)` comment with:

```python
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

Run: `.venv/bin/pytest tests/strategy/test_opening_range_breakout.py -v`
Expected: PASS (all ORB tests).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/examples/opening_range_breakout.py tests/strategy/test_opening_range_breakout.py
git commit  # feat(strategy): ORB stop/TP overlay (risk-first, before time exit)
```

---

## Task 3: Registry — `orb` factory case

**Files:**
- Modify: `src/fxtrader/strategy/registry.py`
- Test: `tests/strategy/test_registry.py`

**Interfaces:**
- `STRATEGY_PARAMS["orb"] = ("range_start_hour", "range_end_hour", "entry_end_hour", "exit_hour")`.
- `make_strategy("orb", symbol=..., params={...4 hours + "size"}, exit_policy=...) -> OpeningRangeBreakout`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/strategy/test_registry.py
from fxtrader.strategy.examples.opening_range_breakout import OpeningRangeBreakout


def test_make_strategy_orb():
    s = make_strategy(
        "orb", symbol="EURUSD",
        params={"range_start_hour": 0, "range_end_hour": 7, "entry_end_hour": 16,
                "exit_hour": 21, "size": 10_000},
        exit_policy=FixedPips(10, 20, 0.0001),
    )
    assert isinstance(s, OpeningRangeBreakout)


def test_strategy_params_has_orb():
    assert STRATEGY_PARAMS["orb"] == ("range_start_hour", "range_end_hour", "entry_end_hour", "exit_hour")
```

(`FixedPips`, `STRATEGY_PARAMS`, `make_strategy` are already imported in this test file from the meanrev task — confirm and add only what's missing.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/strategy/test_registry.py -k orb -v`
Expected: FAIL — `KeyError: 'orb'` / `ValueError: unknown strategy 'orb'`.

- [ ] **Step 3: Write minimal implementation**

In `src/fxtrader/strategy/registry.py`: add the import, the `STRATEGY_PARAMS` entry, and the `make_strategy` branch (before the final `raise ValueError`).

```python
from fxtrader.strategy.examples.opening_range_breakout import OpeningRangeBreakout
```
```python
    "orb": ("range_start_hour", "range_end_hour", "entry_end_hour", "exit_hour"),
```
```python
    if name == "orb":
        return OpeningRangeBreakout(
            symbol=symbol, range_start_hour=params["range_start_hour"],
            range_end_hour=params["range_end_hour"], entry_end_hour=params["entry_end_hour"],
            exit_hour=params["exit_hour"], size=params["size"], exit_policy=exit_policy,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/strategy/test_registry.py -v`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/registry.py tests/strategy/test_registry.py
git commit  # feat(strategy): register orb in the strategy factory
```

---

## Task 4: CLI — `run_backtest` orb kwargs + `backtest --strategy orb`

**Files:**
- Modify: `src/fxtrader/cli.py` (`run_backtest` signature + its `make_strategy` params dict; `backtest` command options + guard + an `orb` branch)
- Test: `tests/test_cli_orb.py` (new)

**Interfaces:**
- `run_backtest(...)` gains kwargs (defaulted, other-strategy-preserving): `range_start_hour: int = 0`, `range_end_hour: int = 0`, `entry_end_hour: int = 0`, `exit_hour: int = 0`; the `make_strategy` params dict gains those four keys.
- `backtest` command: `--range-start`/`--range-end`/`--entry-end`/`--exit-hour` (ints); accept `strategy == "orb"` (extend the guard to `("ma","trend","meanrev","orb")`); add an `orb` branch (mirroring the meanrev branch) that calls `run_backtest(..., strategy_name="orb", range_start_hour=..., ...)` and prints a summary containing `"orb"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_orb.py
from datetime import datetime, timedelta, timezone

from typer.testing import CliRunner

from fxtrader.cli import app
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore

runner = CliRunner()


def _seed(tmp_path, days=6):
    """Daily pattern: flat 1.1000 during 00:00-06:59 UTC (the range), then 1.1100 from 07:00
    (a clean upside break), back to 1.1000 after 21:00. 30-min ticks across each weekday."""
    store = TickStore(tmp_path)
    ticks = []
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    for d in range(days):
        day0 = base + timedelta(days=d)
        for slot in range(48):  # 30-min slots across 24h
            ts = day0 + timedelta(minutes=30 * slot)
            hour = ts.hour
            px = 1.1100 if 7 <= hour < 21 else 1.1000
            ticks.append(Tick("EURUSD", ts, px, px + 0.00002))
    store.write_ticks("EURUSD", ticks)


def test_backtest_orb_runs_end_to_end(tmp_path):
    _seed(tmp_path)
    result = runner.invoke(app, [
        "backtest", "--strategy", "orb", "--symbol", "EURUSD",
        "--start", "2024-01-01", "--end", "2024-01-08",
        "--range-start", "0", "--range-end", "7", "--entry-end", "16", "--exit-hour", "21",
        "--exit", "fixed", "--stop-pips", "30", "--tp-pips", "60",
        "--bar-minutes", "30", "--data-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "orb" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_orb.py -v`
Expected: FAIL — strategy `"orb"` rejected / unknown option `--range-start` (exit 2).

- [ ] **Step 3: Write minimal implementation**

In `run_backtest` add the four kwargs after `er_threshold: float = 0.0,`:
```python
    range_start_hour: int = 0,
    range_end_hour: int = 0,
    entry_end_hour: int = 0,
    exit_hour: int = 0,
```
and extend the `make_strategy` params dict (the one already containing fast/slow/trend/size/lookback/entry_z/exit_z/er_threshold):
```python
            "range_start_hour": range_start_hour, "range_end_hour": range_end_hour,
            "entry_end_hour": entry_end_hour, "exit_hour": exit_hour,
```

In the `backtest` command, add options:
```python
    range_start: int = typer.Option(0, "--range-start", help="Range-build start hour UTC (orb)"),
    range_end: int = typer.Option(7, "--range-end", help="Range-build end / entry start hour UTC (orb)"),
    entry_end: int = typer.Option(16, "--entry-end", help="Stop taking entries after this hour UTC (orb)"),
    exit_hour: int = typer.Option(21, "--exit-hour", help="Flatten-all hour UTC (orb)"),
```
Update the guard and add the branch (place it alongside the other strategy branches, before the MA/config path):
```python
    if strategy not in ("ma", "trend", "meanrev", "orb"):
        raise typer.BadParameter("strategy must be ma | trend | meanrev | orb")
    if strategy == "orb":
        if not (symbol and start and end):
            raise typer.BadParameter("orb strategy needs --symbol/--start/--end")
        result, metrics = run_backtest(
            symbol=symbol, start=_parse_day(start), end=_parse_day(end), data_dir=data_dir,
            size=size, bar_minutes=bar_minutes, strategy_name="orb",
            range_start_hour=range_start, range_end_hour=range_end,
            entry_end_hour=entry_end, exit_hour=exit_hour,
            exit_kind=exit_, stop_pips=stop_pips, tp_pips=tp_pips,
            atr_period=atr_period, stop_atr=stop_atr, tp_atr=tp_atr,
            daily_loss_limit=daily_loss_limit or float("inf"),
            max_position=max_position or float("inf"),
        )
        typer.echo(f"symbol={symbol}  bars={bar_minutes}m  orb(range={range_start}-{range_end} entry<{entry_end} exit@{exit_hour})  exit={exit_}")
        typer.echo(f"  trades        : {metrics['num_trades']}")
        typer.echo(f"  total return  : {metrics['total_return']:.2%}")
        typer.echo(f"  Sharpe        : {metrics['sharpe']:.2f}")
        typer.echo(f"  profit factor : {metrics['profit_factor']:.2f}")
        return
```

> Read the current `backtest` command (cli.py:329+) to confirm the existing `meanrev`/`trend` branch placement and the option names (`exit_`, etc.); mirror them. Do not alter the ma/config path.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cli_orb.py -v && .venv/bin/pytest -q`
Expected: new test PASS; full suite green (ma/trend/meanrev unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/cli.py tests/test_cli_orb.py
git commit  # feat(cli): --strategy orb backtest with exit options
```

---

## Task 5: CLI — `optimize --strategy orb`

**Files:**
- Modify: `src/fxtrader/cli.py` (`run_optimize` orb grid + `*_hour_values` params; `_strategy_run_fn` orb validation + run_backtest call; `optimize` command options)
- Test: `tests/test_cli_orb.py`

**Interfaces:**
- `run_optimize(...)` gains `range_start_values`, `range_end_values`, `entry_end_values`, `exit_hour_values: list[int] | None = None`. For `strategy == "orb"` the grid is `{"range_start_hour":…, "range_end_hour":…, "entry_end_hour":…, "exit_hour":…}`; ma/trend/meanrev grids unchanged.
- `_strategy_run_fn`: add an `orb` validation branch — return `None` unless `0 <= range_start < range_end <= entry_end < exit_hour < 24`; forward the four orb params to `run_backtest` via `p.get(...)`.
- `optimize` command: add `--range-start`/`--range-end`/`--entry-end`/`--exit-hour` (comma int strings via `_int_list`), accept `"orb"`, pass `*_values`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_cli_orb.py
from fxtrader.cli import run_optimize


def test_optimize_orb_param_grid(tmp_path):
    _seed(tmp_path, days=20)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 20, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy="orb",
        range_start_values=[0], range_end_values=[7],
        entry_end_values=[16], exit_hour_values=[20, 21],
        n_splits=2, bar_minutes=30, objective="sharpe",
    )
    assert out["param_grid"] == {
        "range_start_hour": [0], "range_end_hour": [7],
        "entry_end_hour": [16], "exit_hour": [20, 21],
    }
    assert "folds" in out and "mean_oos_objective" in out


def test_optimize_meanrev_grid_still_works(tmp_path):
    _seed(tmp_path, days=20)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 20, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy="meanrev",
        lookback_values=[20], entry_z_values=[2.0], exit_z_values=[0.5], er_threshold_values=[0.3],
        n_splits=2, bar_minutes=30,
    )
    assert out["param_grid"] == {"lookback": [20], "entry_z": [2.0], "exit_z": [0.5], "er_threshold": [0.3]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_orb.py -k optimize -v`
Expected: FAIL — `run_optimize() got an unexpected keyword argument 'range_start_values'`.

- [ ] **Step 3: Write minimal implementation**

In `run_optimize` add the params:
```python
    range_start_values: list[int] | None = None,
    range_end_values: list[int] | None = None,
    entry_end_values: list[int] | None = None,
    exit_hour_values: list[int] | None = None,
```
Add the `orb` grid branch (as the first branch, before the `meanrev` branch):
```python
    if strategy == "orb":
        grid = {
            "range_start_hour": range_start_values or [], "range_end_hour": range_end_values or [],
            "entry_end_hour": entry_end_values or [], "exit_hour": exit_hour_values or [],
        }
    elif strategy == "meanrev":
        grid = { ... }   # unchanged
    else:
        grid = {"fast": fast_values, "slow": slow_values}
        if strategy == "trend":
            grid["trend"] = trend_values or []
```

In `_strategy_run_fn`, add the orb validation branch (after the meanrev `elif`) and forward the params in the `run_backtest` call:
```python
        elif strategy_name == "orb":
            if not (0 <= p["range_start_hour"] < p["range_end_hour"] <= p["entry_end_hour"] < p["exit_hour"] < 24):
                return None
```
and in the `run_backtest(...)` call add:
```python
            range_start_hour=p.get("range_start_hour", 0), range_end_hour=p.get("range_end_hour", 0),
            entry_end_hour=p.get("entry_end_hour", 0), exit_hour=p.get("exit_hour", 0),
```

In the `optimize` command, add options + accept orb + pass `*_values`:
```python
    range_start: str = typer.Option(None, "--range-start", help="Comma range-start hours (orb)"),
    range_end: str = typer.Option(None, "--range-end", help="Comma range-end hours (orb)"),
    entry_end: str = typer.Option(None, "--entry-end", help="Comma entry-end hours (orb)"),
    exit_hour: str = typer.Option(None, "--exit-hour", help="Comma exit hours (orb)"),
```
```python
    # in the run_optimize(...) call:
        range_start_values=_int_list(range_start) if range_start else None,
        range_end_values=_int_list(range_end) if range_end else None,
        entry_end_values=_int_list(entry_end) if entry_end else None,
        exit_hour_values=_int_list(exit_hour) if exit_hour else None,
```

> Confirm the ma/trend/meanrev grid branches and the `_strategy_run_fn` ma/trend/meanrev validation are unchanged (existing `tests/test_cli_optimize.py`, `tests/test_cli_meanrev.py`, `tests/test_cli_trend.py` assert those grids). The orb branch is additive.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cli_orb.py -v && .venv/bin/pytest -q`
Expected: both new tests PASS; full suite green (all prior optimize tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/cli.py tests/test_cli_orb.py
git commit  # feat(cli): --strategy orb optimize (hour param grids)
```

---

## Self-Review

**Spec coverage:** §3 ORB logic → Tasks 1–2. §4 reuse (ExitPolicy/registry/CLI) → Tasks 2/3/4/5. §6 repo layout → all tasks. §7 testing (range/entry/one-per-day/time-exit/stop-first/day-rollover/ValueError) → Tasks 1–2 tests; CLI end-to-end → Tasks 4–5. §9 DoD (suite green, end-to-end orb, other strategies unchanged) → Tasks 4–5. §5 staged research run is the intentionally separate job.

**Placeholder scan:** Tasks 1/2/4 reference reading the current `backtest` command to mirror branch placement, with the exact code to add given inline — a real procedure, not a TBD. No "add validation"-style placeholders.

**Type consistency:** `OpeningRangeBreakout(symbol, range_start_hour, range_end_hour, entry_end_hour, exit_hour, size, exit_policy)` identical across Tasks 1, 3. The four `run_backtest` kwargs (`range_start_hour`/`range_end_hour`/`entry_end_hour`/`exit_hour`) match across Tasks 4 and 5. `STRATEGY_PARAMS["orb"]` tuple matches Task 3's definition and Task 5's grid keys. The `run_optimize` `*_values` names (`range_start_values`/`range_end_values`/`entry_end_values`/`exit_hour_values`) are consistent between Task 5's signature and its call. `_last_exit_reason` values (`"time"`/`"stop"`/`"tp"`) consistent across Tasks 1–2.

**Known executor caveats (flagged):** the `_drive_bars` harness is a new time-aware fixture (distinct from `_drive`); it builds bars at explicit UTC timestamps. The Task 4/5 CLI seed produces a clean daily up-break so orb reliably trades; if warmup/fold sizing yields too few trading days, widen `days=`. Branch/option placement in `cli.py` must mirror the existing meanrev/trend cases — the exact code is given; read the surrounding command first.
