# P4 — Quant Research Agent + Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A walk-forward parameter optimizer (programmatic) plus a Claude Code subagent that drives it — proposing strategy ideas, running backtests/optimizations via the CLI, reading `results.json`, and reporting findings. Human approves before anything goes live.

**Architecture:** Pure, dependency-injected search functions (`grid_search`, `walk_forward_optimize`) take a `run_fn(params, start, end) -> metrics` so they never import the CLI (no circular dependency) and are fully testable with a fake run function. A `walk_forward_splits` helper produces train/test folds. The CLI builds the real `run_fn` from P2a's `run_backtest`, exposes `fxtrader optimize`, and writes a strict-JSON result. A `.claude/agents/quant-researcher.md` subagent definition drives the whole thing in a human-in-the-loop research loop.

**Tech Stack:** Python 3.12 (`.venv`), stdlib only for the optimizer (itertools, dataclasses) — **no Optuna** (deterministic grid/walk-forward; Optuna is a documented future drop-in for continuous spaces). Reuses P2a `run_backtest` and P2b `_json_safe`. pytest. The subagent uses the existing `fxtrader` CLI.

**Conventions:**
- All timestamps tz-aware UTC. Use `.venv/bin/pytest` / `.venv/bin/python` (shell `source` doesn't persist). pytest uses `--import-mode=importlib`.
- Optimizer functions are PURE (inject `run_fn`) — never import `fxtrader.cli` (avoids a cycle). The CLI wires the real run function.
- Strict JSON output (reuse `fxtrader.metrics.report._json_safe`).
- Commit after each task. No placeholders.

---

## File structure

```
src/fxtrader/research/__init__.py        # Task 1
src/fxtrader/research/walkforward.py     # Task 1 — Fold, walk_forward_splits
src/fxtrader/research/optimize.py        # Task 2 — grid_search
                                         # Task 3 — walk_forward_optimize
src/fxtrader/cli.py                      # Task 4 — optimize command (MODIFY)
.claude/agents/quant-researcher.md       # Task 5 — the subagent
tests/research/test_walkforward.py       # Task 1
tests/research/test_optimize.py          # Tasks 2 & 3
tests/test_cli_optimize.py               # Task 4
```

---

## Task 1: Walk-forward splitter

**Files:**
- Create: `src/fxtrader/research/__init__.py`
- Create: `src/fxtrader/research/walkforward.py`
- Create: `tests/research/test_walkforward.py`

`walk_forward_splits` divides `[start, end)` into `n_splits` equal contiguous windows; within each, the first `train_ratio` fraction is the train period and the remainder is the (out-of-sample) test period.

- [ ] **Step 1: Write the failing test `tests/research/test_walkforward.py`**

```python
from datetime import datetime, timezone

import pytest

from fxtrader.research.walkforward import Fold, walk_forward_splits


def _dt(month, day):
    return datetime(2024, month, day, tzinfo=timezone.utc)


def test_single_split_70_30():
    start, end = _dt(1, 1), _dt(1, 11)  # 10 days
    folds = walk_forward_splits(start, end, n_splits=1, train_ratio=0.7)
    assert len(folds) == 1
    f = folds[0]
    assert f.train_start == start
    assert f.train_end == _dt(1, 8)   # 7 days in
    assert f.test_start == _dt(1, 8)
    assert f.test_end == end


def test_two_splits_are_contiguous_and_cover_the_range():
    start, end = _dt(1, 1), _dt(1, 11)  # 10 days -> two 5-day windows
    folds = walk_forward_splits(start, end, n_splits=2, train_ratio=0.6)
    assert len(folds) == 2
    # window 1: [01-01, 01-06); train 3 days, test 2 days
    assert folds[0].train_start == _dt(1, 1)
    assert folds[0].train_end == _dt(1, 4)
    assert folds[0].test_end == _dt(1, 6)
    # window 2 starts where window 1 ended
    assert folds[1].train_start == _dt(1, 6)
    assert folds[1].test_end == end


def test_returns_fold_dataclass():
    folds = walk_forward_splits(_dt(1, 1), _dt(1, 11), n_splits=1, train_ratio=0.5)
    assert isinstance(folds[0], Fold)


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        walk_forward_splits(_dt(1, 1), _dt(1, 11), n_splits=0, train_ratio=0.7)
    with pytest.raises(ValueError):
        walk_forward_splits(_dt(1, 1), _dt(1, 11), n_splits=1, train_ratio=1.0)
    with pytest.raises(ValueError):
        walk_forward_splits(_dt(1, 11), _dt(1, 1), n_splits=1, train_ratio=0.7)  # end <= start
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/research/test_walkforward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.research'`

- [ ] **Step 3: Create `src/fxtrader/research/__init__.py`**

```python
"""Quant research: parameter optimization and walk-forward analysis."""
```

- [ ] **Step 4: Write `src/fxtrader/research/walkforward.py`**

```python
"""Walk-forward train/test splitting for out-of-sample strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Fold:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


def walk_forward_splits(
    start: datetime,
    end: datetime,
    n_splits: int,
    train_ratio: float = 0.7,
) -> list[Fold]:
    """Split [start, end) into n_splits contiguous windows, each train_ratio train / rest test."""
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if end <= start:
        raise ValueError("end must be after start")

    total = (end - start) / n_splits
    folds: list[Fold] = []
    for i in range(n_splits):
        win_start = start + total * i
        win_end = start + total * (i + 1)
        split = win_start + (win_end - win_start) * train_ratio
        folds.append(Fold(win_start, split, split, win_end))
    return folds
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/research/test_walkforward.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Full suite + lint + Commit**

Run: `.venv/bin/pytest -q` → all pass.
Run: `.venv/bin/ruff check src/fxtrader/research tests/research` → clean.
```bash
git add src/fxtrader/research/__init__.py src/fxtrader/research/walkforward.py tests/research/test_walkforward.py
git commit -m "feat: walk-forward train/test splitter"
```

---

## Task 2: grid_search

**Files:**
- Create: `src/fxtrader/research/optimize.py`
- Create: `tests/research/test_optimize.py`

`grid_search` sweeps the Cartesian product of a param grid, calls an injected `run_fn(params, start, end)` for each combo (skipping combos where `run_fn` returns `None`, e.g. invalid params), and returns the results ranked by an objective metric.

- [ ] **Step 1: Write the failing test `tests/research/test_optimize.py`**

```python
from datetime import datetime, timezone

from fxtrader.research.optimize import grid_search


S, E = datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 2, tzinfo=timezone.utc)


def test_grid_search_ranks_by_objective_descending():
    # fake run_fn: sharpe = fast (higher fast = better), ignore slow
    def run_fn(params, start, end):
        return {"sharpe": float(params["fast"]), "num_trades": 5}

    ranked = grid_search({"fast": [5, 20, 10], "slow": [50]}, run_fn, S, E, objective="sharpe")
    assert [r["params"]["fast"] for r in ranked] == [20, 10, 5]
    assert ranked[0]["metrics"]["sharpe"] == 20.0


def test_grid_search_sweeps_full_product():
    seen = []

    def run_fn(params, start, end):
        seen.append((params["fast"], params["slow"]))
        return {"sharpe": 1.0}

    grid_search({"fast": [5, 10], "slow": [30, 50]}, run_fn, S, E, objective="sharpe")
    assert sorted(seen) == [(5, 30), (5, 50), (10, 30), (10, 50)]


def test_grid_search_skips_none_results():
    # run_fn returns None for fast >= slow (invalid)
    def run_fn(params, start, end):
        if params["fast"] >= params["slow"]:
            return None
        return {"sharpe": 1.0}

    ranked = grid_search({"fast": [10, 60], "slow": [50]}, run_fn, S, E, objective="sharpe")
    assert len(ranked) == 1
    assert ranked[0]["params"] == {"fast": 10, "slow": 50}


def test_grid_search_minimize():
    def run_fn(params, start, end):
        return {"max_drawdown": -float(params["fast"]) / 100}

    # maximize=False -> least-negative drawdown first
    ranked = grid_search({"fast": [5, 20]}, run_fn, S, E, objective="max_drawdown", maximize=False)
    assert ranked[0]["params"]["fast"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/research/test_optimize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.research.optimize'`

- [ ] **Step 3: Write `src/fxtrader/research/optimize.py`**

```python
"""Parameter optimization: grid search + walk-forward, over an injected run function.

run_fn(params, start, end) -> metrics dict (or None to skip). Injecting it keeps this
module pure and free of any cli/engine import cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from itertools import product

RunFn = Callable[[dict, datetime, datetime], "dict | None"]


def grid_search(
    param_grid: dict[str, list],
    run_fn: RunFn,
    start: datetime,
    end: datetime,
    *,
    objective: str,
    maximize: bool = True,
) -> list[dict]:
    """Sweep the param grid; return [{"params":..., "metrics":...}] ranked by objective."""
    keys = list(param_grid.keys())
    results: list[dict] = []
    for combo in product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        metrics = run_fn(params, start, end)
        if metrics is not None:
            results.append({"params": params, "metrics": metrics})
    results.sort(key=lambda r: r["metrics"][objective], reverse=maximize)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/research/test_optimize.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Full suite + lint + Commit**

Run: `.venv/bin/pytest -q` → all pass.
Run: `.venv/bin/ruff check src/fxtrader/research tests/research` → clean.
```bash
git add src/fxtrader/research/optimize.py tests/research/test_optimize.py
git commit -m "feat: grid_search (ranked param sweep over an injected run function)"
```

---

## Task 3: walk_forward_optimize

**Files:**
- Modify: `src/fxtrader/research/optimize.py` (add `walk_forward_optimize`)
- Modify: `tests/research/test_optimize.py` (add WFO tests)

For each fold: grid-search the best params on the TRAIN window, then evaluate those params on the (unseen) TEST window. Aggregate the out-of-sample objective across folds — the real signal of whether a strategy generalizes.

- [ ] **Step 1: Append failing tests to `tests/research/test_optimize.py`**

```python
from datetime import timedelta

from fxtrader.research.optimize import walk_forward_optimize


def test_walk_forward_optimize_structure_and_oos():
    # run_fn: train picks the best fast (highest sharpe = fast); test sharpe is fixed per params
    def run_fn(params, start, end):
        return {"sharpe": float(params["fast"]), "num_trades": 3}

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=10)
    out = walk_forward_optimize(
        run_fn, start, end, {"fast": [5, 20], "slow": [50]},
        n_splits=2, train_ratio=0.7, objective="sharpe",
    )

    assert out["objective"] == "sharpe"
    assert out["param_grid"] == {"fast": [5, 20], "slow": [50]}
    assert len(out["folds"]) == 2
    fold0 = out["folds"][0]
    # best train params (fast=20 wins), and a test_metrics block present
    assert fold0["best_params"] == {"fast": 20, "slow": 50}
    assert fold0["test_metrics"]["sharpe"] == 20.0
    # mean out-of-sample objective across folds
    assert out["mean_oos_objective"] == 20.0


def test_walk_forward_optimize_skips_folds_with_no_valid_params():
    # every combo invalid -> no ranked params -> fold skipped
    def run_fn(params, start, end):
        return None

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=10)
    out = walk_forward_optimize(
        run_fn, start, end, {"fast": [10], "slow": [50]},
        n_splits=2, train_ratio=0.7, objective="sharpe",
    )
    assert out["folds"] == []
    assert out["mean_oos_objective"] == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/research/test_optimize.py -v`
Expected: FAIL with `ImportError: cannot import name 'walk_forward_optimize'`

- [ ] **Step 3: Append to `src/fxtrader/research/optimize.py`**

Add to the imports at the top (merge):
```python
from fxtrader.research.walkforward import walk_forward_splits
```

Append at the end of the file:
```python
def walk_forward_optimize(
    run_fn: RunFn,
    start: datetime,
    end: datetime,
    param_grid: dict[str, list],
    *,
    n_splits: int,
    train_ratio: float = 0.7,
    objective: str = "sharpe",
    maximize: bool = True,
) -> dict:
    """Optimize params per fold on train, evaluate out-of-sample on test, aggregate."""
    folds = walk_forward_splits(start, end, n_splits, train_ratio)
    fold_results: list[dict] = []
    for fold in folds:
        ranked = grid_search(
            param_grid, run_fn, fold.train_start, fold.train_end,
            objective=objective, maximize=maximize,
        )
        if not ranked:
            continue
        best = ranked[0]
        test_metrics = run_fn(best["params"], fold.test_start, fold.test_end)
        fold_results.append({
            "train_start": fold.train_start.isoformat(),
            "train_end": fold.train_end.isoformat(),
            "test_start": fold.test_start.isoformat(),
            "test_end": fold.test_end.isoformat(),
            "best_params": best["params"],
            "train_metrics": best["metrics"],
            "test_metrics": test_metrics,
        })

    oos = [
        fr["test_metrics"][objective]
        for fr in fold_results
        if fr["test_metrics"] is not None and objective in fr["test_metrics"]
    ]
    return {
        "objective": objective,
        "param_grid": param_grid,
        "n_splits": n_splits,
        "train_ratio": train_ratio,
        "folds": fold_results,
        "mean_oos_objective": (sum(oos) / len(oos)) if oos else 0.0,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/research/test_optimize.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Full suite + lint + Commit**

Run: `.venv/bin/pytest -q` → all pass.
Run: `.venv/bin/ruff check src/fxtrader/research tests/research` → clean.
```bash
git add src/fxtrader/research/optimize.py tests/research/test_optimize.py
git commit -m "feat: walk_forward_optimize (per-fold train optimize + OOS test)"
```

---

## Task 4: `fxtrader optimize` CLI

**Files:**
- Modify: `src/fxtrader/cli.py`
- Create: `tests/test_cli_optimize.py`

`run_optimize` builds the real `run_fn` from `run_backtest` (MA fast/slow grid; returns `None` on `fast>=slow`), runs `walk_forward_optimize`, and returns the result dict. The `optimize` command prints a per-fold summary and writes strict JSON.

- [ ] **Step 1: Write the failing test `tests/test_cli_optimize.py`**

```python
import json
from datetime import datetime, timedelta, timezone

from fxtrader.cli import run_optimize
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore


def _seed(tmp_path):
    store = TickStore(tmp_path)
    start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    ticks = []
    # ~3 days of 30s ticks with a couple of dips so crosses fire across bars
    for i in range(8000):
        base = 1.10 + 0.01 * ((i // 400) % 3 - 1)  # oscillates to create crossovers
        ticks.append(Tick("EURUSD", start + timedelta(seconds=i * 30), base, base + 0.00002))
    store.write_ticks("EURUSD", ticks)


def test_run_optimize_returns_walk_forward_structure(tmp_path):
    _seed(tmp_path)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        data_dir=tmp_path,
        fast_values=[2, 3],
        slow_values=[5, 8],
        n_splits=2,
        bar_minutes=5,
        objective="sharpe",
    )
    assert out["objective"] == "sharpe"
    assert out["param_grid"] == {"fast": [2, 3], "slow": [5, 8]}
    assert "mean_oos_objective" in out
    # strictly JSON-serializable (no inf/nan tokens)
    text = json.dumps(out, allow_nan=False)
    assert "fast" in text


def test_run_optimize_invalid_combos_skipped(tmp_path):
    _seed(tmp_path)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end=datetime(2024, 1, 4, tzinfo=timezone.utc),
        data_dir=tmp_path,
        fast_values=[5],
        slow_values=[3],   # fast >= slow -> invalid -> no ranked params
        n_splits=1,
        bar_minutes=5,
        objective="sharpe",
    )
    # no valid params -> folds skipped
    assert out["folds"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_optimize.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_optimize'`

- [ ] **Step 3: Modify `src/fxtrader/cli.py`**

Add to imports at the top (merge):
```python
from fxtrader.metrics.report import _json_safe
from fxtrader.research.optimize import walk_forward_optimize
```

Add these helpers after `run_backtest_config`:
```python
def _ma_run_fn(symbol: str, data_dir: str | Path, bar_minutes: int, size: float):
    """Build a run_fn(params, start, end) -> metrics for the MA grid (None on invalid params)."""
    def run_fn(params: dict, start: datetime, end: datetime):
        if params["fast"] >= params["slow"]:
            return None
        _result, metrics = run_backtest(
            symbol=symbol, start=start, end=end, data_dir=data_dir,
            fast=params["fast"], slow=params["slow"], size=size, bar_minutes=bar_minutes,
        )
        return metrics

    return run_fn


def run_optimize(
    symbol: str,
    start: datetime,
    end: datetime,
    data_dir: str | Path,
    *,
    fast_values: list[int],
    slow_values: list[int],
    n_splits: int = 3,
    train_ratio: float = 0.7,
    bar_minutes: int = 5,
    size: float = 10_000,
    objective: str = "sharpe",
) -> dict:
    """Walk-forward grid optimization of MA fast/slow over stored ticks."""
    run_fn = _ma_run_fn(symbol, data_dir, bar_minutes, size)
    return walk_forward_optimize(
        run_fn, start, end, {"fast": fast_values, "slow": slow_values},
        n_splits=n_splits, train_ratio=train_ratio, objective=objective,
    )
```

Add this command (after the `backtest` command):
```python
def _int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


@app.command()
def optimize(
    symbol: str = typer.Option(..., help="Single symbol, e.g. EURUSD"),
    start: str = typer.Option(..., help="Inclusive start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Exclusive end date YYYY-MM-DD"),
    data_dir: str = typer.Option("data", help="Parquet store root"),
    fast: str = typer.Option("5,10,20", help="Comma-separated fast MA windows"),
    slow: str = typer.Option("30,50,100", help="Comma-separated slow MA windows"),
    n_splits: int = typer.Option(3, help="Walk-forward folds"),
    train_ratio: float = typer.Option(0.7, help="Train fraction of each fold"),
    bar_minutes: int = typer.Option(5, help="Bar interval in minutes"),
    objective: str = typer.Option("sharpe", help="Metric to maximize on train"),
    json_out: str = typer.Option(None, "--json", help="Write optimize results to this path"),
) -> None:
    """Walk-forward optimize the MA-crossover params over stored ticks."""
    out = run_optimize(
        symbol=symbol, start=_parse_day(start), end=_parse_day(end), data_dir=data_dir,
        fast_values=_int_list(fast), slow_values=_int_list(slow),
        n_splits=n_splits, train_ratio=train_ratio, bar_minutes=bar_minutes, objective=objective,
    )
    typer.echo(f"symbol={symbol}  objective={objective}  folds={len(out['folds'])}")
    for i, fr in enumerate(out["folds"]):
        oos = fr["test_metrics"].get(objective) if fr["test_metrics"] else None
        typer.echo(f"  fold {i}: best={fr['best_params']}  OOS {objective}={oos:.2f}" if oos is not None
                   else f"  fold {i}: best={fr['best_params']}  OOS {objective}=n/a")
    typer.echo(f"  mean OOS {objective}: {out['mean_oos_objective']:.2f}")
    if json_out:
        Path(json_out).write_text(json.dumps(_json_safe(out), indent=2, allow_nan=False))
        typer.echo(f"wrote {json_out}")
```

Note: `json` is already imported at the top of cli.py? If not, add `import json` to the top imports. Check and add if missing.

- [ ] **Step 4: Run the CLI test + full suite + lint**

Run: `.venv/bin/pytest tests/test_cli_optimize.py -v` → 2 passed
Run: `.venv/bin/pytest -q` → all pass
Run: `.venv/bin/ruff check src/fxtrader` → clean

- [ ] **Step 5: Real-data run — produce an actual optimization**

Run: `mkdir -p out && .venv/bin/fxtrader optimize --symbol EURUSD --start 2024-01-02 --end 2024-04-01 --data-dir data --fast 5,10,20 --slow 30,50,100 --n-splits 3 --bar-minutes 15 --json out/opt_eurusd.json`
This walk-forward optimizes MA params over 3 months of real EUR/USD (may take 1-3 min — it runs many backtests). CAPTURE and REPORT the per-fold best params + OOS Sharpe + mean OOS, and confirm `out/opt_eurusd.json` is valid JSON. (Honest expectation: OOS Sharpe will likely be poor/negative — naive MA isn't alpha — that's the point of walk-forward: it exposes overfitting.)

- [ ] **Step 6: Commit**

```bash
git add src/fxtrader/cli.py tests/test_cli_optimize.py
git commit -m "feat: fxtrader optimize (walk-forward MA grid optimization)"
```

---

## Task 5: The quant-researcher subagent

**Files:**
- Create: `.claude/agents/quant-researcher.md`

A Claude Code subagent definition. It drives the platform's CLI to do research, with a human-in-the-loop gate before any live deployment.

- [ ] **Step 1: Create `.claude/agents/quant-researcher.md`**

```markdown
---
name: quant-researcher
description: FX quant research agent. Proposes strategy hypotheses, parametrizes/edits strategies, runs backtests and walk-forward optimizations via the fxtrader CLI, reads the strict-JSON results, iterates, and writes a findings report. Use when the user wants to research, optimize, or evaluate an FX trading strategy over the downloaded tick data. NEVER deploys to live trading — a human reviews and promotes.
tools: Bash, Read, Write, Edit, Grep, Glob
---

You are a disciplined quantitative researcher for the `fxtrader` platform. You form hypotheses, test them rigorously over historical tick data, and report honest findings. You are skeptical of your own results and you assume overfitting until walk-forward evidence says otherwise.

## What you have

- **Data:** ~3 years of Dukascopy ticks under `data/` for EURUSD, GBPUSD, USDJPY (2023-06 → 2026-05), already validated.
- **CLI (use these; do not re-implement them):**
  - `fxtrader backtest --symbol EURUSD --start YYYY-MM-DD --end YYYY-MM-DD --data-dir data --fast N --slow M --bar-minutes K --json out/run.json` — single backtest; writes a strict-JSON `results.json` (metrics, equity curve, trades) you parse.
  - `fxtrader optimize --symbol EURUSD --start ... --end ... --fast 5,10,20 --slow 30,50,100 --n-splits 3 --bar-minutes 15 --objective sharpe --json out/opt.json` — walk-forward grid optimization; writes per-fold train/test results.
  - Use `--daily-loss-limit` / `--max-position` on `backtest` to bound risk.
- **Strategies:** `src/fxtrader/strategy/examples/` (currently `MaCrossover`). To test a NEW strategy idea, write a new `Strategy` subclass following `MaCrossover` as the template, then wire a thin runner (ask the user / extend the CLI) — for the built-in MA strategy, just use the CLI flags.
- **Metrics meaning:** `sharpe`/`sortino` (annualized, 0.0 if the account is `ruined`), `max_drawdown` (floored at -100%), `profit_factor` (null if no losses), `win_rate`, `total_return`, `num_trades`, `ruined` (bool — treat a ruined run as a failure regardless of other metrics).

## Your research loop

1. **Hypothesis.** State a specific, falsifiable claim (e.g. "a slower MA pair (20/100) on 15m bars reduces overtrading and improves OOS Sharpe on EURUSD vs 10/30"). Write it down.
2. **In-sample probe.** Run a few `fxtrader backtest` runs to sanity-check direction. Read the JSON, not just the printed summary.
3. **Walk-forward.** Run `fxtrader optimize` to get OUT-OF-SAMPLE evidence. The number that matters is `mean_oos_objective`, not the best in-sample fold. A strategy that's great in-sample and poor OOS is overfit — say so.
4. **Cost honesty.** Always keep commission/slippage on. If a result only works at zero cost, it's not real. Watch `num_trades` — high turnover usually means costs dominate.
5. **Iterate** a bounded number of times (say 3-5 hypotheses), keeping a log of what you tried and what the OOS evidence showed.
6. **Report.** Write `docs/research/<date>-<topic>.md`: the hypotheses, the commands you ran, the OOS results table, and an honest verdict (most naive strategies do NOT beat costs — saying "no edge found" is a valid, valuable result).

## Hard rules

- **Never trade live and never claim a strategy is production-ready.** You produce research; a human reviews and decides. If asked to "deploy", stop and hand back to the human.
- **Out-of-sample or it didn't happen.** Never recommend params chosen by in-sample fit alone.
- **Report failures.** Negative results are results. Do not p-hack by cherry-picking windows or hiding losing runs.
- **Don't fabricate numbers.** Every metric you cite must come from a JSON file you actually produced; reference the file path.
- **Stay in scope.** Use the existing CLI and data. If you need a genuinely new strategy implemented, write the `Strategy` subclass and its test, but flag that the CLI currently only wires `MaCrossover` and ask before extending it.

When done, summarize: hypotheses tested, the best honest OOS result, and a clear recommendation (usually: "no robust edge in what was tested" or "X is worth a human's deeper look").
```

- [ ] **Step 2: Verify the agent file is well-formed**

Run: `.venv/bin/python -c "import pathlib,re; t=pathlib.Path('.claude/agents/quant-researcher.md').read_text(); assert t.startswith('---'); fm=t.split('---')[1]; assert 'name: quant-researcher' in fm and 'tools:' in fm; print('frontmatter OK; body', len(t), 'chars')"`
Expected: prints `frontmatter OK; body <n> chars`.

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/quant-researcher.md
git commit -m "feat: quant-researcher subagent (drives the optimizer, human-in-the-loop)"
```

---

## Self-review

**Spec coverage (P4 = "both" per the design §5.8):** programmatic optimizer — `grid_search` + `walk_forward_optimize` (Tasks 2-3) + `walk_forward_splits` (Task 1); CLI `fxtrader optimize` with strict-JSON output (Task 4); the LLM subagent `.claude/agents/quant-researcher.md` that drives the CLI, reads `results.json`, and keeps a human in the loop (Task 5). ✓ Deviation from spec (documented): grid/walk-forward search instead of Optuna — deterministic, dependency-free, sufficient for the discrete MA grid; Optuna is a drop-in future option (the injected `run_fn`/`grid_search` boundary makes swapping the search engine trivial).

**Placeholder scan:** none — all code/tests/agent content complete.

**Type/signature consistency:** `Fold(train_start, train_end, test_start, test_end)`; `walk_forward_splits(start, end, n_splits, train_ratio) -> list[Fold]`; `grid_search(param_grid, run_fn, start, end, *, objective, maximize) -> list[{"params","metrics"}]`; `walk_forward_optimize(run_fn, start, end, param_grid, *, n_splits, train_ratio, objective, maximize) -> dict`; `run_optimize(symbol, start, end, data_dir, *, fast_values, slow_values, n_splits, train_ratio, bar_minutes, size, objective) -> dict`. The optimizer never imports cli (run_fn injected) — no cycle. Consistent across tasks.

**Scope:** Delivers a working walk-forward optimizer + a research subagent that drives the whole platform. Optuna and genuinely-new strategies beyond MaCrossover are future work, flagged in the agent's rules.
