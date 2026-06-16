"""Parameter optimization: grid search + walk-forward, over an injected run function.

run_fn(params, start, end) -> metrics dict (or None to skip). Injecting it keeps this
module pure and free of any cli/engine import cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from itertools import product

from fxtrader.research.walkforward import walk_forward_splits

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
    # Rank by the raw objective value: maximize -> highest first, minimize -> lowest first.
    # (For a "less is better" but negative-valued metric like max_drawdown, the least-bad value
    # is the largest, so you maximize it — sign-aware, no abs() hack.)
    results.sort(key=lambda r: r["metrics"][objective], reverse=maximize)
    return results


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
