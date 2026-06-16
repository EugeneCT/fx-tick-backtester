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
    # Rank by the raw objective value: maximize -> highest first, minimize -> lowest first.
    # (For a "less is better" but negative-valued metric like max_drawdown, the least-bad value
    # is the largest, so you maximize it — sign-aware, no abs() hack.)
    results.sort(key=lambda r: r["metrics"][objective], reverse=maximize)
    return results
