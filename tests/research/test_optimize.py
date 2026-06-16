from datetime import datetime, timedelta, timezone

from fxtrader.research.optimize import grid_search, walk_forward_optimize


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
    # Minimize a positive "less is better" metric (e.g. turnover): smallest raw value first.
    def run_fn(params, start, end):
        return {"turnover": float(params["fast"])}

    ranked = grid_search({"fast": [5, 20]}, run_fn, S, E, objective="turnover", maximize=False)
    assert ranked[0]["params"]["fast"] == 5
    assert ranked[-1]["params"]["fast"] == 20


def test_grid_search_least_bad_drawdown_via_maximize():
    # max_drawdown is <= 0; "least bad" = closest to zero = the MAX value -> maximize=True.
    def run_fn(params, start, end):
        return {"max_drawdown": -float(params["fast"]) / 100}  # fast=5 -> -0.05, fast=20 -> -0.20

    ranked = grid_search({"fast": [5, 20]}, run_fn, S, E, objective="max_drawdown", maximize=True)
    assert ranked[0]["params"]["fast"] == 5  # -0.05 is the least-bad drawdown


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
