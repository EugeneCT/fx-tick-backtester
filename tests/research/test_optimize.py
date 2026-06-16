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
