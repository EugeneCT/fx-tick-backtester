"""Backtest reporting: machine-readable results.json (and HTML, see write_html_report)."""

from __future__ import annotations

import json
from pathlib import Path

from fxtrader.config import BacktestConfig
from fxtrader.engine.backtest import BacktestResult
from fxtrader.metrics.performance import trade_pnls


def build_results(config: BacktestConfig, result: BacktestResult, metrics: dict) -> dict:
    """A deterministic, JSON-serializable summary of a backtest run."""
    return {
        "config": config.model_dump(mode="json"),
        "metrics": metrics,
        "equity_curve": [[ts.isoformat(), equity] for ts, equity in result.equity_curve],
        "trades": trade_pnls(result.fills),
        "num_fills": len(result.fills),
    }


def write_results_json(
    path: str | Path,
    config: BacktestConfig,
    result: BacktestResult,
    metrics: dict,
) -> None:
    Path(path).write_text(json.dumps(build_results(config, result, metrics), indent=2))
