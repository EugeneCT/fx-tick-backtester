"""Backtest reporting: machine-readable results.json (and HTML, see write_html_report)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from fxtrader.config import BacktestConfig
from fxtrader.engine.backtest import BacktestResult
from fxtrader.metrics.performance import trade_pnls


def _json_safe(value):
    """Recursively replace non-finite floats (inf/-inf/nan) with None.

    metrics can legitimately contain inf (e.g. profit_factor on an all-wins run),
    but json.dumps emits the non-standard `Infinity` token that strict parsers
    (JS/Go/Rust/jq) reject. results.json is the P4 agent's contract, so it must be
    spec-valid JSON: non-finite -> null.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def build_results(config: BacktestConfig, result: BacktestResult, metrics: dict) -> dict:
    """A deterministic, strictly-JSON-serializable summary of a backtest run."""
    return _json_safe(
        {
            "config": config.model_dump(mode="json"),
            "metrics": metrics,
            "equity_curve": [[ts.isoformat(), equity] for ts, equity in result.equity_curve],
            "trades": trade_pnls(result.fills),
            "num_fills": len(result.fills),
        }
    )


def write_results_json(
    path: str | Path,
    config: BacktestConfig,
    result: BacktestResult,
    metrics: dict,
) -> None:
    # allow_nan=False guarantees strict JSON (raises if any non-finite slips past
    # _json_safe) rather than silently emitting `Infinity`/`NaN`.
    payload = json.dumps(build_results(config, result, metrics), indent=2, allow_nan=False)
    Path(path).write_text(payload)
