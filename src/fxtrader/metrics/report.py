"""Backtest reporting: machine-readable results.json (and HTML, see write_html_report)."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from fxtrader.config import BacktestConfig  # noqa: E402
from fxtrader.engine.backtest import BacktestResult  # noqa: E402
from fxtrader.metrics.performance import trade_pnls  # noqa: E402


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


def _equity_png_base64(result: BacktestResult) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    if result.equity_curve:
        xs = [ts for ts, _ in result.equity_curve]
        ys = [equity for _, equity in result.equity_curve]
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.0)
    ax.set_title("Equity curve")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def write_html_report(
    path: str | Path,
    config: BacktestConfig,
    result: BacktestResult,
    metrics: dict,
) -> None:
    png = _equity_png_base64(result)
    rows = "".join(
        f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        for key, value in metrics.items()
    )
    html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Backtest — {config.symbol}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; }} td {{ border: 1px solid #ddd; padding: 4px 10px; }}
td:first-child {{ color: #555; }}
</style></head>
<body>
<h1>Backtest — {config.symbol}</h1>
<p>MA({config.fast}/{config.slow}) · {config.bar_minutes}m bars · {config.start} → {config.end}</p>
<img src="data:image/png;base64,{png}" alt="Equity curve">
<h2>Metrics</h2>
<table>{rows}</table>
</body></html>
"""
    Path(path).write_text(html)
