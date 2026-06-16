"""Backtest performance metrics derived from an equity curve and fills."""

from __future__ import annotations

import math
from datetime import datetime

from fxtrader.core.types import Fill, Side

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def trade_pnls(fills: list[Fill]) -> list[float]:
    """Realized price-PnL per position-reducing fill (one entry per closing event)."""
    qty: dict[str, float] = {}
    avg: dict[str, float] = {}
    out: list[float] = []
    for f in fills:
        signed = f.quantity if f.side is Side.BUY else -f.quantity
        q = qty.get(f.symbol, 0.0)
        a = avg.get(f.symbol, 0.0)
        if q != 0 and (q > 0) != (signed > 0):
            closing = min(abs(signed), abs(q))
            direction = 1.0 if q > 0 else -1.0
            out.append((f.price - a) * closing * direction)
        new_q = q + signed
        if new_q == 0:
            a = 0.0
        elif q == 0 or (q > 0) == (signed > 0):
            a = (a * abs(q) + f.price * abs(signed)) / abs(new_q)
        elif abs(signed) > abs(q):
            a = f.price
        qty[f.symbol] = new_q
        avg[f.symbol] = a
    return out


def _zeroed() -> dict:
    return {
        "total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "max_drawdown": 0.0, "num_trades": 0, "win_rate": 0.0,
        "profit_factor": 0.0, "final_equity": 0.0, "num_fills": 0,
    }


def compute_metrics(
    equity_curve: list[tuple[datetime, float]],
    fills: list[Fill],
    *,
    periods_per_year: float,
) -> dict:
    metrics = _zeroed()
    metrics["num_fills"] = len(fills)

    pnls = trade_pnls(fills)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    metrics["num_trades"] = len(pnls)
    metrics["win_rate"] = len(wins) / len(pnls) if pnls else 0.0
    if losses:
        metrics["profit_factor"] = sum(wins) / abs(sum(losses))
    elif wins:
        metrics["profit_factor"] = float("inf")

    if len(equity_curve) < 2:
        if equity_curve:
            metrics["final_equity"] = equity_curve[-1][1]
        return metrics

    eqs = [e for _, e in equity_curve]
    metrics["final_equity"] = eqs[-1]
    if eqs[0] > 0:
        metrics["total_return"] = eqs[-1] / eqs[0] - 1

    rets = [eqs[i] / eqs[i - 1] - 1 for i in range(1, len(eqs)) if eqs[i - 1] != 0]
    if rets:
        n = len(rets)
        mean = sum(rets) / n
        std = (sum((r - mean) ** 2 for r in rets) / n) ** 0.5
        if std > 0:
            metrics["sharpe"] = mean / std * math.sqrt(periods_per_year)
        downside = (sum(min(r, 0.0) ** 2 for r in rets) / n) ** 0.5
        if downside > 0:
            metrics["sortino"] = mean / downside * math.sqrt(periods_per_year)

    peak = eqs[0]
    max_dd = 0.0
    for e in eqs:
        peak = max(peak, e)
        if peak > 0:
            max_dd = min(max_dd, (e - peak) / peak)
    metrics["max_drawdown"] = max_dd

    span = (equity_curve[-1][0] - equity_curve[0][0]).total_seconds()
    years = max(span / _SECONDS_PER_YEAR, 1e-9)
    if eqs[0] > 0 and eqs[-1] > 0:
        metrics["cagr"] = (eqs[-1] / eqs[0]) ** (1 / years) - 1

    return metrics
