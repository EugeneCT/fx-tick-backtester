"""Factory mapping CLI strategy/exit names to objects. Keeps cli.py generic."""

from __future__ import annotations

from fxtrader.core.instruments import pip_size
from fxtrader.strategy.base import Strategy
from fxtrader.strategy.examples.ma_crossover import MaCrossover
from fxtrader.strategy.examples.trend_filtered_crossover import TrendFilteredCrossover
from fxtrader.strategy.exits import AtrMultiple, ExitPolicy, FixedPips, NoExit

STRATEGY_PARAMS: dict[str, tuple[str, ...]] = {
    "ma": ("fast", "slow"),
    "trend": ("fast", "slow", "trend"),
}


def make_exit_policy(kind: str, *, params: dict, symbol: str) -> ExitPolicy:
    if kind == "none":
        return NoExit()
    if kind == "fixed":
        return FixedPips(params["stop_pips"], params["tp_pips"], pip_size(symbol))
    if kind == "atr":
        return AtrMultiple(params["atr_period"], params["stop_atr"], params["tp_atr"])
    raise ValueError(f"unknown exit kind {kind!r}")


def make_strategy(name: str, *, symbol: str, params: dict, exit_policy: ExitPolicy | None = None) -> Strategy:
    if name == "ma":
        return MaCrossover(symbol=symbol, fast=params["fast"], slow=params["slow"], size=params["size"])
    if name == "trend":
        return TrendFilteredCrossover(
            symbol=symbol, fast=params["fast"], slow=params["slow"], trend=params["trend"],
            size=params["size"], exit_policy=exit_policy,
        )
    raise ValueError(f"unknown strategy {name!r}")
