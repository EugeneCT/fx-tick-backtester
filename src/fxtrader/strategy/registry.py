"""Factory mapping CLI strategy/exit names to objects. Keeps cli.py generic."""

from __future__ import annotations

from fxtrader.core.instruments import pip_size
from fxtrader.strategy.base import Strategy
from fxtrader.strategy.examples.ma_crossover import MaCrossover
from fxtrader.strategy.examples.mean_reversion_fader import MeanReversionFader
from fxtrader.strategy.examples.opening_range_breakout import OpeningRangeBreakout
from fxtrader.strategy.examples.pairs_spread_reversion import PairsSpreadReversion
from fxtrader.strategy.examples.price_action import PriceActionTrend
from fxtrader.strategy.examples.trend_filtered_crossover import TrendFilteredCrossover
from fxtrader.strategy.exits import AtrMultiple, ExitPolicy, FixedPips, NoExit

STRATEGY_PARAMS: dict[str, tuple[str, ...]] = {
    "ma": ("fast", "slow"),
    "trend": ("fast", "slow", "trend"),
    "meanrev": ("lookback", "entry_z", "exit_z", "er_threshold"),
    "orb": ("range_start_hour", "range_end_hour", "entry_end_hour", "exit_hour"),
    "xpair": ("lookback", "entry_z", "exit_z", "stop_z"),
    "pa": ("pattern", "trend", "wick_ratio", "body_frac",
           "swing_lookback", "inside_break_bars", "max_hold_bars"),
}


def make_exit_policy(kind: str, *, params: dict, symbol: str) -> ExitPolicy:
    if kind == "none":
        return NoExit()
    if kind == "fixed":
        return FixedPips(params["stop_pips"], params["tp_pips"], pip_size(symbol))
    if kind == "atr":
        return AtrMultiple(params["atr_period"], params["stop_atr"], params["tp_atr"])
    raise ValueError(f"unknown exit kind {kind!r}")


def make_strategy(name: str, *, symbol: str, params: dict,
                  exit_policy: ExitPolicy | None = None, symbol2: str | None = None) -> Strategy:
    if name == "ma":
        return MaCrossover(symbol=symbol, fast=params["fast"], slow=params["slow"], size=params["size"])
    if name == "trend":
        return TrendFilteredCrossover(
            symbol=symbol, fast=params["fast"], slow=params["slow"], trend=params["trend"],
            size=params["size"], exit_policy=exit_policy,
        )
    if name == "meanrev":
        return MeanReversionFader(
            symbol=symbol, lookback=params["lookback"], entry_z=params["entry_z"],
            exit_z=params["exit_z"], er_threshold=params["er_threshold"],
            size=params["size"], exit_policy=exit_policy,
        )
    if name == "orb":
        return OpeningRangeBreakout(
            symbol=symbol, range_start_hour=params["range_start_hour"],
            range_end_hour=params["range_end_hour"], entry_end_hour=params["entry_end_hour"],
            exit_hour=params["exit_hour"], size=params["size"], exit_policy=exit_policy,
        )
    if name == "xpair":
        return PairsSpreadReversion(
            primary=symbol, secondary=symbol2, lookback=params["lookback"],
            entry_z=params["entry_z"], exit_z=params["exit_z"], stop_z=params["stop_z"],
            size=params["size"],
        )
    if name == "pa":
        return PriceActionTrend(
            symbol=symbol, pattern=params["pattern"], trend=params["trend"], size=params["size"],
            wick_ratio=params.get("wick_ratio", 2.0), body_frac=params.get("body_frac", 0.34),
            swing_lookback=params.get("swing_lookback", 20),
            inside_break_bars=params.get("inside_break_bars", 3),
            max_hold_bars=params.get("max_hold_bars", 0),
            exit_policy=exit_policy,
        )
    raise ValueError(f"unknown strategy {name!r}")
