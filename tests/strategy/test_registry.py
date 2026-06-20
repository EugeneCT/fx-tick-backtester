import pytest

from fxtrader.strategy.examples.ma_crossover import MaCrossover
from fxtrader.strategy.examples.trend_filtered_crossover import TrendFilteredCrossover
from fxtrader.strategy.exits import AtrMultiple, FixedPips, NoExit
from fxtrader.strategy.registry import make_exit_policy, make_strategy


def test_make_strategy_ma():
    s = make_strategy("ma", symbol="EURUSD", params={"fast": 10, "slow": 30, "size": 10_000})
    assert isinstance(s, MaCrossover)


def test_make_strategy_trend_with_exit():
    s = make_strategy(
        "trend", symbol="EURUSD",
        params={"fast": 10, "slow": 30, "trend": 100, "size": 10_000},
        exit_policy=FixedPips(10, 20, 0.0001),
    )
    assert isinstance(s, TrendFilteredCrossover)


def test_make_strategy_unknown_raises():
    with pytest.raises(ValueError):
        make_strategy("nope", symbol="EURUSD", params={})


def test_make_exit_policy_variants():
    assert isinstance(make_exit_policy("none", params={}, symbol="EURUSD"), NoExit)
    assert isinstance(make_exit_policy(
        "fixed", params={"stop_pips": 10, "tp_pips": 20}, symbol="EURUSD"), FixedPips)
    assert isinstance(make_exit_policy(
        "atr", params={"atr_period": 14, "stop_atr": 2.0, "tp_atr": 3.0}, symbol="EURUSD"),
        AtrMultiple)
    with pytest.raises(ValueError):
        make_exit_policy("bad", params={}, symbol="EURUSD")
