from datetime import date

import pytest

from fxtrader.config import BacktestConfig, load_config


def test_minimal_config_has_defaults():
    cfg = BacktestConfig(symbol="EURUSD", start="2024-01-02", end="2024-02-01")
    assert cfg.symbol == "EURUSD"
    assert cfg.start == date(2024, 1, 2)
    assert cfg.fast == 20 and cfg.slow == 50
    assert cfg.bar_minutes == 1
    assert cfg.daily_loss_limit is None  # disabled by default


def test_to_run_kwargs_maps_dates_and_unbounded_limits():
    cfg = BacktestConfig(symbol="EURUSD", start="2024-01-02", end="2024-02-01",
                         fast=10, slow=30, daily_loss_limit=None, max_position=None)
    kw = cfg.to_run_kwargs()
    from datetime import datetime, timezone
    assert kw["symbol"] == "EURUSD"
    assert kw["start"] == datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert kw["end"] == datetime(2024, 2, 1, tzinfo=timezone.utc)
    assert kw["fast"] == 10 and kw["slow"] == 30
    assert kw["daily_loss_limit"] == float("inf")  # None -> inf
    assert kw["max_position"] == float("inf")


def test_explicit_limits_pass_through():
    cfg = BacktestConfig(symbol="EURUSD", start="2024-01-02", end="2024-02-01",
                         daily_loss_limit=5_000, max_position=50_000)
    kw = cfg.to_run_kwargs()
    assert kw["daily_loss_limit"] == 5_000
    assert kw["max_position"] == 50_000


def test_rejects_unknown_field():
    with pytest.raises(Exception):
        BacktestConfig(symbol="EURUSD", start="2024-01-02", end="2024-02-01", bogus=1)


def test_load_config_from_yaml(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text(
        "symbol: GBPUSD\nstart: 2023-06-01\nend: 2024-06-01\n"
        "fast: 15\nslow: 40\nbar_minutes: 5\nsize: 25000\n"
    )
    cfg = load_config(p)
    assert cfg.symbol == "GBPUSD"
    assert cfg.fast == 15 and cfg.slow == 40
    assert cfg.bar_minutes == 5
    assert cfg.size == 25_000
