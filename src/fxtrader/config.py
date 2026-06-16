"""Typed, YAML-loadable backtest run configuration."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class BacktestConfig(BaseModel):
    """A reproducible backtest specification. Maps to cli.run_backtest via to_run_kwargs()."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    start: date
    end: date
    fast: int = 20
    slow: int = 50
    size: float = 10_000
    bar_minutes: int = 1
    starting_cash: float = 100_000.0
    commission_rate: float = 0.00002
    min_commission: float = 2.0
    slippage: float = 0.0
    daily_loss_limit: float | None = None  # None -> unbounded
    max_position: float | None = None      # None -> unbounded

    def to_run_kwargs(self) -> dict:
        """Keyword args for cli.run_backtest (dates -> UTC datetimes; None -> inf)."""
        return {
            "symbol": self.symbol,
            "start": datetime(self.start.year, self.start.month, self.start.day, tzinfo=timezone.utc),
            "end": datetime(self.end.year, self.end.month, self.end.day, tzinfo=timezone.utc),
            "fast": self.fast,
            "slow": self.slow,
            "size": self.size,
            "bar_minutes": self.bar_minutes,
            "starting_cash": self.starting_cash,
            "commission_rate": self.commission_rate,
            "min_commission": self.min_commission,
            "slippage": self.slippage,
            "daily_loss_limit": self.daily_loss_limit if self.daily_loss_limit is not None else float("inf"),
            "max_position": self.max_position if self.max_position is not None else float("inf"),
        }


def load_config(path: str | Path) -> BacktestConfig:
    data = yaml.safe_load(Path(path).read_text())
    return BacktestConfig(**data)
