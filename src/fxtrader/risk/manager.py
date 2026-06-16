"""RiskManager — order vetting + daily-loss kill switch, identical in backtest and live."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from fxtrader.core.clock import Clock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Order, Side


@dataclass
class RiskLimits:
    max_position: float = float("inf")          # abs base units per symbol
    max_gross_notional: float = float("inf")    # sum over symbols of |qty| * price
    daily_loss_limit: float = float("inf")      # positive; halt when day P&L <= -limit
    max_orders_per_minute: int = 1_000_000


class RiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self.halted = False
        self._order_times: deque = deque()
        self._day = None
        self._day_start_equity: float | None = None

    def update_equity(self, equity: float, clock: Clock) -> None:
        day = clock.now().date()
        if day != self._day:
            self._day = day
            self._day_start_equity = equity
        if self._day_start_equity is not None:
            if equity - self._day_start_equity <= -self.limits.daily_loss_limit:
                self.halted = True

    def check(self, order: Order, portfolio: Portfolio, prices: dict, clock: Clock) -> bool:
        if self.halted:
            return False

        now = clock.now()
        while self._order_times and (now - self._order_times[0]).total_seconds() > 60:
            self._order_times.popleft()
        if len(self._order_times) >= self.limits.max_orders_per_minute:
            return False

        signed = order.quantity if order.side is Side.BUY else -order.quantity

        projected = {s: p.quantity for s, p in portfolio.positions.items()}
        projected[order.symbol] = projected.get(order.symbol, 0.0) + signed

        if abs(projected[order.symbol]) > self.limits.max_position:
            return False

        gross = sum(abs(q) * prices[s] for s, q in projected.items() if s in prices)
        if gross > self.limits.max_gross_notional:
            return False

        self._order_times.append(now)
        return True
