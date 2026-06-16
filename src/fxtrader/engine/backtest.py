"""Event-driven backtest engine: ticks -> strategy -> risk -> broker -> portfolio."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from fxtrader.broker.base import Broker
from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Order, Side, Tick
from fxtrader.data.feed import Resampler
from fxtrader.risk.manager import RiskManager
from fxtrader.strategy.base import Strategy


@dataclass
class BacktestResult:
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    fills: list = field(default_factory=list)
    final_equity: float = 0.0
    portfolio: Portfolio | None = None


class BacktestEngine:
    def __init__(
        self,
        feed: Iterable[Tick],
        strategy: Strategy,
        broker: Broker,
        portfolio: Portfolio,
        risk: RiskManager,
        *,
        equity_interval: timedelta = timedelta(hours=1),
        bar_interval: timedelta | None = None,
    ) -> None:
        self.feed = feed
        self.strategy = strategy
        self.broker = broker
        self.portfolio = portfolio
        self.risk = risk
        self.equity_interval = equity_interval
        self.bar_interval = bar_interval
        self.clock = SimClock()
        self.equity_curve: list[tuple[datetime, float]] = []
        self._last_price: dict[str, float] = {}
        self._resamplers: dict[str, Resampler] = {}
        self._flattened = False
        self._eq_bucket: datetime | None = None

    def _resampler(self, symbol: str) -> Resampler:
        rs = self._resamplers.get(symbol)
        if rs is None:
            rs = Resampler(self.bar_interval)
            self._resamplers[symbol] = rs
        return rs

    def _on_order(self, order: Order) -> None:
        if self.risk.halted:
            return
        if order.symbol not in self._last_price:
            return
        if self.risk.check(order, self.portfolio, self._last_price, self.clock):
            self.broker.submit(order)

    def _on_fill(self, fill) -> None:
        self.portfolio.apply_fill(fill)
        self.strategy.on_fill(fill)

    def _flatten(self) -> None:
        for symbol, pos in list(self.portfolio.positions.items()):
            if pos.quantity != 0 and symbol in self._last_price:
                side = Side.SELL if pos.quantity > 0 else Side.BUY
                self.broker.submit(Order(symbol, side, abs(pos.quantity)))
        self._flattened = True

    def _floor(self, ts: datetime) -> datetime:
        step = self.equity_interval.total_seconds()
        epoch = ts.timestamp()
        return datetime.fromtimestamp(epoch - (epoch % step), tz=timezone.utc)

    def run(self) -> BacktestResult:
        self.strategy.bind(self.clock, self.portfolio, self._on_order)
        self.broker.set_fill_callback(self._on_fill)
        self.strategy.on_start()

        for tick in self.feed:
            self.clock.set_time(tick.ts)
            self._last_price[tick.symbol] = tick.mid
            self.broker.update_market(tick)

            equity = self.portfolio.equity(self._last_price)
            self.risk.update_equity(equity, self.clock)
            if self.risk.halted:
                if not self._flattened:
                    self._flatten()
            else:
                # Not halted (incl. after the daily kill switch re-arms next session):
                # clear the flatten latch so a future breach can flatten again.
                self._flattened = False
                self.strategy.on_tick(tick)
                if self.bar_interval is not None:
                    bar = self._resampler(tick.symbol).update(tick)
                    if bar is not None:
                        self.strategy.on_bar(bar)

            bucket = self._floor(tick.ts)
            if bucket != self._eq_bucket:
                self.equity_curve.append((bucket, equity))
                self._eq_bucket = bucket

        if self.bar_interval is not None and not self.risk.halted:
            for rs in self._resamplers.values():
                bar = rs.flush()
                if bar is not None:
                    self.strategy.on_bar(bar)

        self.strategy.on_stop()
        final_equity = self.portfolio.equity(self._last_price)
        self.equity_curve.append((self.clock.now(), final_equity))
        return BacktestResult(
            equity_curve=self.equity_curve,
            fills=list(self.portfolio.fills),
            final_equity=final_equity,
            portfolio=self.portfolio,
        )
