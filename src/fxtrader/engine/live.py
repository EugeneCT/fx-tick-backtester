"""LiveEngine — real-time, callback-driven execution mirroring the backtest loop.

Same Strategy / RiskManager as the backtester. Mode-gated order submission
(DRY_RUN sends nothing). Kill-switch flattens via the broker. State persists to disk.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from fxtrader.broker.base import Broker
from fxtrader.core.clock import WallClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Order, Side, Tick
from fxtrader.risk.manager import RiskManager
from fxtrader.strategy.base import Strategy


class TradingMode(str, Enum):
    DRY_RUN = "DRY_RUN"  # compute orders but send nothing to the broker
    PAPER = "PAPER"      # paper account
    LIVE = "LIVE"        # real money


class LiveEngine:
    def __init__(
        self,
        broker: Broker,
        strategy: Strategy,
        portfolio: Portfolio,
        risk: RiskManager,
        *,
        mode: TradingMode = TradingMode.DRY_RUN,
        state_path: str | Path | None = None,
    ) -> None:
        self.broker = broker
        self.strategy = strategy
        self.portfolio = portfolio
        self.risk = risk
        self.mode = mode
        self.state_path = Path(state_path) if state_path else None
        self.clock = WallClock()
        self._last_price: dict[str, float] = {}
        self._flattened = False

    def start(self) -> None:
        self.broker.set_fill_callback(self._on_fill)
        self.broker.set_tick_callback(self._on_tick)
        self.strategy.bind(self.clock, self.portfolio, self._on_order)
        # Anchor the daily-loss baseline to the session's opening equity *before*
        # any fills, so a same-tick loss is measured against the true start equity
        # (live sees one update per tick; a backtest gets many ticks pre-loss).
        self.risk.update_equity(self.portfolio.equity(self._last_price), self.clock)
        self.broker.connect()
        self.strategy.on_start()

    def stop(self) -> None:
        self.strategy.on_stop()
        self._persist_state()
        self.broker.disconnect()

    def _on_order(self, order: Order) -> None:
        if self.risk.halted:
            return
        if order.symbol not in self._last_price:
            return
        if not self.risk.check(order, self.portfolio, self._last_price, self.clock):
            return
        if self.mode is TradingMode.DRY_RUN:
            return  # computed but not sent
        self.broker.submit(order)

    def _on_fill(self, fill) -> None:
        self.portfolio.apply_fill(fill)
        self.strategy.on_fill(fill)

    def _on_tick(self, tick: Tick) -> None:
        self._last_price[tick.symbol] = tick.mid
        self.broker.update_market(tick)

        equity = self.portfolio.equity(self._last_price)
        self.risk.update_equity(equity, self.clock)
        if self.risk.halted:
            if not self._flattened:
                self._flatten()
        else:
            self._flattened = False
            self.strategy.on_tick(tick)

        self._persist_state()

    def _flatten(self) -> None:
        for symbol, pos in list(self.portfolio.positions.items()):
            if pos.quantity != 0 and symbol in self._last_price:
                side = Side.SELL if pos.quantity > 0 else Side.BUY
                if self.mode is not TradingMode.DRY_RUN:
                    self.broker.submit(Order(symbol, side, abs(pos.quantity)))
        self._flattened = True

    def _persist_state(self) -> None:
        if self.state_path is None:
            return
        state = {
            "mode": self.mode.value,
            "ts": self.clock.now().isoformat(),
            "equity": self.portfolio.equity(self._last_price),
            "cash": self.portfolio.cash,
            "halted": self.risk.halted,
            "positions": {
                s: p.quantity for s, p in self.portfolio.positions.items() if p.quantity
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.state_path)
