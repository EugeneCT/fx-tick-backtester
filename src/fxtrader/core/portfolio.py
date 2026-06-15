"""Position/cash bookkeeping via cash-flow accounting."""

from __future__ import annotations

from fxtrader.core.types import Fill, Position, Side


class Portfolio:
    def __init__(self, starting_cash: float) -> None:
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.realized_pnl = 0.0

    def position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol))

    def apply_fill(self, fill: Fill) -> None:
        signed = fill.quantity if fill.side is Side.BUY else -fill.quantity
        pos = self.position(fill.symbol)

        # Realize PnL on the portion that reduces an existing opposite position.
        if pos.quantity != 0 and (pos.quantity > 0) != (signed > 0):
            closing = min(abs(signed), abs(pos.quantity))
            direction = 1.0 if pos.quantity > 0 else -1.0
            self.realized_pnl += (fill.price - pos.avg_price) * closing * direction

        new_qty = pos.quantity + signed
        if new_qty == 0:
            pos.avg_price = 0.0
        elif pos.quantity == 0 or (pos.quantity > 0) == (signed > 0):
            # Opening or adding in the same direction -> weighted average.
            pos.avg_price = (
                pos.avg_price * abs(pos.quantity) + fill.price * abs(signed)
            ) / abs(new_qty)
        elif abs(signed) > abs(pos.quantity):
            # Flipped through zero -> remaining lot is at the fill price.
            pos.avg_price = fill.price
        # else: partial close, avg_price of the remaining lot is unchanged.

        pos.quantity = new_qty
        self.cash -= signed * fill.price
        self.cash -= fill.commission
        self.fills.append(fill)

    def equity(self, prices: dict[str, float]) -> float:
        mark = sum(p.quantity * prices[p.symbol] for p in self.positions.values() if p.quantity)
        return self.cash + mark
