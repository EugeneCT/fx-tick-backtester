"""Per-trade Maximum Adverse / Favorable Excursion from fills + bars. Pure post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from fxtrader.core.types import Bar, Fill, Side


@dataclass(frozen=True, slots=True)
class TradeRecord:
    entry_ts: datetime
    exit_ts: datetime
    direction: int   # +1 long, -1 short
    pnl: float
    mae_pips: float
    mfe_pips: float


def _round_trips(fills: list[Fill]):
    """Yield (entry_ts, exit_ts, direction, entry_price, pnl) per closed round trip, single-symbol."""
    qty = 0.0
    entry_price = 0.0
    entry_ts: datetime | None = None
    direction = 0
    for f in fills:
        signed = f.quantity if f.side is Side.BUY else -f.quantity
        if qty == 0:
            qty = signed
            entry_price = f.price
            entry_ts = f.ts
            direction = 1 if signed > 0 else -1
            continue
        if (qty > 0) == (signed > 0):           # adding to position (rare here)
            entry_price = (entry_price * abs(qty) + f.price * abs(signed)) / abs(qty + signed)
            qty += signed
            continue
        # reducing / closing
        closing = min(abs(signed), abs(qty))
        pnl = (f.price - entry_price) * closing * direction
        yield (entry_ts, f.ts, direction, entry_price, pnl)
        qty += signed
        if qty != 0:                            # reversed straight through zero
            entry_price = f.price
            entry_ts = f.ts
            direction = 1 if qty > 0 else -1


def trade_excursions(fills: list[Fill], bars: list[Bar], pip_size: float) -> list[TradeRecord]:
    out: list[TradeRecord] = []
    for entry_ts, exit_ts, direction, entry_price, pnl in _round_trips(fills):
        mae = mfe = 0.0
        for b in bars:
            if b.ts < entry_ts or b.ts > exit_ts:
                continue
            if direction == 1:
                mae = max(mae, entry_price - b.low)
                mfe = max(mfe, b.high - entry_price)
            else:
                mae = max(mae, b.high - entry_price)
                mfe = max(mfe, entry_price - b.low)
        out.append(TradeRecord(
            entry_ts=entry_ts, exit_ts=exit_ts, direction=direction,
            pnl=pnl,
            mae_pips=mae / pip_size, mfe_pips=mfe / pip_size,
        ))
    return out
