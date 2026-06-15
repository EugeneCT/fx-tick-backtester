"""Historical tick replay and on-the-fly bar resampling."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import polars as pl

from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore


class HistoricalTickFeed:
    """Yields Ticks from a TickStore in global time order across symbols."""

    def __init__(
        self,
        store: TickStore,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> None:
        self._store = store
        self._symbols = symbols
        self._start = start
        self._end = end

    def __iter__(self) -> Iterator[Tick]:
        frames = [
            self._store.read_ticks(sym, self._start, self._end)
            for sym in self._symbols
        ]
        non_empty = [f for f in frames if f.height]
        if not non_empty:
            return iter(())
        merged = pl.concat(non_empty).sort("ts", maintain_order=True)
        return (
            Tick(
                symbol=row["symbol"],
                ts=row["ts"],
                bid=row["bid"],
                ask=row["ask"],
                bid_volume=row["bid_volume"],
                ask_volume=row["ask_volume"],
            )
            for row in merged.iter_rows(named=True)
        )
