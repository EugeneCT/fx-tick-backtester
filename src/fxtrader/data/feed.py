"""Historical tick replay and on-the-fly bar resampling."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

import polars as pl

from fxtrader.core.types import Bar, Tick
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


class Resampler:
    """Builds time bars from a single symbol's ticks, using mid price."""

    def __init__(self, interval: timedelta) -> None:
        self._interval = interval
        self._bucket: datetime | None = None
        self._symbol: str | None = None
        self._o = self._h = self._l = self._c = 0.0
        self._volume = 0

    def _floor(self, ts: datetime) -> datetime:
        epoch = ts.timestamp()
        step = self._interval.total_seconds()
        floored = epoch - (epoch % step)
        return datetime.fromtimestamp(floored, tz=ts.tzinfo)

    def update(self, tick: Tick) -> Bar | None:
        bucket = self._floor(tick.ts)
        emitted: Bar | None = None
        if self._bucket is not None and bucket != self._bucket:
            emitted = self._build()
            self._bucket = None
        mid = tick.mid
        if self._bucket is None:
            self._bucket = bucket
            self._symbol = tick.symbol
            self._o = self._h = self._l = self._c = mid
            self._volume = 1
        else:
            self._h = max(self._h, mid)
            self._l = min(self._l, mid)
            self._c = mid
            self._volume += 1
        return emitted

    def flush(self) -> Bar | None:
        if self._bucket is None:
            return None
        bar = self._build()
        self._bucket = None
        return bar

    def _build(self) -> Bar:
        assert self._bucket is not None and self._symbol is not None
        return Bar(
            symbol=self._symbol,
            ts=self._bucket,
            open=self._o,
            high=self._h,
            low=self._l,
            close=self._c,
            volume=float(self._volume),
        )
