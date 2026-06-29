"""Parquet-backed tick store, partitioned by symbol/year/month."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from fxtrader.core.types import Tick

_COLUMNS = ["symbol", "ts", "bid", "ask", "bid_volume", "ask_volume"]
_EMPTY = pl.DataFrame(
    schema={
        "symbol": pl.String,
        "ts": pl.Datetime(time_unit="us", time_zone="UTC"),
        "bid": pl.Float64,
        "ask": pl.Float64,
        "bid_volume": pl.Float64,
        "ask_volume": pl.Float64,
    }
)


class TickStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    def _month_path(self, symbol: str, year: int, month: int) -> Path:
        return self.root / symbol / f"{year:04d}" / f"{month:02d}.parquet"

    def _to_df(self, ticks: list[Tick]) -> pl.DataFrame:
        # Build in chunks: materializing all 6 columns for a whole dense month
        # (~5-6M ticks for gold) at once spikes memory enough to OOM small boxes.
        # Per-chunk Python column lists stay bounded; Arrow frames concat cheaply.
        if not ticks:
            return _EMPTY.clear()
        chunk = 500_000
        frames = [
            pl.DataFrame(
                {
                    "symbol": [t.symbol for t in part],
                    "ts": [t.ts for t in part],
                    "bid": [t.bid for t in part],
                    "ask": [t.ask for t in part],
                    "bid_volume": [t.bid_volume for t in part],
                    "ask_volume": [t.ask_volume for t in part],
                },
                schema=_EMPTY.schema,
            )
            for part in (ticks[i : i + chunk] for i in range(0, len(ticks), chunk))
        ]
        return frames[0] if len(frames) == 1 else pl.concat(frames)

    def write_ticks(self, symbol: str, ticks: list[Tick]) -> None:
        if not ticks:
            return
        self.write_frame(symbol, self._to_df(ticks))

    def write_frames(self, symbol: str, frames: list[pl.DataFrame]) -> None:
        """Concatenate per-hour frames and partition-write them (memory-frugal path)."""
        frames = [f for f in frames if not f.is_empty()]
        if not frames:
            return
        self.write_frame(symbol, frames[0] if len(frames) == 1 else pl.concat(frames))

    def frame_from_ticks(self, ticks: list[Tick]) -> pl.DataFrame:
        """Public converter: a small batch of ticks -> compact Arrow frame."""
        return self._to_df(ticks)

    def write_frame(self, symbol: str, df: pl.DataFrame) -> None:
        """Partition-write an already-built tick DataFrame (one parquet per month).

        Lets callers (e.g. backfill) accumulate compact Arrow frames per hour and free
        the heavy per-tick Python objects as they go, instead of holding millions of
        ``Tick`` objects for a whole dense month at once.
        """
        if df.is_empty():
            return
        for keys, group in df.group_by(
            df["ts"].dt.year().alias("y"), df["ts"].dt.month().alias("m")
        ):
            year, month = keys
            path = self._month_path(symbol, year, month)
            path.parent.mkdir(parents=True, exist_ok=True)
            merged = group.drop("y", "m") if "y" in group.columns else group
            if path.exists():
                merged = pl.concat([pl.read_parquet(path), merged])
            merged = merged.unique(subset=["ts"], keep="last").sort("ts")
            merged.select(_COLUMNS).write_parquet(path)
        self._update_manifest(symbol)

    def read_ticks(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        from fxtrader.data.calendar import iter_months

        symbol_dir = self.root / symbol
        if not symbol_dir.exists():
            return _EMPTY.clone()
        paths = []
        for _ym, m_start, _m_end in iter_months(start, end):
            path = self._month_path(symbol, m_start.year, m_start.month)
            if path.exists():
                paths.append(path)
        if not paths:
            return _EMPTY.clone()
        return (
            pl.read_parquet(paths)
            .filter((pl.col("ts") >= start) & (pl.col("ts") < end))
            .sort("ts")
        )

    def _update_manifest(self, symbol: str) -> None:
        manifest: dict = {}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
        files = sorted((self.root / symbol).glob("*/*.parquet"))
        if not files:
            return
        # Lazy streaming scan over just the ts column (uses parquet metadata for
        # min/max/count) — eagerly reading every month into one DataFrame after each
        # write is O(months^2) and balloons memory on dense symbols like gold.
        stats = (
            pl.scan_parquet(files)
            .select(
                pl.col("ts").min().alias("min_ts"),
                pl.col("ts").max().alias("max_ts"),
                pl.len().alias("rows"),
            )
            .collect(engine="streaming")
        )
        manifest[symbol] = {
            "min_ts": stats["min_ts"][0].isoformat(),
            "max_ts": stats["max_ts"][0].isoformat(),
            "rows": int(stats["rows"][0]),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
