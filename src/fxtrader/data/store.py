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
        return pl.DataFrame(
            {
                "symbol": [t.symbol for t in ticks],
                "ts": [t.ts for t in ticks],
                "bid": [t.bid for t in ticks],
                "ask": [t.ask for t in ticks],
                "bid_volume": [t.bid_volume for t in ticks],
                "ask_volume": [t.ask_volume for t in ticks],
            },
            schema=_EMPTY.schema,
        )

    def write_ticks(self, symbol: str, ticks: list[Tick]) -> None:
        if not ticks:
            return
        df = self._to_df(ticks)
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
        symbol_dir = self.root / symbol
        if not symbol_dir.exists():
            return _EMPTY.clone()
        files = sorted(symbol_dir.glob("*/*.parquet"))
        if not files:
            return _EMPTY.clone()
        return (
            pl.read_parquet(files)
            .filter((pl.col("ts") >= start) & (pl.col("ts") < end))
            .sort("ts")
        )

    def _update_manifest(self, symbol: str) -> None:
        manifest: dict = {}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
        files = sorted((self.root / symbol).glob("*/*.parquet"))
        df = pl.read_parquet(files)
        manifest[symbol] = {
            "min_ts": df["ts"].min().isoformat(),
            "max_ts": df["ts"].max().isoformat(),
            "rows": df.height,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
