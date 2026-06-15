"""Dukascopy tick data: .bi5 decoding and URL construction.

.bi5 = LZMA-compressed stream of 20-byte big-endian records:
    (ms_offset: uint32, ask: uint32, bid: uint32, ask_vol: float32, bid_vol: float32)
Integer prices are scaled by the symbol's point value. Dukascopy month is 0-indexed.
"""

from __future__ import annotations

import lzma
import struct
from datetime import datetime, timedelta

from fxtrader.core.types import Tick

POINT_VALUE: dict[str, float] = {
    "EURUSD": 1e-5,
    "GBPUSD": 1e-5,
    "USDJPY": 1e-3,
}

_RECORD = struct.Struct(">IIIff")
_BASE_URL = "https://datafeed.dukascopy.com/datafeed"


def build_url(symbol: str, hour: datetime) -> str:
    """Dukascopy datafeed URL for a single hour. Month is 0-indexed."""
    return (
        f"{_BASE_URL}/{symbol}/{hour.year:04d}/{hour.month - 1:02d}/"
        f"{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"
    )


def decode_bi5(raw: bytes, hour_start: datetime, symbol: str) -> list[Tick]:
    """Decode one hour's compressed .bi5 payload into Ticks."""
    if not raw:
        return []
    point = POINT_VALUE[symbol]
    data = lzma.decompress(raw)
    ticks: list[Tick] = []
    for ms, ask_i, bid_i, ask_v, bid_v in _RECORD.iter_unpack(data):
        ticks.append(
            Tick(
                symbol=symbol,
                ts=hour_start + timedelta(milliseconds=ms),
                bid=bid_i * point,
                ask=ask_i * point,
                bid_volume=bid_v,
                ask_volume=ask_v,
            )
        )
    return ticks
