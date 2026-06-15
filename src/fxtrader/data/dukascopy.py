"""Dukascopy tick data: .bi5 decoding and URL construction.

.bi5 = LZMA-compressed stream of 20-byte big-endian records:
    (ms_offset: uint32, ask: uint32, bid: uint32, ask_vol: float32, bid_vol: float32)
Integer prices are scaled by the symbol's point value. Dukascopy month is 0-indexed.
"""

from __future__ import annotations

import lzma
import struct
import urllib.error
import urllib.request
from collections.abc import Callable
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


Downloader = Callable[[str], bytes]


def _http_download(url: str, *, timeout: float = 30.0) -> bytes:
    """Default downloader. Returns empty bytes for a 404 (no data that hour)."""
    req = urllib.request.Request(url, headers={"User-Agent": "fxtrader/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return b""
        raise


def fetch_hour(
    symbol: str,
    hour: datetime,
    *,
    downloader: Downloader = _http_download,
) -> list[Tick]:
    """Download and decode a single hour of ticks."""
    raw = downloader(build_url(symbol, hour))
    return decode_bi5(raw, hour, symbol)


def download_range(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    downloader: Downloader = _http_download,
) -> list[Tick]:
    """Download ticks for [start, end) one hour at a time, in time order."""
    ticks: list[Tick] = []
    cursor = start.replace(minute=0, second=0, microsecond=0)
    while cursor < end:
        ticks.extend(fetch_hour(symbol, cursor, downloader=downloader))
        cursor += timedelta(hours=1)
    return ticks
