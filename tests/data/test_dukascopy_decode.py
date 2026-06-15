import lzma
import struct
from datetime import datetime, timezone

from fxtrader.data.dukascopy import POINT_VALUE, build_url, decode_bi5


def _make_bi5(records):
    raw = b"".join(struct.pack(">IIIff", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def test_decode_bi5_eurusd():
    hour = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    # (ms_offset, ask_int, bid_int, ask_vol, bid_vol)
    records = [
        (0, 110020, 110000, 1.5, 2.0),
        (1500, 110030, 110010, 0.5, 0.7),
    ]
    ticks = decode_bi5(_make_bi5(records), hour, "EURUSD")

    assert len(ticks) == 2
    assert ticks[0].symbol == "EURUSD"
    assert ticks[0].ts == datetime(2024, 1, 2, 9, 0, 0, tzinfo=timezone.utc)
    assert round(ticks[0].ask, 5) == 1.10020
    assert round(ticks[0].bid, 5) == 1.10000
    assert ticks[0].ask_volume == 1.5
    assert ticks[1].ts == datetime(2024, 1, 2, 9, 0, 1, 500_000, tzinfo=timezone.utc)


def test_decode_bi5_usdjpy_uses_3_decimal_scaling():
    hour = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    ticks = decode_bi5(_make_bi5([(0, 145123, 145100, 1.0, 1.0)]), hour, "USDJPY")
    assert round(ticks[0].ask, 3) == 145.123
    assert round(ticks[0].bid, 3) == 145.100


def test_decode_empty_buffer_returns_no_ticks():
    assert decode_bi5(lzma.compress(b"", format=lzma.FORMAT_ALONE), datetime.now(timezone.utc), "EURUSD") == []


def test_point_values_present():
    assert POINT_VALUE["EURUSD"] == 1e-5
    assert POINT_VALUE["USDJPY"] == 1e-3


def test_build_url_uses_zero_indexed_month():
    url = build_url("EURUSD", datetime(2024, 3, 5, 9, tzinfo=timezone.utc))
    # March -> month index 02
    assert url == (
        "https://datafeed.dukascopy.com/datafeed/EURUSD/2024/02/05/09h_ticks.bi5"
    )
