import json
from datetime import datetime, timezone

from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore


def _tick(ms_min, bid, ask):
    return Tick(
        symbol="EURUSD",
        ts=datetime(2024, 1, 2, 9, ms_min, tzinfo=timezone.utc),
        bid=bid,
        ask=ask,
        bid_volume=1.0,
        ask_volume=1.0,
    )


def test_write_then_read_roundtrip(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [_tick(0, 1.1000, 1.1002), _tick(1, 1.1001, 1.1003)])

    df = store.read_ticks(
        "EURUSD",
        datetime(2024, 1, 2, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 3, 0, tzinfo=timezone.utc),
    )
    assert df.height == 2
    assert df["bid"].to_list() == [1.1000, 1.1001]


def test_read_filters_by_time_range(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [_tick(0, 1.1, 1.1002), _tick(30, 1.2, 1.2002)])

    df = store.read_ticks(
        "EURUSD",
        datetime(2024, 1, 2, 9, 15, tzinfo=timezone.utc),
        datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
    )
    assert df.height == 1
    assert df["bid"].to_list() == [1.2]


def test_write_is_idempotent_and_sorted(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [_tick(1, 1.1001, 1.1003)])
    store.write_ticks("EURUSD", [_tick(0, 1.1000, 1.1002), _tick(1, 1.1001, 1.1003)])

    df = store.read_ticks(
        "EURUSD",
        datetime(2024, 1, 2, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 3, 0, tzinfo=timezone.utc),
    )
    assert df.height == 2  # duplicate dropped
    assert df["ts"].is_sorted()


def test_manifest_tracks_coverage(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [_tick(0, 1.1, 1.1002), _tick(30, 1.2, 1.2002)])

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["EURUSD"]["min_ts"] == "2024-01-02T09:00:00+00:00"
    assert manifest["EURUSD"]["max_ts"] == "2024-01-02T09:30:00+00:00"


def test_read_missing_symbol_returns_empty(tmp_path):
    store = TickStore(tmp_path)
    df = store.read_ticks(
        "GBPUSD",
        datetime(2024, 1, 2, tzinfo=timezone.utc),
        datetime(2024, 1, 3, tzinfo=timezone.utc),
    )
    assert df.height == 0
