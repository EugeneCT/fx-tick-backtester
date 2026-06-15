from datetime import datetime, timezone

from fxtrader.core.types import Tick
from fxtrader.data.feed import HistoricalTickFeed
from fxtrader.data.store import TickStore


def _tick(symbol, minute, bid, ask):
    return Tick(symbol=symbol, ts=datetime(2024, 1, 2, 9, minute, tzinfo=timezone.utc),
                bid=bid, ask=ask)


def test_feed_yields_single_symbol_in_order(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [_tick("EURUSD", 1, 1.1001, 1.1003),
                                 _tick("EURUSD", 0, 1.1000, 1.1002)])

    feed = HistoricalTickFeed(store, ["EURUSD"],
                              datetime(2024, 1, 2, tzinfo=timezone.utc),
                              datetime(2024, 1, 3, tzinfo=timezone.utc))
    out = list(feed)
    assert [t.ts.minute for t in out] == [0, 1]
    assert all(isinstance(t, Tick) for t in out)


def test_feed_merges_symbols_by_time(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [_tick("EURUSD", 0, 1.10, 1.1002),
                                 _tick("EURUSD", 2, 1.10, 1.1002)])
    store.write_ticks("GBPUSD", [_tick("GBPUSD", 1, 1.27, 1.2702)])

    feed = HistoricalTickFeed(store, ["EURUSD", "GBPUSD"],
                              datetime(2024, 1, 2, tzinfo=timezone.utc),
                              datetime(2024, 1, 3, tzinfo=timezone.utc))
    out = [(t.symbol, t.ts.minute) for t in feed]
    assert out == [("EURUSD", 0), ("GBPUSD", 1), ("EURUSD", 2)]


def test_feed_empty_store_yields_nothing(tmp_path):
    feed = HistoricalTickFeed(TickStore(tmp_path), ["EURUSD"],
                              datetime(2024, 1, 2, tzinfo=timezone.utc),
                              datetime(2024, 1, 3, tzinfo=timezone.utc))
    assert list(feed) == []
