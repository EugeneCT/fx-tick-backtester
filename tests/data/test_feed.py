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


# ---------------------------------------------------------------------------
# Streaming (month-by-month) equivalence and boundary tests
# ---------------------------------------------------------------------------

def _ref_sequence(store, symbols, start, end):
    """The OLD whole-window behavior: read all, concat, sort by ts (stable, symbols in order)."""
    import polars as pl
    frames = [store.read_ticks(s, start, end) for s in symbols]
    non_empty = [f for f in frames if f.height]
    if not non_empty:
        return []
    merged = pl.concat(non_empty).sort("ts", maintain_order=True)
    return [(r["symbol"], r["ts"], r["bid"]) for r in merged.iter_rows(named=True)]


def test_streaming_feed_matches_whole_window(tmp_path):
    from datetime import timedelta
    store = TickStore(tmp_path)
    base = datetime(2024, 1, 28, tzinfo=timezone.utc)
    # two symbols, ticks straddling the Jan->Feb boundary, some same-ts ties
    eur, gbp = [], []
    for i in range(8):
        ts = base + timedelta(days=i)   # crosses into February
        eur.append(Tick("EURUSD", ts, 1.10 + i * 1e-4, 1.10 + i * 1e-4 + 1e-4, 1.0, 1.0))
        gbp.append(Tick("GBPUSD", ts, 1.25 + i * 1e-4, 1.25 + i * 1e-4 + 1e-4, 1.0, 1.0))
    store.write_ticks("EURUSD", eur)
    store.write_ticks("GBPUSD", gbp)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 3, 1, tzinfo=timezone.utc)

    feed = HistoricalTickFeed(store, ["EURUSD", "GBPUSD"], start, end)
    got = [(t.symbol, t.ts, t.bid) for t in feed]
    assert got == _ref_sequence(store, ["EURUSD", "GBPUSD"], start, end)


def test_streaming_feed_partial_edge_months(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [
        Tick("EURUSD", datetime(2024, 1, 10, tzinfo=timezone.utc), 1.10, 1.1001, 1.0, 1.0),
        Tick("EURUSD", datetime(2024, 1, 20, tzinfo=timezone.utc), 1.11, 1.1101, 1.0, 1.0),
        Tick("EURUSD", datetime(2024, 2, 10, tzinfo=timezone.utc), 1.12, 1.1201, 1.0, 1.0),
    ])
    # window starts mid-Jan, ends mid-Feb -> excludes the Jan-10 tick, includes Jan-20 + Feb-10
    feed = HistoricalTickFeed(store, ["EURUSD"],
                              datetime(2024, 1, 15, tzinfo=timezone.utc),
                              datetime(2024, 2, 15, tzinfo=timezone.utc))
    assert [t.bid for t in feed] == [1.11, 1.12]


def test_streaming_feed_gap_month_skipped(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [
        Tick("EURUSD", datetime(2024, 1, 10, tzinfo=timezone.utc), 1.10, 1.1001, 1.0, 1.0),
        Tick("EURUSD", datetime(2024, 3, 10, tzinfo=timezone.utc), 1.12, 1.1201, 1.0, 1.0),
    ])  # February is missing entirely
    feed = HistoricalTickFeed(store, ["EURUSD"],
                              datetime(2024, 1, 1, tzinfo=timezone.utc),
                              datetime(2024, 4, 1, tzinfo=timezone.utc))
    assert [t.bid for t in feed] == [1.10, 1.12]   # no error, Feb skipped
