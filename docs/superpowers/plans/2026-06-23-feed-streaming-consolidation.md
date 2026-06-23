# Feed Streaming + Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `HistoricalTickFeed` stream tick data month-by-month (bounding peak memory to ~1 month instead of the whole window), make `TickStore.read_ticks` partition-aware, and fold in two trivial fixes — all behavior-preserving.

**Architecture:** The store is already partitioned `symbol/year/month.parquet`, and `fxtrader.data.calendar.iter_months` already yields per-month `(ym, month_start, next_month_start)` windows. The feed iterates months chronologically, reading + sorting only one month at a time; `read_ticks` reads only the month-partition files overlapping the range. Output is identical to today (same tick sequence), so every existing test must stay green. Spec: `docs/superpowers/specs/2026-06-23-feed-streaming-consolidation-design.md`.

**Tech Stack:** Python 3.12 (`.venv/bin/...`), polars, typer, pytest. No new dependencies.

## Global Constraints

- Python pinned to **3.12**; invoke tooling as `.venv/bin/python` / `.venv/bin/pytest`.
- **Behavior-preserving.** The streaming feed and partition-aware `read_ticks` must produce the EXACT same data as today (same tick sequence, same rows). The full suite (baseline **216 passed, 2 skipped**) must stay green with identical results — any existing test that changes means equivalence broke. Run `.venv/bin/pytest -q` before each commit.
- **No engine/strategy/broker/risk/live behavior changes.** Only `data/feed.py`, `data/store.py`, the `optimize` guard in `cli.py`, and a doc comment in `mean_reversion_fader.py` (+ a one-line spec note).
- **Order-equivalence (the crux):** `iter_months` yields months chronologically; within each month the ticks are concat+sorted exactly as before; same-`ts` ties collide only within a month, and the per-month `pl.concat(non_empty)` keeps the same symbol order, so `maintain_order=True` breaks ties identically. The streamed sequence equals the old whole-window sequence tick-for-tick.
- **Out of scope:** the CLI param-plumbing generalization (separate next cycle).
- All work on branch `feat/feed-streaming-consolidation` (already created; spec committed at `b815850`). Do NOT create other branches.
- **Every commit message ends with** (one blank line before it): `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

| File | Change |
|---|---|
| `src/fxtrader/data/store.py` (Modify) | `read_ticks` partition-aware (reads only overlapping months) |
| `src/fxtrader/data/feed.py` (Modify) | `__iter__` streams per calendar month |
| `src/fxtrader/cli.py` (Modify) | `optimize` command rejects unknown `--strategy` |
| `src/fxtrader/strategy/examples/mean_reversion_fader.py` (Modify) | inert-bracket doc comment |
| `docs/superpowers/specs/2026-06-21-mean-reversion-fader-design.md` (Modify) | one-line inert-bracket note |
| `tests/data/test_store.py` (Modify) | add a partition-aware-read equivalence test |
| `tests/data/test_feed.py` (Modify) | add streaming output-equivalence + boundary tests |
| `tests/test_cli_optimize.py` (Modify) | add `optimize --strategy bogus` → BadParameter test |

---

## Task 1: `TickStore.read_ticks` — partition-aware

**Files:**
- Modify: `src/fxtrader/data/store.py` (`read_ticks`, lines 65-76)
- Test: `tests/data/test_store.py`

**Interfaces:**
- Consumes: `fxtrader.data.calendar.iter_months(start, end) -> Iterator[(ym: str, month_start: datetime, next_month_start: datetime)]`; `self._month_path(symbol, year, month) -> Path`.
- Produces: `read_ticks(symbol, start, end) -> pl.DataFrame` with the SAME rows/order as today (filtered to `[start, end)`, sorted by `ts`), but reading only the month-partition files overlapping the range.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/data/test_store.py (it already imports TickStore, datetime, timezone; reuse its _tick helper)
def test_read_ticks_partition_aware_matches_full_scan(tmp_path):
    store = TickStore(tmp_path)
    # write ticks spanning three months
    from datetime import datetime, timezone
    a = [_tick_at(datetime(2024, 1, 15, tzinfo=timezone.utc), 1.10),
         _tick_at(datetime(2024, 2, 15, tzinfo=timezone.utc), 1.11),
         _tick_at(datetime(2024, 3, 15, tzinfo=timezone.utc), 1.12)]
    store.write_ticks("EURUSD", a)
    # a window covering only Feb should return exactly the Feb tick
    df = store.read_ticks("EURUSD",
                          datetime(2024, 2, 1, tzinfo=timezone.utc),
                          datetime(2024, 3, 1, tzinfo=timezone.utc))
    assert df.height == 1
    assert df["bid"].to_list() == [1.11]
    # a window covering Jan+Feb returns both, sorted
    df2 = store.read_ticks("EURUSD",
                           datetime(2024, 1, 1, tzinfo=timezone.utc),
                           datetime(2024, 3, 1, tzinfo=timezone.utc))
    assert df2["bid"].to_list() == [1.10, 1.11]
```

> Add a small `_tick_at(ts, bid)` helper at the top of the test file if one isn't present:
> ```python
> from fxtrader.core.types import Tick
> def _tick_at(ts, bid):
>     return Tick("EURUSD", ts, bid, bid + 0.0001, 1.0, 1.0)
> ```
> (Check the file's existing tick helper first — it may already have `_tick(...)`; reuse it and adapt the assertion values.)

- [ ] **Step 2: Run test to verify it fails or passes-by-coincidence**

Run: `.venv/bin/pytest tests/data/test_store.py -k partition_aware -v`
Expected: PASS *with the current full-scan implementation too* (it's an equivalence test — the current code already returns correct data). That's fine: this test pins the CONTRACT that must survive the refactor. If it fails now, the helper/values are wrong — fix them so it passes against current code first.

- [ ] **Step 3: Rewrite `read_ticks` partition-aware**

Replace the body of `read_ticks` (store.py:65-76) with:

```python
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
```

> `iter_months` yields months overlapping `[start, end)` in chronological order, so `paths` is already month-sorted (matching the old `sorted(glob(...))` order). The `.filter(...).sort("ts")` is byte-identical to the old tail. Missing month files are skipped.

- [ ] **Step 4: Run the focused test + the existing store tests**

Run: `.venv/bin/pytest tests/data/test_store.py -v`
Expected: PASS (the new test + all existing round-trip / time-range / multi-month / idempotent / missing-symbol tests — they assert the same contract).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/data/store.py tests/data/test_store.py
git commit  # perf(data): partition-aware read_ticks (read only overlapping months)
```

---

## Task 2: `HistoricalTickFeed.__iter__` — stream month-by-month

**Files:**
- Modify: `src/fxtrader/data/feed.py` (`__iter__`, lines 29-48; add the `iter_months` import)
- Test: `tests/data/test_feed.py`

**Interfaces:**
- Consumes: `TickStore.read_ticks` (Task 1), `iter_months`. Produces the SAME `Tick` sequence as today.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/data/test_feed.py (it imports TickStore, HistoricalTickFeed, Tick, datetime/timezone — reuse its helpers)
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
    from datetime import datetime, timezone, timedelta
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
    from datetime import datetime, timezone
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
    from datetime import datetime, timezone
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [
        Tick("EURUSD", datetime(2024, 1, 10, tzinfo=timezone.utc), 1.10, 1.1001, 1.0, 1.0),
        Tick("EURUSD", datetime(2024, 3, 10, tzinfo=timezone.utc), 1.12, 1.1201, 1.0, 1.0),
    ])  # February is missing entirely
    feed = HistoricalTickFeed(store, ["EURUSD"],
                              datetime(2024, 1, 1, tzinfo=timezone.utc),
                              datetime(2024, 4, 1, tzinfo=timezone.utc))
    assert [t.bid for t in feed] == [1.10, 1.12]   # no error, Feb skipped
```

- [ ] **Step 2: Run tests to verify they fail or pass-by-coincidence**

Run: `.venv/bin/pytest tests/data/test_feed.py -k "streaming" -v`
Expected: PASS against current code too (equivalence tests pin the contract). If `test_streaming_feed_matches_whole_window` doesn't pass now, the reference helper is wrong — fix it to match current feed output first.

- [ ] **Step 3: Rewrite `__iter__` to stream per month**

Add the import at the top of feed.py (after the existing `from fxtrader.data.store import TickStore`):
```python
from fxtrader.data.calendar import iter_months
```
Replace `__iter__` (feed.py:29-48) with:
```python
    def __iter__(self) -> Iterator[Tick]:
        for _ym, m_start, m_end in iter_months(self._start, self._end):
            lo = max(m_start, self._start)
            hi = min(m_end, self._end)
            frames = [self._store.read_ticks(sym, lo, hi) for sym in self._symbols]
            non_empty = [f for f in frames if f.height]
            if not non_empty:
                continue
            merged = pl.concat(non_empty).sort("ts", maintain_order=True)
            for row in merged.iter_rows(named=True):
                yield Tick(
                    symbol=row["symbol"],
                    ts=row["ts"],
                    bid=row["bid"],
                    ask=row["ask"],
                    bid_volume=row["bid_volume"],
                    ask_volume=row["ask_volume"],
                )
```

> This makes `__iter__` a generator (was returning a generator expression — same external behavior: iterable once). `iter_months(start, end)` yields nothing if `start >= end`, so an empty window yields an empty iterator (current contract preserved).

- [ ] **Step 4: Run feed tests + full suite**

Run: `.venv/bin/pytest tests/data/test_feed.py -v && .venv/bin/pytest -q`
Expected: feed tests PASS; full suite **216 passed, 2 skipped** with identical results (behavior-preserving — every backtest/optimize test that consumes the feed produces the same metrics).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/data/feed.py tests/data/test_feed.py
git commit  # perf(data): stream HistoricalTickFeed month-by-month (bounds peak memory)
```

---

## Task 3: Optimize unknown-strategy guard + meanrev inert-bracket doc note

**Files:**
- Modify: `src/fxtrader/cli.py` (`optimize` command — add the guard)
- Modify: `src/fxtrader/strategy/examples/mean_reversion_fader.py` (doc comment)
- Modify: `docs/superpowers/specs/2026-06-21-mean-reversion-fader-design.md` (one-line note)
- Test: `tests/test_cli_optimize.py`

**Interfaces:**
- The `optimize` command rejects an unknown `--strategy` with `typer.BadParameter`, matching the `backtest` command's guard.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_cli_optimize.py (it has a CliRunner `runner` and imports app)
def test_optimize_rejects_unknown_strategy(tmp_path):
    result = runner.invoke(app, [
        "optimize", "--strategy", "bogus", "--symbol", "EURUSD",
        "--start", "2024-01-02", "--end", "2024-01-05",
        "--data-dir", str(tmp_path),
    ])
    assert result.exit_code != 0
    assert "strategy must be" in result.output.lower() or "bogus" in result.output.lower()
```

> Confirm the test file's existing import of `runner`/`app` (it uses `typer.testing.CliRunner`); reuse them. If the file uses a `run_optimize`-style direct call instead of CliRunner, add a CliRunner at the top mirroring `tests/test_cli_trend.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_optimize.py -k unknown_strategy -v`
Expected: FAIL — currently `--strategy bogus` falls into the ma-grid branch and runs (exit 0), so the assertion fails.

- [ ] **Step 3: Add the guard**

In the `optimize` command body in `cli.py` (read it to find the spot — it currently begins by calling `run_optimize(...)`), add as the FIRST statement of the command body:
```python
    if strategy not in ("ma", "trend", "meanrev", "orb", "xpair"):
        raise typer.BadParameter("strategy must be ma | trend | meanrev | orb | xpair")
```
(Mirror the exact guard already present in the `backtest` command.)

- [ ] **Step 4: Add the meanrev doc note (doc-only, no behavior change)**

In `src/fxtrader/strategy/examples/mean_reversion_fader.py`, at the point where `self.exit_policy` is used / `distances()` is read for the stop/TP overlay, add a comment:
```python
        # NOTE: the stop/TP ExitPolicy overlay is effectively a DEAD PATH for this strategy —
        # the signal exit (|z| <= exit_z) fires the bar after entry and pre-empts the brackets
        # every time (verified empirically: research/findings-2026-06-21.md). Tuning stop/TP here
        # cannot change behavior; the no-exit baseline is the strategy's ceiling.
```
And add one line to `docs/superpowers/specs/2026-06-21-mean-reversion-fader-design.md` (in §9 Assumptions & risks or a new note): "Empirically, the stop/TP overlay is inert for this strategy — the z-signal exit pre-empts the brackets every bar (`research/findings-2026-06-21.md`)."

- [ ] **Step 5: Run the test + full suite, commit**

Run: `.venv/bin/pytest tests/test_cli_optimize.py -v && .venv/bin/pytest -q`
Expected: new test PASS; full suite **217 passed, 2 skipped** (216 + 1 new test). The doc-only edits don't change behavior.

```bash
git add src/fxtrader/cli.py src/fxtrader/strategy/examples/mean_reversion_fader.py docs/superpowers/specs/2026-06-21-mean-reversion-fader-design.md tests/test_cli_optimize.py
git commit  # fix(cli): optimize rejects unknown --strategy; doc: meanrev inert-bracket note
```

---

## Self-Review

**Spec coverage:** §3.1 partition-aware `read_ticks` → Task 1. §3.2 streaming feed → Task 2. §2 optimize guard + inert-bracket note → Task 3. §5 testing (output-equivalence, boundary, gap, partial-edge, optimize guard, full suite green) → Tasks 1-3 tests + the behavior-preserving full-suite check in each. §2 CLI generalization is explicitly the next cycle (not a task here).

**Placeholder scan:** Tasks reference reading the current files to place edits (the `optimize` command body, the test file's helpers) with the exact code to add given inline — real procedures, not TBDs. The `_tick_at`/helper notes tell the implementer to reuse existing helpers and adapt values. No "add validation"-style placeholders.

**Type consistency:** `read_ticks(symbol, start, end) -> pl.DataFrame` unchanged signature (Task 1). `iter_months(start, end)` yields `(ym, m_start, m_end)` — used consistently in Tasks 1 and 2. The feed's `__iter__` yields `Tick` with the same six fields as before (Task 2). The guard strategy tuple `("ma","trend","meanrev","orb","xpair")` matches the `backtest` command's guard (Task 3).

**Known executor caveats (flagged):** the equivalence tests in Tasks 1-2 are written to PASS against current code (they pin the contract), then must still pass after the refactor — the implementer should confirm they pass before AND after each change. The behavior-preserving full-suite check is the strongest signal: if any existing test's result changes, equivalence broke. The implementer must read the actual test-file helpers (`_tick`, `runner`, `app`) and reuse them rather than assuming names.
