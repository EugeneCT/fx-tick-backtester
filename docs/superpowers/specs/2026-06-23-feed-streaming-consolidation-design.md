# Feed Streaming + Consolidation (Cycle 1) — Design Spec

**Date:** 2026-06-23
**Status:** Approved (design), pending implementation plan
**Author:** Eugene + Claude

## 1. Goal

Consolidate the now-proven platform with its highest-value, lowest-risk cleanup: make the
historical tick feed **stream month-by-month** instead of materializing the whole window in memory.
The current `HistoricalTickFeed.__iter__` does `pl.concat(non_empty).sort("ts")` over the entire
date range, which **OOM-killed every multi-year and two-symbol research run this session** (peak
~5.7 GiB on a 6-month two-symbol optimize; a full-year two-symbol run exceeds the box) and imposed a
~5× runtime tax. Streaming bounds peak memory to ~1 month. Folded in: two trivial fixes surfaced
during the research arc (an optimize unknown-strategy guard; a doc note on the meanrev inert
bracket).

**This is research infrastructure, not a strategy** — five distinct strategies were all net-negative
after costs (`research/findings-2026-06-22-xpair.md` and the four prior), so the work now is making
the platform solid, not hunting edge.

## 2. Scope & decisions (locked)

- **Behavior-preserving refactor.** The streaming feed must yield the **exact same `Tick` sequence**
  (order and content) as the current whole-window concat+sort. This is the correctness crux and the
  primary test.
- **Three changes, one cycle:**
  1. `HistoricalTickFeed.__iter__` streams per calendar month (via the existing
     `fxtrader.data.calendar.iter_months`), reading + sorting only one month at a time.
  2. `TickStore.read_ticks` becomes **partition-aware**: read only the month-partition files
     overlapping `[start, end)` (via `iter_months` + `_month_path`) instead of globbing all of a
     symbol's files. Identical returned rows; fewer file reads (so the per-month feed read is O(one
     file/symbol), not O(all files)).
  3. Two trivial fixes (below).
- **No engine/strategy/broker/risk/live changes.** Only `data/feed.py`, `data/store.py`, and the
  small fixes in `cli.py` + a doc comment in the meanrev strategy.
- **Out of scope (next cycle):** the CLI param-plumbing generalization (replacing the accreted
  per-strategy flags/grids/kwargs across all five strategies) — substantial, touches all five merged
  byte-identical paths, gets its own brainstorm→spec→plan cycle. Not bundled here.

### The two folded-in fixes

- **Optimize unknown-strategy guard** (`cli.py`, `optimize` command): add the guard the `backtest`
  command already has — `if strategy not in ("ma","trend","meanrev","orb","xpair"): raise
  typer.BadParameter("strategy must be ma | trend | meanrev | orb | xpair")`. Today an unknown
  `--strategy` silently falls into the ma-grid branch.
- **Meanrev inert-bracket doc note** (`strategy/examples/mean_reversion_fader.py`): a concise comment
  (and a one-line spec note) recording that the stop/TP `ExitPolicy` is a **dead path** for the
  fader — the z-signal exit (`|z| ≤ exit_z`) fires the bar after entry and pre-empts the brackets
  every time (verified, `research/findings-2026-06-21.md`). Doc-only; no behavior change.

## 3. Data-layer streaming

### 3.1 `TickStore.read_ticks(symbol, start, end)` — partition-aware

Today: `files = sorted(symbol_dir.glob("*/*.parquet"))`, read all, filter to `[start, end)`. Change
to iterate `iter_months(start, end)`, build each candidate path via `_month_path(symbol, year,
month)`, read only the files that exist (skip missing months = data gaps), then concat + filter to
`[start, end)` + sort by `ts`. The returned DataFrame is **identical** to today's (same rows, same
order); only the set of files read shrinks to the overlapping months.

Edge cases: empty range / no overlapping files → empty DataFrame (current contract preserved); a
month file present but with no rows in `[lo, hi)` contributes nothing after the filter.

### 3.2 `HistoricalTickFeed.__iter__` — month-by-month streaming

```
def __iter__(self):
    for _ym, m_start, m_end in iter_months(self._start, self._end):
        lo = max(m_start, self._start)
        hi = min(m_end, self._end)
        frames = [self._store.read_ticks(sym, lo, hi) for sym in self._symbols]
        non_empty = [f for f in frames if f.height]
        if not non_empty:
            continue                                   # gap month → skip
        merged = pl.concat(non_empty).sort("ts", maintain_order=True)   # ONE month only
        for row in merged.iter_rows(named=True):
            yield Tick(symbol=row["symbol"], ts=row["ts"], bid=row["bid"],
                       ask=row["ask"], bid_volume=row["bid_volume"], ask_volume=row["ask_volume"])
```

**Order-equivalence argument (why output is identical):** `iter_months` yields months in chronological
order; within each month the ticks are concat+sorted exactly as before. Same-`ts` ties can only occur
*within* a single month (two ticks at the same timestamp are in the same month), and the per-month
`pl.concat(non_empty)` preserves the same symbol ordering as the old whole-window concat, so
`maintain_order=True` breaks ties identically. Therefore the streamed sequence equals the old
whole-window sequence tick-for-tick. Peak memory drops from "whole window" to "one month".

## 4. Repo layout (changed)

```
src/fxtrader/data/feed.py                  EDIT  __iter__ streams per month
src/fxtrader/data/store.py                 EDIT  read_ticks partition-aware
src/fxtrader/cli.py                         EDIT  optimize unknown-strategy guard
src/fxtrader/strategy/examples/mean_reversion_fader.py  EDIT  inert-bracket doc comment
tests/data/test_feed.py                    EDIT  add streaming output-equivalence + boundary tests
tests/test_cli_optimize.py (or a cli test) EDIT  optimize bogus-strategy → BadParameter
docs/superpowers/specs/2026-06-21-mean-reversion-fader-design.md  EDIT  one-line inert-bracket note
```

## 5. Testing strategy (TDD)

- **Feed output-equivalence (the key test):** seed a multi-month, two-symbol store; build the
  reference sequence by reading ALL seeded ticks and doing a single concat+sort (the old behavior);
  assert the streaming feed yields that exact `Tick` sequence (order + every field).
- **Cross-month-boundary ordering:** ticks straddling a month boundary appear in correct global time
  order (last tick of month N before first tick of month N+1).
- **Partial first/last month:** a `[start, end)` window whose edges fall *inside* a month yields only
  the in-range ticks (window-edge filter still applies per month).
- **Gap month skipped:** a store missing a middle month streams the surrounding months with no error
  and no spurious ticks.
- **`read_ticks` unchanged contract:** existing `tests/data/test_store.py` (round-trip, time-range
  filter, multi-month, idempotent write, missing-symbol-empty) stay green unmodified — the
  partition-aware read returns identical data.
- **Optimize guard:** `optimize --strategy bogus ...` exits non-zero with `BadParameter`.
- **Whole suite green (baseline 216 passed, 2 skipped).** Because the refactor is behavior-preserving,
  EVERY existing backtest/optimize test must produce identical results (same tick sequence in → same
  metrics out). Any existing test that changes means equivalence broke — the strongest signal.

## 6. Assumptions & risks

- **Order-equivalence depends on the tie-break argument in §3.2** — covered head-on by the
  output-equivalence test against a captured reference, plus the unchanged full suite.
- **Per-month re-reads** add a small constant of Parquet open/close overhead vs one big read, but the
  partition-aware `read_ticks` reads far fewer files overall; net runtime improves (the OOM/5× tax
  goes away). Not a perf regression in practice.
- **`iter_months` boundary semantics** (`[m_start, m_end)`, tz-aware UTC) must align with
  `read_ticks`'s `[start, end)` filter — the `lo/hi` clamping in §3.2 handles partial edge months.
- **Empty/zero-symbol or fully-gap windows** must still produce an empty iterator (current contract).
- Streaming is **lazy** (a generator) — consumers already iterate once; no consumer relies on
  materializing the feed twice.

## 7. Definition of done

- `HistoricalTickFeed` streams month-by-month; `read_ticks` is partition-aware; both behavior-preserving
  (output-equivalence + boundary tests pass; full suite green at 216/2 with identical results).
- Optimize command rejects unknown `--strategy` with `BadParameter`; meanrev inert-bracket note added.
- No engine/strategy/broker/risk/live behavior changes; live command never run; nothing committed/pushed
  without request.
- The CLI param-plumbing generalization is the intentionally-separate next cycle.
