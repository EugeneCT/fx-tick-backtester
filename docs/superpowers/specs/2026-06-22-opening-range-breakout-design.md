# Opening-Range Breakout (session/time-of-day) — Design Spec

**Date:** 2026-06-22
**Status:** Approved (design), pending implementation plan
**Author:** Eugene + Claude

## 1. Goal

Add the platform's fourth strategy and its **first feature-based (non price-pattern) signal**: an
Opening-Range Breakout (ORB). Three price-pattern strategies — naive MA `findings-2026-06-18.md`,
trend-filtered `findings-2026-06-19.md`, mean-reversion `findings-2026-06-21.md` — are all
net-negative after costs on 30m majors, strong evidence the edge is not in price shape at this
resolution. ORB bets on *when, not what*: FX liquidity/volatility concentrate by session, and London
breaks of the quiet Asian-session range are a documented structural effect. The signal is
**time-anchored**, derived from `bar.ts` (UTC) — structurally different from anything tried so far.

**Success bar (carried from prior strategies):** positive out-of-sample Sharpe AND profit factor > 1
after commissions, robust across walk-forward folds, with cross-pair agreement (EURUSD *and* GBPUSD).

## 2. Scope & decisions (locked)

- **Strategy type:** single-symbol, bar-based, **daily state machine** keyed off `bar.ts` (UTC hour +
  date). Reuses the engine — **no engine or live-path changes**.
- **Range build:** accumulate high/low of bars whose UTC hour is in `[range_start_hour, range_end_hour)`
  (the Asian session, e.g. 0–7). No trading during range-building.
- **Entry:** during `[range_end_hour, entry_end_hour)` (London), **one trade per day** — the first
  **confirmed-close** break: `close > range_high` → LONG, `close < range_low` → SHORT. Filled at close
  (no stop-order optimism; we have no real stop orders). `_traded` latch prevents re-entry that day.
- **Exit, risk-first:** (1) price stop/TP via the existing `ExitPolicy` (`none`/`fixed`/`atr`), checked
  first; (2) **time exit** — flatten at `hour >= exit_hour` (e.g. 21:00 UTC, NY close). The time exit
  lives in the strategy (it does not fit `ExitPolicy.distances()`, which is price-based).
- **Daily reset:** on `bar.ts.date()` change, rebuild the range and clear the latch.
- **Reuse:** `ExitPolicy`, `pip_size`, the registry, the MAE/MFE tooling, the `--strategy` CLI seam,
  and the exit-overlay state machine (`_entry_price`/`_stop_dist`/`_tp_dist`/`_last_exit_reason`,
  stop-first same-bar rule).
- **Validation:** EURUSD + GBPUSD (USD-quoted; cross-pair check). USDJPY deferred. Same IS/OOS split
  (IS 2023-06-01→2025-06-01, frozen OOS 2025-06-01→2026-05-01), 30m bars (2 bars/hour; `bar.ts` is the
  bar's open time, so `bar.ts.hour` is exact for hour-boundary logic).

### Out of scope (for now — YAGNI)

- **DST-aware session boundaries.** London/NY observe daylight saving; the data is UTC, so boundaries
  drift ~1h seasonally. v1 uses **fixed UTC hours** (a ~1h seasonal drift, acceptable). DST-awareness
  is a future refinement.
- A break buffer (require break by N pips beyond the range) — v1 breaks on close beyond the edge.
- Re-entry after a same-day exit (one trade/day). Trailing stops, tick-level fills, live wiring of the
  new strategy, USDJPY, generalizing the CLI param plumbing (accrete `orb` int flags).

## 3. Strategy logic — `OpeningRangeBreakout`

Params: `range_start_hour`, `range_end_hour`, `entry_end_hour`, `exit_hour` (ints, UTC), `size`
(float), `exit_policy: ExitPolicy`. `__init__` validates: all hours in `[0, 24)` and
`range_start_hour < range_end_hour <= entry_end_hour < exit_hour` → else `ValueError`. Single symbol;
ignores bars for other symbols.

State: `_day` (current UTC date or None), `_hi`/`_lo` (range, or None), `_traded` (bool latch),
`_entry_price`/`_stop_dist`/`_tp_dist`/`_pos_dir`/`_last_exit_reason`.

Per `on_bar` (after symbol guard + `exit_policy.update(bar)`):

```
day = bar.ts.date(); hour = bar.ts.hour
if day != self._day:                 # new trading day → reset
    self._day = day; self._hi = self._lo = None; self._traded = False
position = self.portfolio.position(self.symbol).quantity

# 1. EXIT FIRST (open position)
if position != 0 and self._entry_price is not None:
    long  stop if bar.low  <= entry - stop_dist ; tp if bar.high >= entry + tp_dist
    short stop if bar.high >= entry + stop_dist ; tp if bar.low  <= entry - tp_dist
    #   STOP leg before TP leg (same-bar stop wins)
    on hit: flatten; _last_exit_reason="stop"|"tp"; reset; return
if position != 0 and hour >= self.exit_hour:
    flatten; _last_exit_reason="time"; reset; return

# 2. RANGE BUILD (Asian window) — never trade here
if range_start_hour <= hour < range_end_hour:
    self._hi = bar.high if self._hi is None else max(self._hi, bar.high)
    self._lo = bar.low  if self._lo is None else min(self._lo, bar.low)
    return

# 3. BREAKOUT ENTRY (one per day, London window)
if (self._hi is not None and not self._traded and position == 0
        and range_end_hour <= hour < entry_end_hour):
    desired = 0
    if bar.close > self._hi:   desired = +size
    elif bar.close < self._lo: desired = -size
    if desired != 0:
        self._traded = True
        order(delta); _entry_price = bar.close; (_stop_dist,_tp_dist) = exit_policy.distances()
```

`flatten`/entry use the same `_reconcile`/`_flatten` pattern as the other strategies
(`_pos_dir`, capture-at-entry, `_last_exit_reason`).

## 4. Reuse — exits, registry, CLI

- **Exits:** unchanged `ExitPolicy` for the price stop/TP; the **time exit is intrinsic to the
  strategy** (not an `ExitPolicy`).
- **Registry (`strategy/registry.py`):** add `STRATEGY_PARAMS["orb"] = ("range_start_hour",
  "range_end_hour", "entry_end_hour", "exit_hour")` and a `make_strategy("orb", ...)` branch building
  `OpeningRangeBreakout(symbol, range_start_hour, range_end_hour, entry_end_hour, exit_hour, size,
  exit_policy)`.
- **CLI (`cli.py`):** add `--range-start/--range-end/--entry-end/--exit-hour` (ints, reuse
  `_int_list`) to `backtest` and `optimize`; an `"orb"` grid branch
  (`{range_start_hour, range_end_hour, entry_end_hour, exit_hour}`); the matching `run_backtest`
  kwargs (defaulting harmlessly so ma/trend/meanrev are unaffected); a `backtest --strategy orb`
  branch printing a strategy-aware summary (flag-driven; `--json/--html` remain MA-only). Update the
  strategy guard to include `"orb"`.

## 5. Research process (staged, separate job)

Run as its own background quant-researcher job (with the same guardrail: do NOT modify `src/`; get
single-backtest metrics via a `research/` python script calling `run_backtest`, WFO via `optimize
--json`). Stages: (1) fix the structural range/London/exit hours, sweep the few genuinely-free choices
(e.g. exit_hour, or a couple of range-window variants) via WFO on EURUSD IS; (2) MAE/MFE diagnostic on
the chosen config; (3) exit candidates (fixed + ATR) on top; (4) frozen OOS + GBPUSD cross-pair;
write `research/findings-<date>.md` with an honest verdict. Serial, coarse bars, tiny grids (feed OOM
risk).

## 6. Repo layout (new / changed)

```
src/fxtrader/strategy/examples/opening_range_breakout.py   NEW  OpeningRangeBreakout
src/fxtrader/strategy/registry.py                          EDIT "orb" in STRATEGY_PARAMS + make_strategy
src/fxtrader/cli.py                                        EDIT --range-start/--range-end/--entry-end/
                                                                --exit-hour + orb grid + run_backtest kwargs
tests/strategy/test_opening_range_breakout.py              NEW  (time-aware _drive_bars harness)
tests/strategy/test_registry.py                            EDIT orb factory case
tests/test_cli_orb.py                                      NEW  backtest + optimize end-to-end
```

## 7. Testing strategy (TDD)

Tests use a time-aware harness `_drive_bars(strategy, bars)` (feeds explicit `Bar`s at chosen UTC
`(date, hour)`, applies fills at `bar.close`), since the minute-incrementing `_drive` can't span a day
cheaply.

- **range tracking** — bars in `[range_start, range_end)` set hi/lo; no trades during range-build.
- **breakout long/short** — post-range London bar with `close > hi` → long / `close < lo` → short;
  `_traded` latched.
- **one trade per day** — a second break the same day → no second entry.
- **no entry outside the entry window** (`hour >= entry_end_hour`) → no entry.
- **no entry if no break** (close stays inside range) → flat.
- **time exit** — open position + `hour >= exit_hour` → flatten (`_last_exit_reason="time"`).
- **stop/TP exit** via `FixedPips`, stop-first, checked before the time exit (`_last_exit_reason="stop"`).
- **day rollover** — new date resets range + clears latch; fresh entry next day.
- **`ValueError`** on bad hour ordering / out-of-range hours.
- **ignores other symbols.**
- **registry**: `make_strategy("orb")` builds it with `exit_policy`; `STRATEGY_PARAMS["orb"]`.
- **CLI**: `backtest --strategy orb` end-to-end on a seeded multi-day store (exit 0, summary names the
  strategy); `optimize --strategy orb` → `param_grid == {range_start_hour, range_end_hour,
  entry_end_hour, exit_hour}`, folds, JSON-serializable; **MA/trend/meanrev grids unchanged**.
- Full suite stays green (baseline 188 passed, 2 skipped).

## 8. Assumptions & risks

- **ORB may have no edge either** — if so, four structurally-distinct negatives is itself a strong
  result about these majors at 30m. The time exit + stop/TP at least bound losses.
- **Fixed-UTC-hours / DST drift** — ~1h seasonal boundary error; mitigated by the effect being broad
  (a 1h-off London window still captures most of the break) and confirmed cross-pair + OOS.
- **30m bar granularity vs hour boundaries** — `bar.ts` is the bar open time; with 30m bars the
  `hour` comparisons are exact at `:00`/`:30`. A bar opening at `06:30` is hour 6 (in a `[0,7)` range).
- **Test seed construction** — multi-day, hour-specific bars are fiddly; the `_drive_bars` harness
  builds them explicitly. Flagged.
- **Feed performance** (whole-window materialization → OOM) constrains breadth — serial, coarse bars,
  tiny grids.

## 9. Definition of done

- `OpeningRangeBreakout` with validation + risk-first/time exits, registry `orb` wiring, and the
  `--strategy orb` CLI (backtest + optimize) implemented and unit-tested; full suite green.
- `--strategy orb` runs end-to-end through `backtest` and `optimize`; ma/trend/meanrev paths unchanged.
- No engine/live-path changes; live command never run; nothing committed/pushed without request.
- The staged research run + `research/findings-<date>.md` (EURUSD + GBPUSD) is the intentionally
  separate follow-up job.
