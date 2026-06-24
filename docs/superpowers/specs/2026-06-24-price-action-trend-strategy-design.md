# Price-Action + Trend strategy with MAE-calibrated exits — Design

- **Date:** 2026-06-24
- **Registry name:** `pa`
- **Status:** Design approved; pending implementation plan.
- **Type:** New strategy (a 6th signal family) + a research protocol to evaluate it.

## Context

`CLAUDE.md` documents a rigorous **negative** result: five structurally-distinct strategies
(`ma`, `trend`, `meanrev`, `orb`, `xpair`), a 30/60/120/240m timeframe sweep, a native-4h
re-optimization, and a zero-cost probe all concluded that *simple retail price/time/relative-value
signals have no meaningful edge on EUR/GBP/JPY majors at 30m–4h, 2023-06→2026-05*. The recurring
signature is gross profit factor ≈ 1.0 (a coin-flip before costs).

This work is **not** one of the exhausted experiments:

1. **Price-action patterns** (pin bar, engulfing, inside-bar break, swing-structure break) are a new
   signal family — none of the five tested strategies use candlestick / structure patterns.
2. **MAE/MFE-calibrated exits** are a different lever. `metrics/excursions.py` already computes
   winner/loser MAE/MFE percentiles but has only ever been used as a *diagnostic*, never to *set*
   stops/targets.
3. **8h (480m) bars** were never tested; the prior sweep stopped at 240m (=4h).

It is, however, the same instruments where the edge was not found. We treat this as a genuine edge
hunt with full IS/OOS rigor and a real expectation that most or all variants come back negative. A
clean, fast negative is a valid outcome.

## Goal

Find out whether any famous, mechanically-codable price-action pattern — gated by a same-timeframe
trend filter, with stops/targets calibrated from Maximum-Adverse/Favorable-Excursion analysis —
produces a net-of-cost edge on EURUSD / GBPUSD / USDJPY at 4h and 8h that survives out-of-sample and
is consistent across pairs.

### Non-goals

- No live deployment (research only; a human promotes finalists, per process).
- No multi-timeframe (higher-TF) trend gate — same-TF long MA only (decided).
- No consolidation-cycle-2 CLI generalization — we follow the existing accreted per-strategy
  plumbing pattern and leave the generalization deferred.
- No notional / multi-currency portfolio accounting — USDJPY is bounded by `max_position` instead.

## Two phases

- **Phase A — Build (feature):** implement strategy + pattern detectors + wiring + tests in `src/`,
  via TDD, reviewed, merged to `master` (`--no-ff`). Keep the suite green (baseline 221 passed,
  2 skipped).
- **Phase B — Research (edge hunt):** dispatched to the `quant-researcher` agent, which **never edits
  `src/`**. Drives `run_backtest` / `optimize --json` from `research/` scripts, runs the 2-pass MAE
  calibration, writes a `findings-*.md` report. Controller re-verifies headline numbers against JSON.

---

## Phase A — The strategy

### `PriceActionTrend` (`src/fxtrader/strategy/examples/price_action.py`)

Bar-based, single-symbol, mirroring the EXIT-FIRST structure of `OpeningRangeBreakout` /
`TrendFilteredCrossover`.

**Constructor params:**

| param | meaning |
|---|---|
| `symbol` | traded instrument |
| `pattern` | one of `"pin"`, `"engulf"`, `"inside"`, `"swingbreak"` — exactly one per backtest |
| `trend` | long-lookback MA window; trend direction = sign(close − trend_MA) |
| `size` | position size (units) |
| `max_hold_bars` | hard time-exit: flatten after N bars in a trade (safety on coarse bars; gives Pass-1 a bounded holding window for clean MFE measurement) |
| pattern params | `wick_ratio`, `body_frac` (pin); `swing_lookback` (swingbreak); `inside_break_bars` (inside) — see detectors |
| `exit_policy` | `ExitPolicy` (the existing `AtrMultiple` for calibrated runs; `NoExit`/wide for Pass 1) |

**`on_bar` flow (single symbol; ignore other symbols):**

1. `exit_policy.update(bar)`; maintain the `closes` deque (maxlen = `trend`) and the rolling bar
   history needed by the active detector.
2. **EXIT FIRST** (bar-granular, stop-first same-bar rule — copy the `orb`/`trend` block): if in a
   position, check stop, then TP, then `max_hold_bars` (time), then flatten + reset on hit.
3. If `len(closes) < trend`, return (warm-up).
4. Compute `trend_dir = sign(close − trend_MA)`.
5. If **flat**, evaluate the active pattern detector → `signal ∈ {−1, 0, +1}`. Enter **only if
   `signal != 0` and `signal == trend_dir`** (bullish pattern + uptrend → long; mirror for short).
   On entry: submit market order at close, latch `entry_price`, freeze `(stop_dist, tp_dist)` from
   `exit_policy.distances()`, set `_pos_dir`, reset the bar counter.

**Direction rule rationale:** a bullish pin in an uptrend is the canonical pullback-rejection long;
counter-trend patterns are suppressed by the gate. This is the literal "price action + trend"
combination.

### Pattern detectors (`src/fxtrader/strategy/patterns.py`)

Pure functions, no I/O, no state, unit-tested in isolation. Each returns −1 / 0 / +1. Small-range /
doji guards required (avoid divide-by-zero; `eps` on body).

- **`pin_bar(bar, *, wick_ratio, body_frac) -> int`**
  - `rng = high − low`; `body = |close − open|`; `upper = high − max(open,close)`;
    `lower = min(open,close) − low`.
  - Bullish (+1): `lower ≥ wick_ratio × max(body, eps)` **and** `body ≤ body_frac × rng` **and**
    close in the upper third of the range.
  - Bearish (−1): mirror with `upper`.
- **`engulfing(prev, bar) -> int`**
  - Bullish (+1): `prev` bearish, `bar` bullish, and `bar` body engulfs `prev` body
    (`close ≥ prev.open and open ≤ prev.close`).
  - Bearish (−1): mirror.
- **`is_inside_bar(prev, bar) -> bool`** + strategy-managed break:
  - Inside bar: `bar.high ≤ prev.high and bar.low ≥ prev.low`.
  - When detected, the strategy latches the *mother bar* `(high, low)` and arms a breakout for the
    next `inside_break_bars` bars; entry triggers on a close beyond the mother range in the trend
    direction (+1 above high, −1 below low). (Pattern-specific state lives in the strategy; the
    detector stays pure.)
- **`swing_break(bars, lookback) -> int`**
  - Swing high = max(high) over the prior `lookback` bars; swing low = min(low). Close above swing
    high → +1; below swing low → −1. (Pure structure on the trading TF; **overlaps conceptually with
    `orb`** — flagged. Differs in that it is non-session and trend-gated.)

### Wiring

- `strategy/registry.py`:
  - `STRATEGY_PARAMS["pa"] = ("pattern", "trend", "wick_ratio", "body_frac", "swing_lookback",
    "inside_break_bars", "max_hold_bars")` (the optimizer grids over the subset relevant to the
    chosen `pattern`).
  - `make_strategy` `"pa"` branch constructing `PriceActionTrend(...)` with `exit_policy`.
- `cli.py`:
  - Extend `run_backtest(...)` with the new kwargs (`pattern`, `wick_ratio`, `body_frac`,
    `swing_lookback`, `inside_break_bars`, `max_hold_bars`) and a `"pa"` dispatch branch — following
    the existing per-strategy pattern (NOT generalized; cycle-2 stays deferred).
  - Add a `"pa"` branch to `_strategy_run_fn` so `optimize --json` walk-forward works (needed for the
    rigor bar — WFO directly guards against the single-fold fluke that bit the 240m sweep).
  - A `pa` subcommand on the `backtest` CLI is optional; research drives `run_backtest` directly.

### Tests (Phase A — TDD)

- **Detector unit tests:** for each pattern, a synthetic bar/bar-sequence that is a clean positive
  (long and short) and clear negatives (e.g. a doji is not a pin; a non-engulfing bar; a
  non-contained bar is not inside). Cover the small-range/doji guard.
- **Strategy tests:** trend gate blocks a counter-trend pattern (0 trades); an aligned pattern fires
  one entry; EXIT-FIRST honors stop before TP same-bar; `max_hold_bars` flattens; warm-up suppresses
  trades until `trend` closes are seen.
- Whole suite stays green.

---

## Phase B — Research protocol (edge hunt)

Driven by the `quant-researcher` from `research/` scripts (untracked), mirroring `run_orb_exit.py`.
**All calibration is in-sample only; OOS is touched once.**

### MAE/MFE exit calibration (2-pass, per pattern × pair × timeframe)

1. **Pass 1 — natural excursions.** Run pattern + trend with a wide stop (large `stop_atr` or
   `NoExit`) plus `max_hold_bars`. Capture per-trade MAE/MFE via `trade_excursions(fills, bars,
   pip_size)`; split winners/losers via `summarize_excursions`.
2. **Calibrate.** Set `stop_atr ≈ winners' MAE upper percentile (p75)` expressed in ATR units
   (convert pip percentile ÷ median entry-time ATR-in-pips). Set `tp_atr` from the MFE percentile
   (e.g. winners' MFE p50–p75). Goal: winners survive the stop, losers get cut, target sits where
   favorable excursion typically peaks.
3. **Pass 2 — confirm IS.** Re-run IS with `AtrMultiple(atr_period, stop_atr, tp_atr)`; require it to
   beat the Pass-1 / no-edge baseline net of cost.
4. **OOS — once.** Run the IS-frozen config on 2025-06-01→2026-05-01.

### Scope

- **Patterns:** all four, each evaluated **independently** (no combined multi-signal config — that is
  the multiple-comparison over-fit trap `CLAUDE.md` warns about).
- **Timeframes:** 4h (`bar_minutes=240`) and 8h (`bar_minutes=480`).
- **Pairs:** EURUSD, GBPUSD (USD-quoted, sized cleanly) and **USDJPY bounded via `max_position`** so a
  fixed-size position cannot blow through zero (single-currency sizing gotcha). Treat a `ruined` flag
  on USDJPY as invalid, not signal.
- **Windows:** IS 2023-06-01→2025-06-01 (all tuning incl. WFO via `optimize --json`); OOS
  2025-06-01→2026-05-01 touched once for finalists.
- **Costs:** standard IB retail model (fills at bid/ask + `max($2, 0.2bps×notional)`); at 10k size the
  $2 min binds (~4 pips round-trip single-leg).
- **Runtime:** tick-bound (~7 min per 2yr single-symbol pass). Keep grids tiny and windows to the IS
  year per run; prefer the streaming feed (already month-by-month).

### Survival / kill criteria

A pattern is a **finalist** only if it is IS-positive net of cost **and** holds up OOS **and** is
consistent across both USD-quoted pairs (not a single-fold/single-pair fluke). Per the prior result,
the expectation is that most or all variants die. The deliverable is a `research/findings-2026-06-24-*.md`
report stating, per pattern × pair × timeframe: trade count, gross vs net profit factor, Sharpe,
total return, OOS behavior, and the calibrated stop/TP — and the honest verdict, positive or negative.

## Risks & gotchas

- **USDJPY sizing** — must pass a finite `max_position`; otherwise ~15× leverage can ruin the account.
- **Coarse-bar trade counts** — 8h bars yield few trades; small samples make IS positives easy to
  fluke. WFO + cross-pair consistency + OOS are the guardrails. Report trade counts prominently.
- **`max_hold_bars` interaction with MAE** — Pass 1 must hold long enough to observe natural
  excursions but bounded enough to terminate; pick `max_hold_bars` deliberately and report it.
- **Swing-break / ORB overlap** — note it; differentiate by non-session structure + trend gate.
- **No `src/` edits during Phase B** — research uses `run_backtest`/`optimize --json` only.

## Out of scope / deferred

- Higher-timeframe trend gate, S/R-level detection, multi-pattern ensembles.
- Consolidation cycle 2 (generic per-strategy param mechanism).
- Notional / multi-currency accounting (would let USDJPY be sized honestly).
- Sub-bar microstructure / new instruments / lower-cost execution (a different project).
