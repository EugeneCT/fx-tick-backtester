# Price-action edge hunt on a new instrument (gold → AUDUSD) — Design

- **Date:** 2026-06-26
- **Type:** Phased: a small `src/` infra feature (Phase 1, TDD'd) → data download (Phase 2) →
  research study (Phase 3, quant-researcher, no `src/` edits). Controller re-verifies headline numbers.
- **Status:** Approved; pending execution.

## Motivation

Eight distinct approaches are net-negative on EUR/GBP majors (`CLAUDE.md` Research Conclusion). The one
genuinely-different fork is a **different instrument** — gold/FX-crosses are far less arbitraged than
EUR/GBP majors, so a retail price-action edge is more plausible there. We test the already-built PA
methods on **XAUUSD (gold, primary)** and **AUDUSD (trending USD-quoted cross — also cross-instrument
validation)**. Same IS/OOS windows as everything else, for comparability. A PA pattern net-positive on
gold that **also replicates OOS and appears on AUDUSD** would be the first real lead this project has
produced. Honest expectation remains guarded, but the regime is genuinely new.

## Phase 1 — instrument-metadata infra (small `src/` feature, TDD)

The downloader works by symbol but lacks the integer→price scale for non-FX. Changes:
- `src/fxtrader/data/dukascopy.py` `POINT_VALUE`: add `AUDUSD` (= `1e-5`, like EURUSD) and `XAUUSD`
  (**scale verified empirically first — see below**).
- `src/fxtrader/core/instruments.py` `pip_size`: add `AUDUSD` (0.0001) and `XAUUSD` (set from the
  verified scale; gold "pip" convention ≈ 0.1).
- **Empirical scale verification (de-risk — do FIRST, before the full download):** download one hour of
  XAUUSD `.bi5`, decompress + unpack the integer prices, divide by candidate point values, and confirm
  the result is ≈ spot gold (~$2000–2600) for the period. Commit `POINT_VALUE["XAUUSD"]` only once a
  candidate reproduces a sane price. A wrong scale silently corrupts every downstream number — this
  step prevents that.
- Tests: `pip_size`/`POINT_VALUE` return the expected values for both new symbols; existing FX values
  unchanged. Keep the suite green. (Right-sized: ~a dozen lines + tests — a lightweight TDD'd change,
  not the full multi-task subagent pipeline.)

## Phase 2 — data download

Resilient `fxtrader backfill` of **XAUUSD** and **AUDUSD**, monthly **2023-06 → 2026-05**, `--workers 4`
(Dukascopy 503-avoidance, per the documented operational learning), wrapped in the retry loop
(`run_backfill.sh` pattern). Data + `coverage.json` land gitignored under `data/`. A long but bounded
background job; the controller runs it and waits. Integrity-check tick counts/coverage before research.

## Phase 3 — PA research (staged; research-only, no `src/` edits)

Point the already-built PA harness at gold first, then AUDUSD (via `research/` scripts calling
`run_backtest` — mirror `research/run_pa_stage1.py`):

- **Stage 1 — pattern screen:** the four `pa` patterns (pin/engulf/inside/swingbreak) at **1h
  (`bar_minutes=60`) and 4h (240)**, IS gross+net screen, with the MAE/MFE-calibrated ATR exit where a
  pattern shows life (2-pass calibration per the `pa` Phase-B protocol). Report trade counts, gross PF,
  net PF prominently.
- **Validation gauntlet (identical rigor):** IS net-of-cost edge → **OOS replication** (2025-06→2026-05,
  touched once) → **cross-instrument** (a gold edge must also appear on AUDUSD; an AUDUSD edge on gold).
  Account for the search space (4 patterns × 2 TF × 2 instruments) and report it.
- **Escalation:** if a pattern clears IS+OOS, run the data-derived shape-discovery harness
  (`research/shape_analyze.py` adapted) on that instrument as confirmation/extension.

## Cost & sizing

- Real spread comes from tick bid/ask. Commission via `run_backtest` params; start with the existing
  IB-retail model (`max($2, 0.2bps×notional)`) — note it is FX-calibrated and approximate for gold;
  run a small cost-sensitivity check (commission ladder) on any survivor, as in `findings-2026-06-23`.
- **Sizing:** size to ≈ $10k notional per position (gold ≈ 4 oz at ~$2500; AUDUSD 10k) and pass
  `max_position` so the single-currency Portfolio cannot blow through zero (the USDJPY gotcha). Any
  `ruined=True` result is invalid, not signal.

## Windows, deliverable

- IS 2023-06-01→2025-06-01 (tuning/screen). OOS 2025-06-01→2026-05-01 (touched once).
- `research/findings-2026-06-26-new-instrument-pa.md` + JSON under `research/json/` (e.g.
  `inst_pa_*.json`): per pattern × instrument × timeframe trade count, gross/net PF, Sharpe, OOS, and
  the calibrated exits for any candidate. Honest verdict; flag anything resting on thin n. A clean
  negative is valid; a genuine survivor → a separate strategy-promotion / live-eval cycle.

## Non-goals / guardrails

- Phase 3 makes **no `src/` edits**. No new strategy is built (reuses `pa`); a survivor's promotion is a
  separate cycle.
- No timeframe sweep beyond 1h+4h; no pattern beyond the four; OOS touched once. Every extra degree of
  freedom inflates the false-positive rate.
- USDJPY-style sizing discipline enforced via `max_position`.
