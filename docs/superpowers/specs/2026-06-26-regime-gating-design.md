# Unsupervised regime-gating of existing strategies — Design / research protocol

- **Date:** 2026-06-26
- **Type:** Research study (NOT a `src/` feature). Executed by the `quant-researcher` under the hard
  guardrail (never edits `src/`; works in `research/`). Controller re-verifies headline numbers
  against JSON. A survivor triggers a *separate* feature cycle (regime-gated strategy in `src/`).
- **Status:** Approved; pending dispatch.

## Hypothesis

The six strategies are net-negative *unconditionally*. This tests whether any is net-positive
**conditioned on an unsupervised market regime** — i.e. "the signal isn't dead, it's *untimed*"
(trend-following only pays in trending regimes; mean-reversion only in ranging regimes). This is the
best remaining unsupervised shot because it tests **timing/conditioning**, not raw signal — but the
prior is still strongly negative (seven approaches negative) and regime-splitting thins samples, so
long odds remain. A clean negative is again a valid result.

## 1. Regime model (unsupervised, IS-fit then frozen)

- Two trailing, **no-look-ahead** features per 30m bar:
  - **realized volatility** = stdev of the last 20 bar log-returns (or ATR%), 
  - **trend strength** = Kaufman `efficiency_ratio` (`fxtrader.strategy.indicators.efficiency_ratio`)
    over the last 20 closes.
- Standardize both with **EURUSD-IS-frozen** mean/std.
- Cluster into **K = 3 regimes** with a numpy Gaussian-mixture or k-means (sklearn absent in `.venv`),
  fixed seed, **fit on EURUSD IS only**, then **frozen** (centroids + standardization params saved).
- Label every bar by nearest frozen component. Apply the SAME frozen model to EURUSD (IS+OOS) and
  GBPUSD (IS+OOS) — vol and ER are comparable across the two majors, so a regime means the same thing
  on both pairs (mirrors the shape study's cross-pair approach). Report the interpretable character of
  each regime (e.g. quiet-range / trending / volatile) from its centroid.

## 2. Gating mechanism (research-only — no `src/` edits)

- Run each strategy **once** via `fxtrader.cli.run_backtest` at fixed params (per pair, per window).
- Extract round-trip trades (entry ts, exit ts, **cost-inclusive net P&L** — real fills + commission +
  spread) by FIFO-pairing `result.fills` (reuse the `round_trips_with_commission` logic from
  `research/run_pa_stage1.py`).
- **Tag each trade by the regime of its ENTRY bar** (frozen model). Per regime, compute conditional
  **net profit factor, Sharpe, win-rate, and n (trade count)**.
- This is a **screen**, not a faithful "trade-only-in-regime-X" backtest (position continuity would
  differ). A survivor gets the faithful test at promotion (§5).

## 3. Strategies, params, timeframe

All at **`bar_minutes=30`, size 10k, `exit_kind="none"`** (each strategy's intrinsic logic governs
exits; brackets are off so we test the raw signal conditioned on regime). Params **fixed at the
documented finalists — NOT re-optimized** (re-optimizing params *and* regimes = double-mining):

| strategy | params (from `research/findings-*.md`) |
|---|---|
| `ma` | fast=20, slow=100 |
| `trend` | fast=20, slow=100, trend=200 |
| `meanrev` | lookback=20, entry_z=2.0, exit_z=0.5, er_threshold=0.3 |
| `orb` | range_start=0, range_end=7, entry_end=16, exit_hour=17 |
| `xpair` | lookback=20, entry_z=2.5, exit_z=0.5, stop_z=4.0, symbol2=GBPUSD |

The agent must read the exact finalist values from the findings reports and **report the params it
used**; the controller re-verifies. EURUSD = discovery pair; GBPUSD = cross-pair validation.

**xpair is special-cased:** it is itself a cross-pair (EURUSD/GBPUSD) strategy with no clean third
pair, so its trades are tagged by the **EURUSD** regime and its gate is **IS→OOS only** (no separate
cross-pair gate). The four single-symbol strategies (ma/trend/meanrev/orb) run on EURUSD and GBPUSD
independently and use the full three-gate gauntlet.

## 4. Validation gauntlet + discipline

A strategy×regime cell survives only if **all** hold, in order:
1. **IS:** conditional net profit factor > 1 after cost (and a meaningful trade count).
2. **OOS replication:** the SAME cell (same strategy, params, regime label from the frozen model) is
   net-positive on 2025-06-01→2026-05-01. (This is the first OOS test of the *regime* hypothesis —
   legitimate single look.)
3. **Cross-pair:** the same cell is net-positive on GBPUSD (single-symbol strategies only; xpair
   exempt per §3).

- **Search space:** 5 strategies × 3 regimes ≈ 15 cells/pair — report it. The decisive gate is the
  OOS + cross-pair replication, immune to IS cell-picking.
- **Thin-cell risk is the main hazard.** Splitting each strategy's trades across 3 regimes fragments an
  already-small sample. **Report n per cell PROMINENTLY**; flag any candidate resting on a thin cell
  (e.g. n < ~30) as low-power. 30m was chosen precisely to keep cells populated (the five originals
  take hundreds of trades at 30m vs ~70 at 4h).

## 5. Promotion (separate cycle, not in this study)

A cell clearing gates 1–3 → a separate feature cycle (brainstorm→spec→plan→TDD) builds a
**regime-gated strategy** in `src/` (the strategy computes the live regime and trades only in the
surviving regime) for the definitive engine backtest under the true cost model. Do NOT build it here.

## 6. Windows, IS/OOS, costs

- IS 2023-06-01→2025-06-01 (regime fit + IS screen). OOS 2025-06-01→2026-05-01 (touched once, gate 2).
- Real IB-retail cost model (already in `SimulatedBroker` via `run_backtest`); conditional P&L is
  cost-inclusive by construction.

## 7. Deliverable

- `research/findings-2026-06-26-regime-gating.md` (if harness blocks the subagent write, return full
  text; controller saves it — as in Phase B / shape discovery).
- JSON under `research/json/` (e.g. `regime_*.json`): regime model params/centroids, per-(strategy ×
  pair × regime × window) net PF/Sharpe/win/n, the gauntlet table, search-space count.
- Honest verdict: did any strategy×regime cell clear all gates net-of-cost? If yes, name it (strategy,
  regime character, IS/OOS/GBP numbers, n) and recommend promotion. If no (likely), state the clean
  negative and whether anything came close. Flag anything resting on thin n.

## Non-goals / guardrails

- **No `src/` edits.** No regime-gated strategy is built here (that's promotion).
- **No param re-optimization** — strategies fixed at documented finalists (regime is the only new
  dimension searched).
- **No K sweep** beyond K=3, no extra features beyond vol × trend-strength (every added degree of
  freedom inflates the false-positive rate). OOS touched once.
