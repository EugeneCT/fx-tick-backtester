# Data-driven price-action shape discovery — Design / research protocol

- **Date:** 2026-06-25
- **Type:** Research study (NOT a `src/` feature). Executed by the `quant-researcher` agent under the
  hard guardrail (never edits `src/`; works in `research/`). Controller re-verifies headline numbers
  against JSON. A survivor — if any — triggers a *separate* feature cycle to build a strategy.
- **Status:** Approved; pending dispatch.

## Motivation

Six strategies — all hypothesis-first (impose a known idea, test it) — are net-negative on EUR/GBP
majors (see `CLAUDE.md` Research Conclusion, `research/findings-*.md`). This study inverts the
approach: **let the data derive the price-action shapes**, instead of testing textbook patterns
(pin/engulf/inside/swing-break were just exhausted, `findings-2026-06-25-pa.md`). Unsupervised
clustering of bar-window shapes → test which discovered shapes precede non-random, cost-clearing
forward moves. This is the "richer signal construction" fork `CLAUDE.md` flags as legitimately
unexplored.

**Overfitting is the central hazard.** Mining a dataset for shapes is how false edges are
manufactured, and coarse bars + clustering is the worst case (few occurrences per cluster, many
clusters tested). The validation gauntlet (§4) is the core of this design, not an afterthought. The
expected outcome, given six prior negatives, is another clean negative — a valid result.

## Scope (decided)

- **Method:** unsupervised shape discovery.
- **Encoding:** z-normalized close-path window + k-means.
- **Timeframes:** 4h (`bar_minutes=240`) and 8h (`bar_minutes=480`). (User choice; coarse — small-sample
  risk is real and must be reported per-cluster.)
- **Discovery pair:** EURUSD. **Cross-pair validation:** GBPUSD. (USDJPY out — returns-study pip scale
  differs; not needed.)
- **Windows:** IS 2023-06-01→2025-06-01 (all discovery + calibration). OOS 2025-06-01→2026-05-01 —
  touched once, for the replication gate only.

## 1. Discovery engine (`research/`, no `src/`)

- Read EURUSD bars via `TickStore("data")` + `Resampler(timedelta(minutes=bar_minutes))` over IS
  (the Phase-B `run_pa_stage1.py` shows the exact resample loop). Build the ordered bar list.
- Slide a window of **W bars** ending at (and including) the decision bar `t` (sweep **W ∈ {3, 5}**).
  Take the **close path** of those W bars and **z-normalize** it: `(x − mean) / std` (skip windows with
  `std == 0`). This strips price level and scale → clusters capture SHAPE only.
- Cluster the z-normed windows with **k-means, K = 10** prototypes per (W, timeframe). Use a FIXED
  random seed (e.g. `random_state=0`) for reproducibility. Prefer `sklearn.cluster.KMeans`; if sklearn
  is unavailable in `.venv`, a plain-numpy Lloyd's k-means with a fixed seed is fine. Save each
  centroid (the discovered shape) so it can be plotted/inspected.
- **No look-ahead:** the window uses bars `t−W+1 … t` (all known at `t`'s close); the label (below) is
  realized strictly after `t`.

## 2. Edge definition (cost-aware from the start)

- **Forward return** for window ending at `t`: `close[t+H] − close[t]` in **pips** (via
  `fxtrader.core.instruments.pip_size`), for **H ∈ {1, 3}** bars. Record sign and magnitude.
- **Cost hurdle:** the real IB-retail model at 10k size → ~**4 pips round-trip single-leg** (the $2/fill
  min binds; spread on top). A cluster is a *candidate* only if its mean forward move is (a)
  statistically ≠ 0 AND (b) `|mean| comfortably > ~4 pips` (i.e. a conditional trade — enter at `t`
  close, exit at `t+H` close — has positive **net-of-cost** expectancy). Also compute the
  cluster-conditional **net P&L series** (per-occurrence forward return minus the ~4-pip cost) and its
  hit-rate and a simple Sharpe, so the screen is net, not gross.

## 3. Search-space accounting

Total cluster-hypotheses = Σ over (W ∈ {3,5}) × (H ∈ {1,3}) × (tf ∈ {4h,8h}) of K=10 ≈ **80 clusters**
(direction is the sign of the mean, not a separate hypothesis). Report this count explicitly; it is
the denominator for the multiple-comparison control.

## 4. Validation gauntlet — a shape must pass ALL of these, in order

1. **IS significance + multiple-comparison control.** Per candidate cluster, a two-sided t-test (or
   sign test) that mean forward return ≠ 0. Apply **Benjamini-Hochberg FDR at q = 0.10** across all ~80
   cluster-hypotheses. Surviving IS significance is necessary, not sufficient.
2. **OOS replication (decisive).** FREEZE the IS k-means centroids. Assign each OOS window to its
   nearest IS centroid. A cluster passes only if its OOS forward edge is **same-sign** as IS AND still
   **cost-clearing** (OOS net-of-cost mean > 0). This gate is structurally immune to IS data-mining —
   noise clusters do not replicate out-of-sample.
3. **Cross-pair.** Assign **GBPUSD** windows (IS+OOS) to the **EURUSD** IS centroids (z-norm makes
   shapes pair-agnostic, so EURUSD prototypes apply directly). The shape must show the same-sign,
   cost-clearing edge on GBPUSD. Pair-specific = artifact (this killed `pin`).
4. **Real-engine confirmation (promotion, separate cycle).** Within THIS research study the cost-aware
   measure is the pip-hurdle net simulation (§2). The definitive engine confirmation needs real
   bid/ask fills via `run_backtest`, which requires a strategy class — so a survivor of gates 1–3
   triggers a SEPARATE feature cycle (brainstorm→spec→plan→TDD) to build the cluster rule as a strategy
   and backtest it under the true cost model. Do NOT build it inside this study.

## 5. Per-cluster reporting (mandatory)

For every cluster, report **n (occurrences)** prominently — coarse bars give thin clusters (4h IS
~320/cluster at W=5, K=10; 8h OOS can be ~80/cluster). Any candidate resting on a thin cell must be
flagged as low-power. Report IS mean pips + t-stat + FDR pass/fail, OOS mean pips + sign-match, GBPUSD
mean pips + sign-match, and the net-of-cost hit-rate/Sharpe. Plot or tabulate the centroid shapes.

## 6. Deliverable

- `research/findings-2026-06-25-shape-discovery.md` (if the harness blocks subagent file writes, the
  agent returns the full report text and the controller saves it, as with Phase B).
- Raw results as JSON under `research/json/` (e.g. `shape_*.json`): centroids, per-cluster IS/OOS/xpair
  stats, the FDR table, the search-space count.
- Honest verdict: did ANY shape clear all of gates 1–3 net-of-cost? If yes, name it, show its centroid
  and its IS/OOS/GBPUSD numbers, and recommend promotion. If no (likely), state the clean negative and
  whether anything came close. A negative is fully valid.

## Non-goals / guardrails

- **No `src/` edits** during this study (hard guardrail). No strategy is built here.
- No OHLC-channel / volume features in the main study — close-path shape only (keeps the search space
  bounded; an OHLC-aware variant is a possible follow-up only if a close-path survivor appears).
- OOS is touched ONCE, for gate 2 only. No iterating against OOS.
- No sweeping K / W / H beyond the tiny declared grid (every added degree of freedom inflates the
  search space and the false-positive rate).
