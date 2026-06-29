# fx_trader — Project Guide (CLAUDE.md)

Tick-driven FX algo-trading **research platform**. Built in phases P0–P4 (data → backtester →
research agent → live-on-IB), then used to test strategies. **Status: feature-complete and
consolidated. The headline research result is a thorough, rigorous NEGATIVE — see below.**

## TL;DR for a new session

- **Six strategies built, reviewed, merged. ALL net-negative after costs.** Plus a timeframe sweep
  and a cost-sensitivity probe — every angle exhausted. The edge isn't there; it's a *no-signal*
  problem, not a cost problem (proven — see Research Conclusion).
- **Do NOT re-run the exhausted experiments** (single-signal strategies, 30m/60m/120m/240m frequency
  tuning, native-4h re-optimization, cost ladders). They're done and documented. New work should be a
  *different problem* (new instruments / sub-bar microstructure / different asset class) or the
  deferred **consolidation cycle 2** (CLI param-plumbing generalization).
- Full detail lives in `research/findings-*.md` (7 reports) and the persistent memory
  `fx-trader-roadmap.md`.

## Environment & commands

- **Python pinned to 3.12 in `.venv`** (system is 3.14, lacks quant wheels). ALWAYS invoke tooling as
  `.venv/bin/python` / `.venv/bin/pytest` / `.venv/bin/fxtrader` — shell `source` does not persist
  across tool calls.
- Test suite: `.venv/bin/pytest -q` → **249 passed, 2 skipped** (baseline; keep green).
- CLI: `fxtrader download | backfill | backtest | optimize | live`.
  - `backtest --strategy {ma,trend,meanrev,orb,xpair,pa} ...` (xpair needs `--symbol2`).
  - `optimize --strategy ... --json <path>` (strict-JSON walk-forward output).
  - `live` is **mode-gated** (DRY_RUN default → PAPER → LIVE behind `--i-understand-live`); needs the
    USER to run IB Gateway + a paper account. Never auto-deploy.

## Architecture keystone

One `Strategy` interface, **two engines** — the same strategy code runs in the backtester
(`SimulatedBroker`) and live (`IBBroker`), with a shared `RiskManager` (position/gross/rate limits +
daily-loss kill switch). Engine is **multi-symbol** (per-symbol `Resampler` dict, symbol-keyed
`SimulatedBroker._market`/`Portfolio`). Strategies are bar-based (`on_bar`), single-symbol except
`xpair` (two legs). Spec: `docs/superpowers/specs/2026-06-15-fx-trader-platform-design.md`.

## The six strategies (`src/fxtrader/strategy/examples/`, wired via `strategy/registry.py`)

| name | file | idea | verdict |
|---|---|---|---|
| `ma` | `ma_crossover.py` | naive MA crossover (reference) | no edge |
| `trend` | `trend_filtered_crossover.py` | MA cross gated by long-trend MA + `ExitPolicy` stop/TP | no edge |
| `meanrev` | `mean_reversion_fader.py` | z-score fade gated by efficiency-ratio (ranging) | no edge; **stop/TP overlay is a DEAD PATH** (z-exit pre-empts it) |
| `orb` | `opening_range_breakout.py` | Asian-range breakout at London open + time exit | no edge (closest to break-even gross) |
| `xpair` | `pairs_spread_reversion.py` | market-neutral fade of z(log EURUSD − log GBPUSD) | no edge |
| `pa` | `price_action.py` | trend-gated price-action patterns (pin/engulf/inside/swing-break) + MAE-calibrated ATR exit | no edge — gross PF ≈ coin-flip, failed cross-pair/cross-TF (Phase B 2026-06-25) |

Shared: `strategy/exits.py` (`NoExit`/`FixedPips`/`AtrMultiple` — bar-level stop/TP, stop-first
same-bar rule), `strategy/indicators.py` (`zscore`, `efficiency_ratio`), `metrics/excursions.py`
(MAE/MFE diagnostic).

## Research Conclusion (the durable result)

**Simple, retail-accessible price/time/relative-value/price-action signals have NO meaningful edge on
30m–8h EUR/GBP majors over 2023-06→2026-05.** Established across:
- **6 structurally-distinct strategies** (trend ×2, mean-reversion, session breakout, cross-pair
  stat-arb, trend-gated price-action) — all net-negative after costs, all degrade in-sample → out-of-sample.
- **Timeframe sweep** 30/60/120/240m — in-sample positives at coarse bars were single-lucky-fold
  noise that died OOS and failed cross-pair (`findings-2026-06-22-timeframe.md`).
- **Native 4h re-optimization** (re-tuned params, not reused) — improved numbers but nothing crossed
  break-even (`findings-2026-06-23-4h.md`).
- **Cost-sensitivity probe** (the decisive closer): even at **ZERO cost**, the best configs make
  ~+0.1%/yr (trend) and ~+0.03%/yr (xpair) at Sharpe 0.14–0.23 — noise. **Not a cost problem, a
  no-signal problem.** Cheaper execution / larger size flips the sign but yields no usable strategy.
- **Trend-gated price-action** (pin/engulfing/inside-bar/swing-break × 4h/8h, `findings-2026-06-25-pa.md`):
  gross PF a coin-flip even before commission; only 2 of 8 pattern×TF combos cleared both pairs in-sample
  — exactly the ~2 pure noise yields across 8 combos — and neither held across timeframe. Killed at the
  cross-pair screen; frozen OOS left unspent (a real finalist never appeared).
- **Data-DERIVED shapes** (the only non-hypothesis-first attempt: unsupervised z-norm-window k-means
  clustering on 4h/8h, `findings-2026-06-25-shape-discovery.md`): letting the data pick the shapes finds
  nothing either — of 80 well-populated shape-clusters, **0 reach even uncorrected p<0.05 in-sample** and
  corr(IS mean, OOS mean) ≈ **−0.09**. The IS→OOS shape→return map simply doesn't generalize. Sharpest
  no-signal evidence yet; intersection of IS-net>0 ∩ OOS-replicate ∩ cross-pair = 0.
- **Regime-gating** (tests *timing*, not raw signal: K=3 unsupervised vol×trend-strength regimes,
  strategies' trades tagged by entry-regime, 30m, `findings-2026-06-26-regime-gating.md`): the "signal
  isn't dead, it's untimed" hypothesis is REJECTED. Of 15 strategy×regime cells, 0 cleared IS→OOS→cross-pair;
  the closest (orb×trending: IS PF 1.02 → EURUSD-OOS +118 → **GBPUSD-OOS −311**) is the textbook
  false-positive the gauntlet rejects. meanrev loses even in its target choppy regime; xpair's 2-leg cost
  dominates every regime. No cell is net-positive OOS on both pairs (intersection = 0).
- **New instrument — gold (XAUUSD) + AUDUSD** (the one genuinely-different fork: a less-arbitraged asset
  class; `pa` at 1h/4h, `findings-2026-06-28-new-instrument-pa.md`): also clean-negative. Gold 4h
  *looked* alive (engulf/pin/swingbreak IS net PF 1.28-1.30, and OOS even stronger 1.44-1.78) — the
  project's first IS→OOS replication — but a **pattern-free MA-crossover control reproduces the same
  gold-yes/AUDUSD-no signature** (gold MA +1.7% OOS, AUD MA −0.65% OOS): it is **trend beta** off gold's
  +68%/+43% IS/OOS bull run, not price-action. The lone IS finalist (engulf@4h) **collapsed on AUDUSD
  OOS** (net PF 0.519); intersection of IS-net>0 ∩ OOS-net>0 ∩ both-instruments = **0**. Also note gold
  IS and OOS are the *same continuous bull market* → not independent confirmations. (This study also
  forced a real platform fix: the tick `backfill`/`store` path OOM'd + aborted on gold's ~3×-denser
  ticks; hardened with per-hour round-retry + frame-accumulation + streaming manifest — see git.)

The recurring signature: gross profit factor hovers at ~1.0 (coin-flip before costs), commission
lives just above it. The platform's value is that it produced this **fast, honest, conclusive
negative** and caught every false positive (240m flips, inert brackets, a swallowed-error bug)
before they could mislead.

## Data & cost model

- Real Dukascopy ticks: **EURUSD, GBPUSD, USDJPY**, monthly 2023-06 → 2026-05, partitioned Parquet
  under `data/` (**gitignored**, ~1.5 GB). `coverage.json`/`manifest.json` track it.
- **IS/OOS convention:** in-sample 2023-06-01→2025-06-01 (all optimization); frozen OOS
  2025-06-01→2026-05-01, touched once for finalists.
- **Cost model** (`SimulatedBroker`): fills at real bid/ask (spread cost) + commission
  `max($2, 0.2bps × notional)` per fill = **the Interactive Brokers retail FX schedule**. At the 10k
  trade size used, the $2 min binds → ~4 pips round-trip single-leg / ~8 pips two-leg (worst-case;
  ~10× larger size drops the pip-hurdle ~10× as 0.2bps binds instead).

## Key gotchas (learned the hard way)

- **Feed streaming (FIXED 2026-06-23):** `HistoricalTickFeed.__iter__` now streams **month-by-month**
  (was `pl.concat`-ing the whole window → OOM'd every multi-year/two-symbol run). Peak RSS now ~1
  month (~0.96 GiB for a 2yr run, was 5.7+ GiB). `read_ticks` is partition-aware. Runtime is still
  **tick-bound** (~7 min for a 2yr single-symbol pass) — keep research windows to the IS-year and
  grids tiny.
- **Crossover seed-data gotcha (hit 4×):** MA-cross strategies fire only on an actual *cross* — test
  close series must dip-then-rise across MULTIPLE bars, or 0 trades.
- **USDJPY single-currency sizing:** Portfolio is single-currency; a fixed 10k position on a ~150
  pair ≈ ~15× leverage → can blow through zero (flagged via `ruined`). Validate on **USD-quoted pairs
  (EURUSD/GBPUSD)**; bound USDJPY with `--max-position` or defer until notional accounting exists.
- **`ib_async` live path:** commissions arrive on `commissionReportEvent` (not execDetails); IB sends
  bid/ask=`-1` for no-data; Ticker sizes default to NaN. Run `pip install -e ".[live]"`.

## How work is done here (process)

- **Features:** brainstorm → spec (`docs/superpowers/specs/`) → plan (`docs/superpowers/plans/`) →
  subagent-driven TDD (implementer + reviewer per task) → final whole-branch review → merge to
  `master` (`--no-ff`). Each strategy followed this; commits trailer
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Research:** dispatched to a background **quant-researcher** agent with a HARD guardrail —
  **never edit `src/`** during research (get single-backtest metrics via a `research/` script calling
  `run_backtest`; WFO via `optimize --json`). The controller re-verifies headline numbers against the
  JSON before trusting them. (A prior agent once wired `--json` into a strategy CLI branch — reverted;
  `--json`/`--html` are MA-only by design.)
- `research/` (scripts, `findings-*.md`, `json/`) is **untracked** working data.

## Deferred / open (for whenever work resumes)

- **Consolidation cycle 2:** generalize the accreted CLI param-plumbing — 5 strategies each added
  their own flags / grid-branches / `run_backtest` kwargs; replace with a generic per-strategy param
  mechanism. Substantial; touches all 6 byte-identical paths → its own brainstorm→spec→plan cycle.
- **Notional / multi-currency Portfolio accounting** (unblocks USDJPY + cross-pairs honestly).
- **DST-aware ORB session hours** (currently fixed UTC, ~1h seasonal drift).
- **Rolling-β for xpair** (currently β=1 / synthetic EURGBP).
- **A genuinely different regime** (more instruments, lower-cost execution, sub-bar microstructure) —
  a new project, not yet another single-signal variant.
