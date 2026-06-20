# Trend-Filtered Crossover + MAE-Driven Exits — Design Spec

**Date:** 2026-06-19
**Status:** Approved (design), pending implementation plan
**Author:** Eugene + Claude

## 1. Goal

Add the platform's second strategy: a **trend-filtered MA crossover** with data-driven
stop-loss / take-profit exits. The first research pass (`research/findings-2026-06-18.md`)
showed the naive `MaCrossover` has no edge after costs — it whipsaws in ranges and bleeds
commission. This strategy attacks that directly by (a) only trading crosses that agree with a
longer-term trend and going flat otherwise, and (b) adding exits whose levels are derived from
Maximum Adverse/Favorable Excursion (MAE/MFE) analysis and validated out-of-sample.

**Success bar (agreed):** a config with **positive out-of-sample Sharpe AND profit factor > 1
after commissions**, that holds up **across walk-forward folds** (robust, not one lucky window).
Even a small-but-robust edge counts.

## 2. Scope & decisions (locked)

- **Strategy type:** single-symbol, bar-based (same shape as `MaCrossover`), reuses the existing
  engine with **no engine or live-path changes**.
- **Signal:** MA cross gated by a long-term trend MA; **flat** when cross and trend disagree;
  **auto-flatten** when trend flips against an open position.
- **Exits:** synthetic, bar-granular, implemented inside the strategy as market orders (no broker
  stop/limit orders — those remain deferred in the engine). Two interchangeable parametrizations,
  **both** built and chosen by OOS performance: fixed-pip and ATR-scaled.
- **Level derivation:** new MAE/MFE diagnostic tooling; levels derived **in-sample only**, then
  walk-forward validated.
- **Validation:** EURUSD first (USD-quoted → single-currency Portfolio is correct); optionally
  confirm GBPUSD; **USDJPY deferred** until notional sizing is fixed.
- **IS/OOS split:** in-sample 2023-06-01 → 2025-06-01 (all optimization + MAE derivation); frozen
  out-of-sample 2025-06-01 → 2026-05-01, touched once for finalists.

### Out of scope (for now — YAGNI)

- Trailing stops (distances freeze at entry).
- Tick-level exit fills (exits are detected at bar granularity via `bar.high`/`bar.low`).
- Live wiring of the new strategy (`fxtrader live` keeps defaulting to `ma`).
- USDJPY / cross-pair validation; the `HistoricalTickFeed` performance refactor.
- A min-MA-separation filter; joint signal×exit optimization (we stage instead).

## 3. Strategy logic — `TrendFilteredCrossover`

Three MA windows `fast < slow < trend` (validated in `__init__`, raises `ValueError`), plus
`size` and an `ExitPolicy`. Bar-based, single symbol; ignores bars for other symbols.

State: `_bias ∈ {-1, 0, +1}` = direction implied by the most recent MA cross; `_entry_price`.

Per `on_bar`, once `len(closes) ≥ trend`:

```
fast_ma, slow_ma, trend_ma = means over last fast / slow / trend closes
trend_dir = +1 if close > trend_ma else (-1 if close < trend_ma else 0)

# 1. exit check FIRST (against open position), using the bar's range
if long  and (bar.low  <= entry - stop_dist or bar.high >= entry + tp_dist): flatten; _bias = 0
if short and (bar.high >= entry + stop_dist or bar.low  <= entry - tp_dist): flatten; _bias = 0
#   conservative same-bar rule: if both stop and TP are touched in one bar, assume STOP first.

# 2. update bias on a fresh cross (same detection as MaCrossover)
if crossed_up:   _bias = +1
if crossed_down: _bias = -1

# 3. gate: hold only when cross and trend agree, else flat
desired = _bias * size  if (_bias != 0 and _bias == trend_dir) else 0
delta = desired - current_position
if delta != 0: order(BUY/SELL, abs(delta))  # records _entry_price on a new entry
```

Behavior vs. naive crossover: bullish cross **below** the trend MA → flat (skip the whipsaw);
holding long while trend flips down → flatten; stop/TP hit → flatten and disarm until the next
cross. After a stop/TP exit, `_bias = 0` prevents immediate re-entry on a stale signal.

## 4. Exit policy — `strategy/exits.py`

`ExitPolicy` protocol keeps exits as an isolated, independently-testable unit:

- `update(bar) -> None` — feed each bar (for stateful policies like ATR).
- `distances() -> tuple[float | None, float | None]` — `(stop_dist, tp_dist)` in **price** terms;
  `None` disables that leg.

Implementations:

- `NoExit` — returns `(None, None)`. Used for the Phase-1 measurement run and to reproduce the
  pure trend-gate strategy.
- `FixedPips(stop_pips, tp_pips, pip_size)` — constant distances `pips * pip_size`.
- `AtrMultiple(period, stop_mult, tp_mult)` — maintains ATR over OHLC
  (`TR = max(high-low, |high-prev_close|, |low-prev_close|)`, simple mean over `period`); distances
  = `mult * ATR`, **frozen at entry** (no trailing in v1).

Validation: negative pips/mults/period → `ValueError`.

## 5. MAE/MFE diagnostic — `metrics/excursions.py`

Pure post-processing, reuses the fill-pairing logic already in `metrics/performance.trade_pnls`
(a position opens when qty leaves 0, closes when it returns to / crosses 0).

- `trade_excursions(fills, bars, pip_size) -> list[TradeRecord]` where `TradeRecord` =
  `(entry_ts, exit_ts, direction, pnl, mae_pips, mfe_pips)`. For each trade, walk the bars in
  `[entry_ts, exit_ts]`: for a long, `mae = max(entry - bar.low)`, `mfe = max(bar.high - entry)`
  (mirrored for short), converted to pips via `pip_size`.
- `summarize_excursions(records) -> dict` — winner-vs-loser MAE/MFE percentiles (e.g. p50/p75/p90).
  This is the diagnostic that seeds candidate stop/TP levels: winners typically show small MAE;
  the stop goes just beyond where winners normally sit, the TP near where MFE plateaus.

`pip_size(symbol)` helper in `core` (0.01 for `*JPY`, else 0.0001).

## 6. CLI wiring — `strategy/registry.py` + `cli.py`

A small registry makes `--strategy` generic so the next strategy is a one-line add:

```
STRATEGIES = {
  "ma":    spec(cls=MaCrossover,             params=("fast", "slow")),
  "trend": spec(cls=TrendFilteredCrossover,  params=("fast", "slow", "trend")),
}
```

- `backtest` / `optimize` gain `--strategy {ma,trend}` (default `ma`, preserving current behavior),
  `--trend` (int / comma-separated grid), and exit options `--exit {none,fixed,atr}` with
  `--stop-pips/--tp-pips` (fixed) and `--atr-period/--stop-atr/--tp-atr` (atr).
- The backtest builder and the optimize `run_fn` are refactored to be strategy-generic: build the
  strategy + `ExitPolicy` from the selected spec and only its relevant params; the optimize param
  grid is built from the spec's param names plus the chosen exit params.
- Unknown `--strategy` / `--exit` → `typer.BadParameter` (consistent with the existing `--mode`
  handling).

## 7. Research process (staged optimization)

Staging keeps grids small (the ~22–37 min/run feed bottleneck makes joint grids impractical) and
shrinks the overfit surface. **Assumes signal and exit params are roughly separable** — a
deliberate simplification.

1. **Signal:** walk-forward `fast/slow/trend` with `NoExit` on in-sample EURUSD → pick robust params.
2. **Diagnose:** run the MAE/MFE diagnostic on those trades → read winner/loser percentiles → derive
   a *small* candidate set of fixed-pip levels **and** ATR multipliers.
3. **Exits:** walk-forward the exit candidates (fixed + ATR) on top of the chosen signal → pick the
   best family by OOS.
4. **Confirm:** run the single finalist once on the frozen OOS window; report IS→OOS honestly. Update
   `research/findings-2026-06-19.md`.

Coarse bars (15–30m) for sweeps; EURUSD primary, GBPUSD optional confirm.

## 8. Repo layout (new / changed)

```
src/fxtrader/
  strategy/
    examples/trend_filtered_crossover.py   NEW  TrendFilteredCrossover
    exits.py                               NEW  ExitPolicy + NoExit/FixedPips/AtrMultiple
    registry.py                            NEW  strategy name -> spec
  metrics/excursions.py                    NEW  trade_excursions + summarize_excursions
  core/ (pip_size helper)                  EDIT small instrument helper
  cli.py                                   EDIT --strategy + exit options on backtest/optimize
tests/
  strategy/test_trend_filtered_crossover.py NEW
  strategy/test_exits.py                    NEW
  metrics/test_excursions.py                NEW
  (cli tests)                               EDIT --strategy trend end-to-end
```

## 9. Testing strategy (TDD)

- **Exits:** `FixedPips` returns constant distances; `AtrMultiple` computes ATR correctly and scales;
  `NoExit` returns `(None, None)`; touch detection with crafted bars (long stop on `low`, TP on
  `high`; same-bar both → stop wins); `ValueError` on bad params.
- **Strategy:** the trend-gate truth table (bullish/bearish cross × up/down trend → long/short/flat);
  auto-flatten on trend flip with no new cross; bias reset on stop/TP exit and no re-entry until a new
  cross; warmup (no trades before `trend` closes); `ValueError` on `fast≥slow` or `slow≥trend`;
  ignores other symbols. **Gotcha (seen 3× already):** crossover seed data must dip-then-rise across
  *multiple* bars or it fires zero trades.
- **Excursions:** known fills + bar path → expected MAE/MFE pips; winner/loser summary percentiles.
- **Registry/CLI:** registry builds the right strategy + params; unknown name errors; `--strategy
  trend` with each exit mode runs `backtest` + `optimize` on a small fixture.
- Full suite stays green (currently 140 passed, 2 skipped).

## 10. Assumptions & risks

- **Bar-granular exits** approximate intrabar fills; the same-bar stop-first rule keeps this
  pessimistic, not optimistic. Acceptable for research; revisit if a finalist looks exit-sensitive.
- **Signal/exit separability** (staged optimize) may miss interaction effects; accepted to stay
  tractable and reduce the overfitting surface.
- **MAE-derived fixed levels overfit** if not validated OOS — mitigated by deriving in-sample only and
  walk-forward validating.
- **Trend-following may simply have no edge** on EURUSD at these timeframes; if so, going flat in chop
  still reduces losses toward zero, and the plumbing built here makes the mean-reversion strategy (B)
  a small next step.
- **Feed performance** (~22–37 min/2yr run, OOM on concurrent runs) constrains breadth; mitigated by
  coarse bars, small staged grids, and serial runs. The feed refactor remains a separate concern.

## 11. Definition of done

- `TrendFilteredCrossover`, `ExitPolicy` (3 impls), `trade_excursions`/`summarize_excursions`,
  `pip_size`, and `--strategy`/exit CLI options implemented and unit-tested; full suite green.
- `--strategy trend` runs end-to-end through `backtest` and `optimize`.
- Staged research process executed on EURUSD; `research/findings-2026-06-19.md` records the
  signal params, MAE/MFE diagnostic, chosen exit family, and the one-shot OOS result — including an
  honest "no edge survived" verdict if that's the outcome.
- Nothing committed/pushed without request; live command never run; no engine/live-path changes.
