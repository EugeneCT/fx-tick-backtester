# Cross-Pair Spread Reversion (EURUSD/GBPUSD stat-arb) — Design Spec

**Date:** 2026-06-22
**Status:** Approved (design), pending implementation plan
**Author:** Eugene + Claude

## 1. Goal

Add the platform's fifth strategy and its **first relative-value (non-directional, multi-symbol)
signal**: a market-neutral pairs trade on EURUSD vs GBPUSD. Four directional single-pair strategies
(trend ×2, mean-reversion, session breakout) plus a frequency sweep are all net-negative after costs
(`research/findings-2026-06-22-timeframe.md`) — directional prediction of a major has ~no gross edge.
A pairs trade is structurally different: long one leg / short the other cancels the common USD move
and bets **only on the EUR-vs-GBP spread mean-reverting** — a different edge source (spread reversion
/ stat-arb) no single-pair signal can capture. Because `log(EURUSD) − log(GBPUSD) = log(EURGBP)`, the
spread is the synthetic EUR/GBP cross, traded as two USD-quoted legs (which the single-currency
Portfolio handles correctly).

**Success bar (carried):** positive out-of-sample Sharpe AND profit factor > 1 after commissions,
robust across walk-forward folds. (Cross-*pair* confirmation is N/A here — see §2 caveat.)

## 2. Scope & decisions (locked)

- **Strategy type:** **two-symbol** bar-based — `primary` (EURUSD, the "clock") and `secondary`
  (GBPUSD). The engine is already multi-symbol (per-symbol `Resampler` dict); **no engine or
  live-path changes**.
- **Spread:** `s = log(primary.close) − log(secondary.close)`, **equal-weight (β=1)** — the synthetic
  EUR/GBP cross. Sampled once per period on the **primary's bar** using the latest secondary close
  (≤1-period stale worst case — acceptable at 30m). z-score of `s` over `lookback` (reuses the
  `zscore` indicator).
- **Entry (flat only):** `z > entry_z` (primary rich) → **SELL primary + BUY secondary** (short the
  spread); `z < −entry_z` → **BUY primary + SELL secondary** (long the spread). Each leg `size` units
  (equal notional, β=1).
- **Exit, risk-first:** signal exit on reversion (`|z| ≤ exit_z`, close both legs) OR **divergence
  stop** (`|z| ≥ stop_z`, the spread widened against us — close both legs). Either closes both legs and
  `return`s (no same-bar re-entry). **No `ExitPolicy`** (price stop/TP doesn't fit a spread; the
  z-based divergence stop is the risk control).
- **One pair at a time**, no flips; `std==0` spread window → skip.
- **Reuse:** `zscore` indicator, the registry/CLI `--strategy` seam, the `--lookback/--entry-z/--exit-z`
  options (from meanrev). NOT the `ExitPolicy`.
- **Validation:** EURUSD/GBPUSD (the strategy *is* this pair — the one well-correlated USD-quoted pair
  in the data; USDJPY isn't correlated with EUR/GBP). Same IS/OOS split (IS 2023-06-01→2025-06-01,
  frozen OOS 2025-06-01→2026-05-01), 30m bars.

### Out of scope (for now — YAGNI)

- **Cross-pair confirmation** — there is no independent second correlated USD-quoted pair in the data,
  so robustness rests on **walk-forward + frozen OOS alone**. Stated honestly in the findings; not a
  code gap.
- Rolling-regression hedge ratio (β=1 in v1), a max-hold/time exit, an `ExitPolicy` overlay, the
  bid/ask-aware leg balancing, USDJPY, generalizing the CLI param plumbing (accrete `xpair` flags),
  live wiring of the new strategy.

## 3. Strategy logic — `PairsSpreadReversion`

Params: `primary: str`, `secondary: str`, `lookback: int`, `entry_z`, `exit_z`, `stop_z` (floats),
`size: float`. `__init__` validates `lookback ≥ 2` and `0 ≤ exit_z < entry_z < stop_z` → else
`ValueError`. Ignores bars for symbols other than primary/secondary.

State: `_last: dict[str,float]` (latest close per leg), `_spreads: deque(maxlen=lookback)`, `_pos_dir`
(spread direction held: +1 long-spread / −1 short-spread / 0), `_last_exit_reason`.

```
on_bar(bar):
  if bar.symbol == secondary: self._last[secondary] = bar.close; return
  if bar.symbol != primary: return
  self._last[primary] = bar.close
  if secondary not in self._last: return
  import math; s = math.log(self._last[primary]) - math.log(self._last[secondary])
  self._spreads.append(s)
  if len(self._spreads) < lookback: return
  if pstdev(self._spreads) == 0: return
  z = zscore(list(self._spreads))

  pos = self.portfolio.position(primary).quantity   # spread leg state
  # 1. EXIT FIRST
  if pos != 0:
    if abs(z) <= exit_z: self._close("signal"); return
    if abs(z) >= stop_z: self._close("stop"); return
  # 2. ENTRY (flat only)
  if pos == 0:
    if z > entry_z:    self._open(-1)   # short spread: SELL primary + BUY secondary
    elif z < -entry_z: self._open(+1)   # long spread:  BUY primary + SELL secondary
```

`_open(direction)`: submits the two leg orders (SELL primary/BUY secondary for −1; mirror for +1),
each `size` units, sets `_pos_dir`. `_close(reason)`: reads each leg's signed position and submits the
offsetting order per leg (flatten both), sets `_last_exit_reason`, resets `_pos_dir`.

## 4. Multi-symbol wiring

The engine already keeps a per-symbol `Resampler` and dispatches each symbol's bars; the feed merges
symbols by time; `Portfolio.positions` is keyed by symbol and cash is USD for both legs. The wiring:

- **`run_backtest(...)`** gains `symbol2: str | None = None`; builds the feed with `[symbol]` or
  `[symbol, symbol2]`; adds a `stop_z: float = 0.0` kwarg (`lookback/entry_z/exit_z` already exist from
  meanrev); the `make_strategy` call passes `symbol2` + the params dict gains `stop_z`.
- **`make_strategy(...)`** gains `symbol2: str | None = None` and an `xpair` branch:
  `PairsSpreadReversion(primary=symbol, secondary=symbol2, lookback=…, entry_z=…, exit_z=…, stop_z=…,
  size=…)`. `STRATEGY_PARAMS["xpair"] = ("lookback","entry_z","exit_z","stop_z")`.
- **`backtest` command:** `--symbol2` (str) + `--stop-z` (float); accept `strategy=="xpair"`; an
  `xpair` branch routing through `run_backtest(strategy_name="xpair", symbol2=…, stop_z=…, exit_kind=
  "none", …)` and printing a summary naming both legs.
- **`optimize`/`run_optimize`/`_strategy_run_fn`:** thread `symbol2` + `stop_z_values`; an `xpair`
  grid `{lookback, entry_z, exit_z, stop_z}` (`_int_list` for lookback, `_float_list` for the z's); an
  `xpair` validation branch (`lookback≥2 and 0≤exit_z<entry_z<stop_z`) forwarding `symbol2`+`stop_z` to
  `run_backtest`. ma/trend/meanrev/orb grids + validation unchanged.

The `SimulatedBroker` fills per-symbol from each symbol's latest tick (symbol-keyed; confirmed during
planning). Metrics aggregate fills across both legs.

## 5. Research process (staged, separate job)

Background quant-researcher job (no `src/` edits; single-backtest metrics via a `research/` script
calling `run_backtest`, WFO via `optimize --json`). Stages: signal WFO over `lookback/entry_z/exit_z/
stop_z` on the IS window → frozen OOS → honest findings (no cross-pair confirmation available;
robustness from WFO + OOS). Serial, coarse bars, tiny grids (feed OOM).

## 6. Repo layout (new / changed)

```
src/fxtrader/strategy/examples/pairs_spread_reversion.py   NEW  PairsSpreadReversion
src/fxtrader/strategy/registry.py                          EDIT "xpair" + symbol2 in make_strategy
src/fxtrader/cli.py                                        EDIT symbol2 thread + --symbol2/--stop-z + xpair grid
tests/strategy/test_pairs_spread_reversion.py              NEW  (multi-symbol _drive_pairs harness)
tests/strategy/test_registry.py                            EDIT xpair factory case
tests/test_cli_xpair.py                                    NEW  backtest + optimize end-to-end (2-symbol seed)
```

## 7. Testing strategy (TDD)

Multi-symbol harness `_drive_pairs(strategy, bars)` feeds interleaved primary+secondary `Bar`s (in
time order), applies fills per-symbol at `bar.close`, tracks both positions.

- warmup (no trade before `lookback` spread samples); secondary bar updates `_last` but never trades.
- enter short-spread (`z>entry_z`) → SELL primary + BUY secondary; long-spread (`z<−entry_z`) → mirror
  (assert BOTH leg positions + order sides).
- signal exit (`|z|≤exit_z`) → both legs flat (`_last_exit_reason="signal"`); divergence stop
  (`|z|≥stop_z`) → both flat (`"stop"`).
- one pair at a time; no same-bar re-entry; `std==0` → skip; `ValueError` on bad params; ignores a
  third symbol.
- registry: `make_strategy("xpair", symbol, symbol2, params)` builds it; `STRATEGY_PARAMS["xpair"]`.
- CLI: `backtest --strategy xpair --symbol EURUSD --symbol2 GBPUSD …` e2e on a seeded **two-symbol**
  store (exit 0, summary names both legs); `optimize --strategy xpair` → `param_grid == {lookback,
  entry_z, exit_z, stop_z}`, folds, JSON-serializable; **ma/trend/meanrev/orb grids unchanged**.
- Full suite stays green (baseline 203 passed, 2 skipped).

## 8. Assumptions & risks

- **The spread may not mean-revert profitably after costs** — two legs = two commissions per round
  trip (4 fills), a higher cost hurdle than single-pair. If even the relative-value structure fails,
  that is a strong result. The divergence stop bounds blow-ups (the classic pairs-trade tail: the
  spread keeps diverging — a structural break in the EUR/GBP relationship).
- **β=1 is an assumption**, not estimated; if EURUSD/GBPUSD betas drift, the trade isn't perfectly
  neutral. Rolling-β is the documented refinement.
- **Clock-symbol ≤1-period staleness** in the secondary leg — minor at 30m.
- **No cross-pair confirmation** (only one correlated pair in the data) — robustness from WF + OOS only.
- **Multi-symbol CLI threading** is the largest plumbing change yet; ma/trend/meanrev/orb paths must
  stay byte-identical (single-symbol callers pass `symbol2=None`).
- **Feed performance** (whole-window materialization → OOM; two symbols ≈ 2× ticks in memory) — serial,
  coarse bars, tiny grids.

## 9. Definition of done

- `PairsSpreadReversion`, registry `xpair` + `symbol2`, and the `--strategy xpair` CLI (backtest +
  optimize, two-symbol) implemented and unit-tested; full suite green.
- `--strategy xpair` runs end-to-end through `backtest` and `optimize`; ma/trend/meanrev/orb paths
  unchanged (single-symbol callers unaffected by `symbol2`).
- No engine/live-path changes; live command never run; nothing committed/pushed without request.
- The staged research run + `research/findings-<date>.md` (WF + OOS, no cross-pair confirmation) is the
  intentionally separate follow-up job.
