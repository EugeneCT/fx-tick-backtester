# Mean-Reversion Fader (z-score + Efficiency-Ratio gate) — Design Spec

**Date:** 2026-06-21
**Status:** Approved (design), pending implementation plan
**Author:** Eugene + Claude

## 1. Goal

Add the platform's third strategy: a **z-score mean-reversion fader** with a ranging-regime
gate. The two trend-following variants (naive MA `research/findings-2026-06-18.md`;
trend-filtered `research/findings-2026-06-19.md`) are both net-negative after costs on EURUSD —
weak evidence that 30m EURUSD is closer to mean-reverting/noise than trending at these horizons.
This strategy makes the **opposite bet**: fade stretched moves back toward a rolling mean, but only
when the market is genuinely ranging (not trending), with the existing stop/TP overlay as the risk
backstop.

**Success bar (carried from prior strategies):** a config with **positive out-of-sample Sharpe AND
profit factor > 1 after commissions**, that holds up **across walk-forward folds** (robust, not one
lucky window). Cross-pair agreement (EURUSD *and* GBPUSD) strengthens the result.

## 2. Scope & decisions (locked)

- **Strategy type:** single-symbol, bar-based (same shape as `MaCrossover`/`TrendFilteredCrossover`),
  reuses the engine with **no engine or live-path changes**.
- **Entry:** fade when stretched — `z > entry_z` → SHORT, `z < -entry_z` → LONG, where
  `z = (close - SMA(lookback)) / stdev(lookback)` — **only when the regime is ranging**
  (Kaufman Efficiency Ratio `er < er_threshold`). Entry gating applies to opening only.
- **Exit (fade to the mean), risk-first:** (1) stop/TP via the existing `ExitPolicy`
  (`none`/`fixed`/`atr`) checked first; (2) signal exit when `|z| <= exit_z` (reverted). Either
  exit flattens and **returns** (no same-bar re-entry → bounds turnover).
- **One position at a time, no direct flips:** a fade exits to flat; a later stretched+ranging bar
  may re-enter.
- **Reuse:** `ExitPolicy` (`strategy/exits.py`), `pip_size`, the registry, the MAE/MFE tooling, the
  `--strategy` CLI seam, and the exit-overlay state machine
  (`_entry_price`/`_stop_dist`/`_tp_dist`/`_last_exit_reason`, stop-first same-bar rule).
- **Validation:** **EURUSD + GBPUSD** (USD-quoted → correct single-currency sizing; cross-pair
  agreement is the robustness check). **USDJPY deferred** (bound with `--max-position` if run).
- **IS/OOS split:** in-sample 2023-06-01 → 2025-06-01 (all optimization + MAE derivation); frozen
  out-of-sample 2025-06-01 → 2026-05-01, touched once for finalists. 30m bars, serial runs.

### Out of scope (for now — YAGNI)

- A separate ER window (the z `lookback` doubles as the ER window).
- A max-hold / time-stop exit (only signal + stop/TP in v1).
- Trailing stops, tick-level exit fills, live wiring of the new strategy, USDJPY validation.
- **Generalizing the CLI param plumbing.** v1 *accretes* `meanrev` flags (consistent with how
  `--trend` was added), keeping the merged MA/trend paths byte-identical. With three strategies on
  accreted flags, a focused refactor to generic per-strategy param plumbing is worth its own later
  cycle (separate spec/review), not bundled here.

## 3. Strategy logic — `MeanReversionFader`

Params: `lookback` (int, the z **and** ER window), `entry_z` (float), `exit_z` (float), `er_threshold`
(float), `size` (float), `exit_policy: ExitPolicy`. `__init__` validates: `lookback >= 2`,
`0 <= exit_z < entry_z`, `0 < er_threshold <= 1` → else `ValueError`. Single symbol; ignores bars for
other symbols.

State: a `deque` of closes (`maxlen = lookback + 1`, ER needs `lookback` consecutive diffs), plus the
exit-overlay fields `_entry_price`, `_stop_dist`, `_tp_dist`, `_pos_dir`, `_last_exit_reason`.

Per `on_bar`, once `len(closes) >= lookback + 1`:

```
exit_policy.update(bar); append close
sma  = mean(last `lookback` closes)
std  = stdev(last `lookback` closes)
if std == 0: return            # flat window — no z, no div-by-zero
z    = (close - sma) / std
er   = efficiency_ratio(last `lookback`+1 closes)   # Kaufman ER in [0,1]

# 1. EXIT FIRST (open position only)
if position != 0 and _entry_price is not None:
    long  stop if bar.low  <= entry - stop_dist ; tp if bar.high >= entry + tp_dist
    short stop if bar.high >= entry + stop_dist ; tp if bar.low  <= entry - tp_dist
    #   STOP leg checked before TP leg (same-bar stop wins)
    on hit: flatten; _last_exit_reason = "stop"|"tp"; reset entry/dists/_pos_dir; return
    if abs(z) <= exit_z:        # reverted to the mean — take profit
        flatten; _last_exit_reason = "signal"; reset; return

# 2. ENTRY (only when flat AND ranging)
desired = 0
if position == 0 and er < er_threshold:
    if z >  entry_z:  desired = -size      # stretched high -> fade SHORT
    elif z < -entry_z: desired = +size      # stretched low  -> fade LONG

reconcile to desired (single order for the delta);
on a new non-zero entry: _entry_price = close, (_stop_dist, _tp_dist) = exit_policy.distances()
```

`er >= er_threshold` (trending) → no new fade is opened; a held position still exits normally. After
any exit the bar `return`s, so re-entry waits for the next bar.

## 4. Indicators — `strategy/indicators.py`

Two pure functions, independently testable:

- `zscore(closes: Sequence[float]) -> float` — `(closes[-1] - mean) / stdev` over the given window;
  returns `0.0` when `stdev == 0`.
- `efficiency_ratio(closes: Sequence[float]) -> float` — Kaufman ER =
  `abs(closes[-1] - closes[0]) / sum(abs(c_i - c_{i-1}))`; returns `0.0` when the denominator is 0
  (a flat window is not "trending"). Result in `[0, 1]`.

## 5. Reuse — exits, registry, CLI

- **Exits:** unchanged `ExitPolicy` (`NoExit`/`FixedPips`/`AtrMultiple`). The fader holds one
  `exit_policy` and uses the same captured-at-entry distances + stop-first same-bar rule as
  `TrendFilteredCrossover`.
- **Registry (`strategy/registry.py`):** add `STRATEGY_PARAMS["meanrev"] = ("lookback", "entry_z",
  "exit_z", "er_threshold")` and a `make_strategy("meanrev", ...)` branch constructing
  `MeanReversionFader(symbol, lookback, entry_z, exit_z, er_threshold, size, exit_policy)`.
- **CLI (`cli.py`):** add `--lookback` (int), `--entry-z`/`--exit-z`/`--er-threshold` (float) to
  `backtest` and `optimize`; a `_float_list` parser for the float grids; a `"meanrev"` branch in the
  optimize grid-builder (`{"lookback":…, "entry_z":…, "exit_z":…, "er_threshold":…}`); and the
  matching `run_backtest` kwargs (defaulting harmlessly so the `ma`/`trend` paths are unaffected).
  `backtest --strategy meanrev` prints a strategy-aware summary and routes through `run_backtest`
  (flag-driven; `--json/--html` remain MA-only, per the merged convention).

## 6. Research process (staged, separate job)

Same discipline as the trend study; run as its own background quant-researcher job, human reviews:
1. **Signal WFO:** small grid over `lookback`/`entry_z`/`exit_z`/`er_threshold` with `exit none`,
   EURUSD in-sample, 30m, n-splits 3 → robust signal params (per-fold OOS reported).
2. **MAE/MFE diagnostic** on the chosen signal (no exits) → derive candidate fixed-pip + ATR levels.
3. **Exit WFO/backtests** on top of the fixed signal → best family by OOS.
4. **Confirm** finalist once on frozen OOS; **repeat on GBPUSD** for cross-pair agreement; write
   `research/findings-<date>.md` with an honest verdict (incl. "no edge" if that's the truth).
Compute: serial, coarse bars, tiny grids (feed materializes the whole window → OOM risk).

## 7. Repo layout (new / changed)

```
src/fxtrader/strategy/
  indicators.py                          NEW  zscore + efficiency_ratio (pure)
  examples/mean_reversion_fader.py       NEW  MeanReversionFader
  registry.py                            EDIT add "meanrev" to STRATEGY_PARAMS + make_strategy
src/fxtrader/cli.py                      EDIT --lookback/--entry-z/--exit-z/--er-threshold +
                                              _float_list + meanrev grid + run_backtest kwargs
tests/strategy/test_indicators.py        NEW
tests/strategy/test_mean_reversion_fader.py NEW
tests/strategy/test_registry.py          EDIT meanrev factory case
tests/ (cli meanrev tests)               NEW/EDIT backtest+optimize end-to-end
```

## 8. Testing strategy (TDD)

- **indicators:** `zscore` known series → known z; `std==0` → 0.0. `efficiency_ratio` monotonic ramp
  → ≈1; oscillating-no-net → ≈0; zero-gross → 0.0.
- **strategy:** warmup (no trades before `lookback+1`); SHORT on `z>entry_z & er<thr`; LONG on
  `z<-entry_z & er<thr`; **regime gate** — stretched but `er>=thr` (a monotonic ramp) → **no entry**;
  signal exit at `|z|<=exit_z` (`_last_exit_reason="signal"`); stop/TP exit via `FixedPips`,
  stop-first (`_last_exit_reason="stop"`); no same-bar re-entry after an exit; `std==0` → no trade;
  `ValueError` on `exit_z>=entry_z` / `lookback<2` / `er_threshold` out of `(0,1]`; ignores other
  symbols. (Seed series must hit a target z and ER together — natural, since a monotonic ramp yields
  both a high ER and a large z.)
- **registry:** `make_strategy("meanrev")` builds it with `exit_policy`; `STRATEGY_PARAMS["meanrev"]`.
- **CLI:** `backtest --strategy meanrev` runs end-to-end on a seeded store (exit 0, summary names the
  strategy); `optimize --strategy meanrev` (float grids) → `param_grid == {lookback, entry_z, exit_z,
  er_threshold}`, has folds, JSON-serializable; **MA/trend grids unchanged** (existing tests green).
- Full suite stays green (baseline at spec time: 171 passed, 2 skipped).

## 9. Assumptions & risks

- **Mean-reversion may have no edge either** — if so, the regime gate + stop/TP at least limit the
  bleed, and we'll have tested both structural hypotheses (trend and fade) on the platform.
- **Regime gate over-fits** if `er_threshold` is tuned hard — mitigated by deriving it in-sample via
  walk-forward and confirming OOS + on GBPUSD.
- **Seed-data construction** for tests is fiddly (z and ER coupled) — flagged; a monotonic ramp gives
  the high-ER/high-z regime-gate case naturally.
- **Feed performance** (whole-window materialization → OOM, ~5× runtime) constrains breadth — serial,
  coarse bars, tiny grids; the feed refactor remains a separate concern.
- **CLI accretion:** three strategies now share accreted flags; generalizing the param plumbing is a
  deliberately deferred, separate cycle.

## 10. Definition of done

- `indicators.zscore`/`efficiency_ratio`, `MeanReversionFader`, registry `meanrev` wiring, and the
  `--strategy meanrev` CLI (backtest + optimize) implemented and unit-tested; full suite green.
- `--strategy meanrev` runs end-to-end through `backtest` and `optimize`; MA/trend paths unchanged.
- No engine/live-path changes; live command never run; nothing committed/pushed without request.
- The staged research run + `research/findings-<date>.md` (EURUSD + GBPUSD) is the intentionally
  separate follow-up job.
