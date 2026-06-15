# FX Trader Platform — Design Spec

**Date:** 2026-06-15
**Status:** Approved (design), pending implementation plan
**Author:** Eugene + Claude

## 1. Goal

Build an algorithmic FX trading platform that:

1. Ingests and stores **Dukascopy tick data** for a handful of major pairs.
2. **Backtests** strategies against that tick data with a realistic, event-driven simulator.
3. Trades the *same* strategy logic **live on Interactive Brokers** (paper account first).
4. Hosts a **quant research agent** (built last) — an LLM subagent that drives a programmatic optimizer.

The defining principle: **a strategy you backtest is byte-for-byte the same code that trades live.** This is achieved with one `Strategy` interface and a swappable `Broker`/`DataFeed` behind it.

## 2. Scope & decisions (locked)

| Decision | Choice |
|---|---|
| Strategy source | Build a **framework + reference strategy** (MA-crossover). Real alpha comes later via the research agent. |
| Live execution | **IB paper account first.** Real money is gated behind explicit config. |
| Backtest fidelity | **Tick-level, event-driven**, with optional bar resampling. |
| Data scope | **EUR/USD, GBP/USD, USD/JPY**, ~3 years of ticks. |
| Research agent | **Both** — an LLM subagent (`.claude/agents`) that drives a programmatic optimizer. |

### Out of scope (for now — YAGNI)
- Portfolio/cross-sectional strategies across many pairs.
- Non-IB brokers; non-Dukascopy data sources (interfaces allow them later, but only one impl each is built).
- Real-money live trading (interfaces support it; it stays gated until explicitly enabled).
- A web UI / dashboard. Reports are HTML/markdown + JSON files.

## 3. Tech stack

| Concern | Choice | Rationale |
|---|---|---|
| Language / runtime | **Python pinned to 3.12** in a dedicated venv (via pyenv) | 3.14 is installed but quant wheels lag; 3.12 is fully supported. |
| Data download | Custom **Dukascopy `.bi5` fetcher** (LZMA decompress + binary struct decode), behind a `DataSource` interface | No reliance on unmaintained wrappers; full control of retries/gaps. |
| Storage | **Parquet**, partitioned by `symbol/year/month`; queried via **DuckDB** | Columnar, compresses ticks well, fast range scans, no DB server. |
| Dataframes | **Polars** in the data layer | Far faster than pandas at tick volumes. |
| Backtest engine | **Custom event-driven** engine, numba-optimizable hot loop if needed | Full tick fidelity; same strategy code runs live. |
| IB connectivity | **`ib_async`** (maintained successor to `ib_insync`) | Clean async API. Requires IB Gateway/TWS running. |
| Config | **pydantic** typed settings + **YAML** run configs | Validation + reproducibility. |
| CLI | **typer** | Scriptable surface the research agent drives. |
| Optimizer (P4) | **Optuna** + custom walk-forward harness | Standard; supports pruning + parallel sweeps. |
| Plotting / report | **matplotlib** → HTML/markdown; plus machine-readable **JSON** | Human + agent consumers. |
| Testing | **pytest** | TDD throughout. |

## 4. Architecture

### 4.1 The keystone: one Strategy, two engines

```
            ┌──────────────────────────────┐
            │          Strategy            │  on_tick / on_bar / on_fill
            │  (pure logic, no I/O)        │  → emits Orders
            └──────────────┬───────────────┘
                           │ Orders
                  ┌────────▼─────────┐
                  │   RiskManager    │  enforced in BOTH paths
                  └────────┬─────────┘
                           │ approved Orders
                ┌──────────▼───────────┐
                │    Broker (iface)    │
                ├──────────┬───────────┤
   backtest →   │ Simulated│  IBBroker │   ← live (paper)
                │  Broker  │           │
                └──────────┴───────────┘
```

The `Strategy` never imports a broker, a data source, or IB. It receives events and emits intents. The **engine** wires the concrete `DataFeed` + `Broker` + `Clock`.

### 4.2 Shared core abstractions (`fxtrader.core`)
- **Events / types:** `Tick(symbol, ts, bid, ask, bid_vol, ask_vol)`, `Bar(symbol, ts, o,h,l,c,v)`, `Order`, `Fill`, `Position`.
- **`Portfolio`** — positions, cash, realized/unrealized PnL, equity curve, trade log.
- **`Clock`** — `SimClock` (driven by event timestamps) vs `WallClock` (live). Strategies read time only from the clock.
- **`DataFeed`** interface — `HistoricalTickFeed` (Parquet) ↔ `IBLiveFeed`. Both yield the same `Tick` events in time order.
- **`Broker`** interface — `submit(order)`, `cancel(order)`, position/account queries, and a fill callback.
- **`RiskManager`** — vets every order before it reaches the broker.

### 4.3 Data flow

**Backtest:** `Parquet ticks → HistoricalTickFeed → (optional Resampler → Bars) → Strategy → Orders → RiskManager → SimulatedBroker (fill model) → Fills → Portfolio → metrics → report (HTML + JSON)`

**Live (paper):** `IB ticks → IBLiveFeed → Strategy → Orders → RiskManager → IBBroker → IB Gateway (paper) → Fills → Portfolio → state persistence + heartbeat log`

## 5. Component designs

### 5.1 Data layer (`fxtrader.data`)
- **`dukascopy.py`** — fetch hourly `.bi5` from `datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM-1:02d}/{DD:02d}/{HH:02d}h_ticks.bi5` (note: Dukascopy months are **0-indexed**). Decompress LZMA, decode the 20-byte big-endian records `(ms_offset:int32, ask:int32, bid:int32, ask_vol:float32, bid_vol:float32)`, convert prices using the instrument's point value. Retries, polite rate-limiting, resumable.
- **`store.py`** — write/read Parquet partitioned `data/<symbol>/<year>/<month>.parquet`; maintain a `manifest.json` of stored ranges; gap detection.
- **`feed.py`** — `HistoricalTickFeed(symbols, start, end)` streams ticks in global time order across symbols; `Resampler` builds bars on the fly.
- **Validation:** round-trip tests; sanity checks (bid ≤ ask, monotonic timestamps, plausible spreads); spot-compare a known day against a reference.

### 5.2 Backtester (`fxtrader.engine.backtest`, `fxtrader.broker.simulated`)
- **`SimulatedBroker`** — fills market orders at the current ask (buy) / bid (sell) plus configurable **slippage**; supports limit/stop; applies an **IB-like commission model** (per-side, min ticket). Models spread implicitly via bid/ask in the tick.
- **`BacktestEngine`** — pulls events from the feed, advances `SimClock`, dispatches to strategy, processes resulting orders through `RiskManager` → broker, applies fills to `Portfolio`.
- **Determinism:** identical input + seed → identical output.

### 5.3 Strategy (`fxtrader.strategy`)
- **`Strategy` ABC** — lifecycle hooks `on_start`, `on_tick`, `on_bar`, `on_fill`, `on_stop`; helper `self.order(...)`; reads `self.clock`, `self.portfolio` (read-only views).
- **Reference: `MaCrossover`** — fast/slow MA on resampled bars; long when fast crosses above, flat/short on cross below; fixed position size. Fully parametrized (windows, size, symbol) so it doubles as the optimizer's first target.

### 5.4 Risk (`fxtrader.risk`)
- **`RiskManager`** hard limits: max position per symbol, max gross exposure, **daily-loss kill switch** (flatten all + halt for the session), max orders/minute runaway guard, reject orders when disconnected.
- Configured via the run config; identical enforcement in backtest and live.

### 5.5 Live trading (`fxtrader.engine.live`, `fxtrader.broker.ib`)
- **`IBBroker`** — wraps `ib_async`: connect to Gateway/TWS, subscribe to FX market data, place/cancel orders, reconcile positions/fills, map IB callbacks to core `Fill` events.
- **`LiveEngine`** — `WallClock`, real-time feed, the same strategy + risk path; persists state (positions, daily PnL, kill-switch status) to disk so a restart is safe; heartbeat log.
- **Mode gating:** `DRY_RUN` (no orders sent) → `PAPER` (paper account) → `LIVE` (real; requires `mode: live` *and* an explicit `--i-understand-live` flag). Default is `PAPER`.

### 5.6 Metrics & reporting (`fxtrader.metrics`)
- **`performance.py`** — returns, Sharpe, Sortino, max drawdown, CAGR, win rate, profit factor, exposure, turnover, trade stats.
- **`report.py`** — renders an HTML/markdown report with an equity-curve plot **and** writes a `results.json` (every metric machine-readable) so the research agent can parse outcomes.

### 5.7 CLI (`fxtrader.cli`)
```
fxtrader download  --symbols EURUSD,GBPUSD,USDJPY --start 2023-01-01 --end 2025-12-31
fxtrader backtest  --config config/ma_eurusd.yaml          # → report.html + results.json
fxtrader optimize  --config config/ma_eurusd.yaml --param-space ... --walk-forward ...   # P4
fxtrader live      --config config/ma_eurusd_live.yaml --mode paper
```
Every command is non-interactive and emits JSON, so the agent can drive the whole loop.

### 5.8 Research agent (Phase 4)
- **Optimizer backend** (`fxtrader.research.optimize`) — Optuna study over a strategy's param space with **walk-forward** train/test windows; outputs ranked configs + per-fold metrics as JSON. Deterministic given a seed.
- **`.claude/agents/quant-researcher.md`** — a subagent that: forms a hypothesis → parametrizes/edits a strategy → runs `backtest`/`optimize` via the CLI → reads `results.json` → iterates → writes a findings report. **It never deploys to live; a human reviews and promotes.** Constrained toolset (Bash for the CLI, Read/Write for strategies & reports).

## 6. Repo layout

```
fx_trader/
  pyproject.toml            # 3.12 target, deps, ruff/pytest config
  README.md                 # setup incl. IB Gateway steps
  .env.example              # IB host/port/clientId, data dir
  .gitignore                # data/, .venv/, secrets
  config/                   # YAML run configs (backtest + live)
  data/                     # Parquet store (gitignored)
  src/fxtrader/
    core/      events.py types.py portfolio.py clock.py
    data/      dukascopy.py store.py feed.py
    strategy/  base.py examples/ma_crossover.py
    broker/    base.py simulated.py ib.py
    risk/      manager.py
    engine/    backtest.py live.py
    metrics/   performance.py report.py
    research/  optimize.py            # P4
    cli.py
  tests/                    # pytest, mirrors src layout
  .claude/agents/quant-researcher.md  # P4
  docs/superpowers/specs/
```

## 7. Build phases

Each phase is independently testable and lands before the next starts.

- **P0 — Scaffold:** venv(3.12), `pyproject.toml`, ruff+pytest, `core` types/events/clock/portfolio with unit tests.
- **P1 — Data:** Dukascopy downloader + Parquet store + `HistoricalTickFeed` + resampler; verified on real EUR/USD days.
- **P2 — Backtester:** `SimulatedBroker` + `BacktestEngine` + `RiskManager` + metrics/report + `MaCrossover`; golden-path integration test + a full EUR/USD backtest.
- **P3 — Live (paper):** `IBBroker` + `LiveEngine` + kill switch + state persistence; reference strategy on IB **paper** via Gateway.
- **P4 — Research agent:** Optuna walk-forward optimizer + `quant-researcher` subagent + report templates.

## 8. Testing strategy

- **Unit:** core types, Parquet round-trip, resampler correctness, fill/slippage/commission math, every risk limit, metric formulas.
- **Integration (golden path):** a small synthetic tick set → backtest → asserted equity curve & trade log (locks determinism).
- **Live smoke (manual/CI-gated):** connect to IB paper, place + cancel a tiny order, confirm reconciliation.
- **TDD** during implementation (superpowers `test-driven-development`).

## 9. Assumptions & risks

1. **IB market data** — real-time FX quotes on IB may require a market-data subscription; paper accounts mirror the linked live account's subscriptions. The live loop uses IB data; documented as a caveat. (Fallback option: drive signals from a separate feed — not built now.)
2. **IB Gateway/TWS must be running** for any live/paper run — a one-time manual setup, documented in the README.
3. **Dukascopy format** — `.bi5` is LZMA-compressed, months are 0-indexed, prices are integers scaled by the instrument point value; weekends/holidays have no data (handled as gaps, not errors).
4. **Python 3.14 wheel gaps** — mitigated by pinning the project venv to 3.12.
5. **Tick-loop performance** — pure-Python per-tick may be slow over 3 years × 3 pairs; design keeps the hot loop isolated so numba/vectorization can be added without touching strategy code.

## 10. Definition of done (core platform, P0–P3)

- `fxtrader download` populates a Parquet store for the three majors.
- `fxtrader backtest` produces a report + `results.json` for the reference strategy on EUR/USD.
- `fxtrader live --mode paper` runs the reference strategy against IB paper with the kill switch active.
- All unit + golden-path integration tests pass.
- README documents setup end to end. P4 (research agent) is specced and ready as the next cycle.
