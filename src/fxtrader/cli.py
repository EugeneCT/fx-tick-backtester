"""fxtrader command-line interface."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import typer

from fxtrader.broker.ib import IBBroker
from fxtrader.broker.simulated import SimulatedBroker
from fxtrader.config import BacktestConfig, load_config
from fxtrader.core.portfolio import Portfolio
from fxtrader.data.backfill import CoverageLedger, backfill as _backfill
from fxtrader.data.dukascopy import Downloader, _http_download, download_range
from fxtrader.data.feed import HistoricalTickFeed
from fxtrader.data.store import TickStore
from fxtrader.engine.backtest import BacktestEngine
from fxtrader.engine.live import LiveEngine, TradingMode
from fxtrader.metrics.performance import compute_metrics
from fxtrader.metrics.report import _json_safe, write_html_report, write_results_json
from fxtrader.research.optimize import walk_forward_optimize
from fxtrader.risk.manager import RiskLimits, RiskManager
from fxtrader.strategy.examples.ma_crossover import MaCrossover
from fxtrader.strategy.registry import make_exit_policy, make_strategy

app = typer.Typer(add_completion=False, help="fxtrader CLI")


@app.callback()
def _root() -> None:
    """fxtrader CLI.

    A no-op root callback so Typer keeps subcommands (``download``, and later
    ``backtest``/``optimize``/``live``) instead of collapsing a single command
    into the program itself.
    """


def _parse_day(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def run_download(
    symbols: list[str],
    start: datetime,
    end: datetime,
    data_dir: str | Path,
    downloader: Downloader = _http_download,
) -> dict[str, int]:
    """Download [start, end) for each symbol into a TickStore. Returns tick counts."""
    store = TickStore(data_dir)
    summary: dict[str, int] = {}
    for symbol in symbols:
        ticks = download_range(symbol, start, end, downloader=downloader)
        store.write_ticks(symbol, ticks)
        summary[symbol] = len(ticks)
    return summary


def run_backfill(
    symbols: list[str],
    start: datetime,
    end: datetime,
    data_dir: str | Path,
    downloader: Downloader = _http_download,
    workers: int = 8,
    retries: int = 3,
    resume: bool = True,
    progress=lambda msg: None,
) -> dict[str, int]:
    """Resilient bulk download into a TickStore, resumable via a coverage ledger."""
    store = TickStore(data_dir)
    ledger = CoverageLedger(Path(data_dir) / "coverage.json") if resume else None
    return _backfill(
        symbols, start, end, store,
        downloader=downloader, workers=workers, retries=retries,
        ledger=ledger, progress=progress,
    )


def run_backtest(
    symbol: str,
    start: datetime,
    end: datetime,
    data_dir: str | Path,
    *,
    fast: int = 20,
    slow: int = 50,
    size: float = 10_000,
    bar_minutes: int = 1,
    starting_cash: float = 100_000.0,
    commission_rate: float = 0.00002,
    min_commission: float = 2.0,
    slippage: float = 0.0,
    daily_loss_limit: float = float("inf"),
    max_position: float = float("inf"),
    strategy_name: str = "ma",
    trend: int = 0,
    exit_kind: str = "none",
    stop_pips: float = 0.0,
    tp_pips: float = 0.0,
    atr_period: int = 0,
    stop_atr: float = 0.0,
    tp_atr: float = 0.0,
    lookback: int = 0,
    entry_z: float = 0.0,
    exit_z: float = 0.0,
    er_threshold: float = 0.0,
    range_start_hour: int = 0,
    range_end_hour: int = 0,
    entry_end_hour: int = 0,
    exit_hour: int = 0,
):
    """Run a backtest over stored ticks. Returns (BacktestResult, metrics)."""
    store = TickStore(data_dir)
    feed = HistoricalTickFeed(store, [symbol], start, end)
    exit_policy = make_exit_policy(
        exit_kind,
        params={
            "stop_pips": stop_pips, "tp_pips": tp_pips,
            "atr_period": atr_period, "stop_atr": stop_atr, "tp_atr": tp_atr,
        },
        symbol=symbol,
    )
    strategy = make_strategy(
        strategy_name, symbol=symbol,
        params={
            "fast": fast, "slow": slow, "trend": trend, "size": size,
            "lookback": lookback, "entry_z": entry_z, "exit_z": exit_z, "er_threshold": er_threshold,
            "range_start_hour": range_start_hour, "range_end_hour": range_end_hour,
            "entry_end_hour": entry_end_hour, "exit_hour": exit_hour,
        },
        exit_policy=exit_policy,
    )
    broker = SimulatedBroker(
        commission_rate=commission_rate, min_commission=min_commission, slippage=slippage
    )
    portfolio = Portfolio(starting_cash)
    risk = RiskManager(RiskLimits(max_position=max_position, daily_loss_limit=daily_loss_limit))
    bar_interval = timedelta(minutes=bar_minutes)
    engine = BacktestEngine(
        feed=feed, strategy=strategy, broker=broker, portfolio=portfolio, risk=risk,
        bar_interval=bar_interval, equity_interval=bar_interval,
    )
    result = engine.run()
    periods_per_year = (365.25 * 24 * 3600) / bar_interval.total_seconds()
    metrics = compute_metrics(result.equity_curve, result.fills, periods_per_year=periods_per_year)
    return result, metrics


def run_backtest_config(config: BacktestConfig, data_dir: str | Path):
    """Run a backtest from a BacktestConfig. Returns (BacktestResult, metrics)."""
    return run_backtest(data_dir=data_dir, **config.to_run_kwargs())


def _strategy_run_fn(
    strategy_name: str,
    symbol: str,
    data_dir: str | Path,
    bar_minutes: int,
    base_params: dict,
    exit_kind: str,
):
    """Build a run_fn(params, start, end) -> metrics for any strategy grid (None on invalid params)."""
    def run_fn(params: dict, start: datetime, end: datetime):
        p = {**base_params, **params}
        if strategy_name in ("ma", "trend"):
            if p["fast"] >= p["slow"]:
                return None
            if strategy_name == "trend" and not (p["fast"] < p["slow"] < p["trend"]):
                return None
        elif strategy_name == "meanrev":
            if not (p["lookback"] >= 2 and 0 <= p["exit_z"] < p["entry_z"] and 0 < p["er_threshold"] <= 1):
                return None
        _result, metrics = run_backtest(
            symbol=symbol, start=start, end=end, data_dir=data_dir,
            fast=p.get("fast", 20), slow=p.get("slow", 50), size=p.get("size", 10_000),
            bar_minutes=bar_minutes, strategy_name=strategy_name,
            trend=p.get("trend", 0), exit_kind=exit_kind,
            stop_pips=p.get("stop_pips", 0.0), tp_pips=p.get("tp_pips", 0.0),
            atr_period=p.get("atr_period", 14), stop_atr=p.get("stop_atr", 0.0),
            tp_atr=p.get("tp_atr", 0.0),
            lookback=p.get("lookback", 0), entry_z=p.get("entry_z", 0.0),
            exit_z=p.get("exit_z", 0.0), er_threshold=p.get("er_threshold", 0.0),
        )
        return metrics

    return run_fn


def run_optimize(
    symbol: str,
    start: datetime,
    end: datetime,
    data_dir: str | Path,
    *,
    fast_values: list[int] | None = None,
    slow_values: list[int] | None = None,
    n_splits: int = 3,
    train_ratio: float = 0.7,
    bar_minutes: int = 5,
    size: float = 10_000,
    objective: str = "sharpe",
    strategy: str = "ma",
    trend_values: list[int] | None = None,
    exit_kind: str = "none",
    stop_pips: float = 0.0,
    tp_pips: float = 0.0,
    atr_period: int = 14,
    stop_atr: float = 0.0,
    tp_atr: float = 0.0,
    lookback_values: list[int] | None = None,
    entry_z_values: list[float] | None = None,
    exit_z_values: list[float] | None = None,
    er_threshold_values: list[float] | None = None,
) -> dict:
    """Walk-forward grid optimization over stored ticks. Returns strict-JSON-safe dict."""
    # Build the signal param grid from the selected strategy's keys.
    # MA: {"fast": ..., "slow": ...}  (UNCHANGED — existing tests assert exactly this)
    # Trend: {"fast": ..., "slow": ..., "trend": ...}
    # Meanrev: {"lookback": ..., "entry_z": ..., "exit_z": ..., "er_threshold": ...}
    if strategy == "meanrev":
        grid: dict = {
            "lookback": lookback_values or [],
            "entry_z": entry_z_values or [],
            "exit_z": exit_z_values or [],
            "er_threshold": er_threshold_values or [],
        }
    else:
        grid = {"fast": fast_values, "slow": slow_values}
        if strategy == "trend":
            grid["trend"] = trend_values or []

    # Fixed (non-gridded) params shared across all run_fn calls.
    base_params: dict = {
        "size": size,
        "stop_pips": stop_pips,
        "tp_pips": tp_pips,
        "atr_period": atr_period,
        "stop_atr": stop_atr,
        "tp_atr": tp_atr,
    }

    run_fn = _strategy_run_fn(strategy, symbol, data_dir, bar_minutes, base_params, exit_kind)
    out = walk_forward_optimize(
        run_fn, start, end, grid,
        n_splits=n_splits, train_ratio=train_ratio, objective=objective,
    )
    return _json_safe(out)


def build_live_engine(
    symbol: str,
    *,
    mode: TradingMode,
    fast: int = 20,
    slow: int = 50,
    size: float = 10_000,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
    starting_cash: float = 100_000.0,
    daily_loss_limit: float = float("inf"),
    max_position: float = float("inf"),
    state_path: str | Path | None = "live_state.json",
) -> LiveEngine:
    """Wire an IBBroker + MaCrossover + RiskManager into a LiveEngine (not yet connected)."""
    broker = IBBroker(host=host, port=port, client_id=client_id, symbols=[symbol])
    strategy = MaCrossover(symbol=symbol, fast=fast, slow=slow, size=size)
    portfolio = Portfolio(starting_cash)
    risk = RiskManager(RiskLimits(max_position=max_position, daily_loss_limit=daily_loss_limit))
    return LiveEngine(
        broker=broker, strategy=strategy, portfolio=portfolio, risk=risk,
        mode=mode, state_path=state_path,
    )


def _resolve_mode(mode: str, *, confirm_live: bool) -> TradingMode:
    key = mode.strip().upper().replace("-", "_")
    try:
        resolved = TradingMode(key)
    except ValueError:
        raise typer.BadParameter(f"mode must be dry-run | paper | live, got {mode!r}") from None
    if resolved is TradingMode.LIVE and not confirm_live:
        raise typer.BadParameter("LIVE trading requires --i-understand-live (real money).")
    return resolved


@app.command()
def download(
    symbols: str = typer.Option(..., help="Comma-separated, e.g. EURUSD,GBPUSD,USDJPY"),
    start: str = typer.Option(..., help="Inclusive start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Exclusive end date YYYY-MM-DD"),
    data_dir: str = typer.Option("data", help="Parquet store root"),
) -> None:
    """Download Dukascopy ticks into the local Parquet store."""
    summary = run_download(
        symbols=[s.strip() for s in symbols.split(",") if s.strip()],
        start=_parse_day(start),
        end=_parse_day(end),
        data_dir=data_dir,
    )
    for sym, count in summary.items():
        typer.echo(f"{sym}: {count} ticks")


@app.command()
def backfill(
    symbols: str = typer.Option(..., help="Comma-separated, e.g. EURUSD,GBPUSD,USDJPY"),
    start: str = typer.Option(..., help="Inclusive start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Exclusive end date YYYY-MM-DD"),
    data_dir: str = typer.Option("data", help="Parquet store root"),
    workers: int = typer.Option(8, help="Concurrent hourly downloads"),
    retries: int = typer.Option(3, help="Retries per hour on transient errors"),
    resume: bool = typer.Option(True, help="Skip months already in the coverage ledger"),
) -> None:
    """Resilient bulk download of Dukascopy ticks (concurrent, resumable)."""
    summary = run_backfill(
        symbols=[s.strip() for s in symbols.split(",") if s.strip()],
        start=_parse_day(start),
        end=_parse_day(end),
        data_dir=data_dir,
        workers=workers,
        retries=retries,
        resume=resume,
        progress=typer.echo,
    )
    for sym, count in summary.items():
        typer.echo(f"TOTAL {sym}: {count} ticks")


@app.command()
def backtest(
    symbol: str = typer.Option(None, help="Single symbol, e.g. EURUSD (or use --config)"),
    start: str = typer.Option(None, help="Inclusive start date YYYY-MM-DD"),
    end: str = typer.Option(None, help="Exclusive end date YYYY-MM-DD"),
    data_dir: str = typer.Option("data", help="Parquet store root"),
    fast: int = typer.Option(20, help="Fast MA window (bars)"),
    slow: int = typer.Option(50, help="Slow MA window (bars)"),
    size: float = typer.Option(10_000, help="Position size (base units)"),
    bar_minutes: int = typer.Option(1, help="Bar interval in minutes"),
    max_position: float = typer.Option(0.0, help="Max abs position per symbol (0 = unlimited)"),
    daily_loss_limit: float = typer.Option(0.0, help="Daily-loss kill switch (0 = disabled)"),
    config: str = typer.Option(None, "--config", help="YAML run config (supersedes the flags above)"),
    json_out: str = typer.Option(None, "--json", help="Write results.json to this path"),
    html_out: str = typer.Option(None, "--html", help="Write an HTML report to this path"),
    strategy: str = typer.Option("ma", help="ma | trend | meanrev | orb"),
    trend: int = typer.Option(200, help="Trend-filter MA window (trend strategy)"),
    lookback: int = typer.Option(20, help="Z/ER window (meanrev)"),
    entry_z: float = typer.Option(2.0, "--entry-z", help="Entry z-threshold (meanrev)"),
    exit_z: float = typer.Option(0.5, "--exit-z", help="Exit z-threshold (meanrev)"),
    er_threshold: float = typer.Option(0.3, "--er-threshold", help="Max efficiency ratio to fade (meanrev)"),
    range_start: int = typer.Option(0, "--range-start", help="Range-build start hour UTC (orb)"),
    range_end: int = typer.Option(7, "--range-end", help="Range-build end / entry start hour UTC (orb)"),
    entry_end: int = typer.Option(16, "--entry-end", help="Stop taking entries after this hour UTC (orb)"),
    exit_hour: int = typer.Option(21, "--exit-hour", help="Flatten-all hour UTC (orb)"),
    exit_: str = typer.Option("none", "--exit", help="none | fixed | atr"),
    stop_pips: float = typer.Option(0.0, help="Stop in pips (exit=fixed)"),
    tp_pips: float = typer.Option(0.0, help="Take-profit in pips (exit=fixed)"),
    atr_period: int = typer.Option(14, help="ATR period (exit=atr)"),
    stop_atr: float = typer.Option(0.0, help="Stop = mult x ATR (exit=atr)"),
    tp_atr: float = typer.Option(0.0, help="TP = mult x ATR (exit=atr)"),
) -> None:
    """Backtest a strategy over stored ticks."""
    if strategy not in ("ma", "trend", "meanrev", "orb"):
        raise typer.BadParameter("strategy must be ma | trend | meanrev | orb")
    if strategy == "orb":
        if not (symbol and start and end):
            raise typer.BadParameter("orb strategy needs --symbol/--start/--end")
        result, metrics = run_backtest(
            symbol=symbol, start=_parse_day(start), end=_parse_day(end), data_dir=data_dir,
            size=size, bar_minutes=bar_minutes, strategy_name="orb",
            range_start_hour=range_start, range_end_hour=range_end,
            entry_end_hour=entry_end, exit_hour=exit_hour,
            exit_kind=exit_, stop_pips=stop_pips, tp_pips=tp_pips,
            atr_period=atr_period, stop_atr=stop_atr, tp_atr=tp_atr,
            daily_loss_limit=daily_loss_limit or float("inf"),
            max_position=max_position or float("inf"),
        )
        typer.echo(f"symbol={symbol}  bars={bar_minutes}m  orb(range={range_start}-{range_end} entry<{entry_end} exit@{exit_hour})  exit={exit_}")
        typer.echo(f"  trades        : {metrics['num_trades']}")
        typer.echo(f"  total return  : {metrics['total_return']:.2%}")
        typer.echo(f"  Sharpe        : {metrics['sharpe']:.2f}")
        typer.echo(f"  profit factor : {metrics['profit_factor']:.2f}")
        return
    if strategy == "meanrev":
        if not (symbol and start and end):
            raise typer.BadParameter("meanrev strategy needs --symbol/--start/--end")
        result, metrics = run_backtest(
            symbol=symbol, start=_parse_day(start), end=_parse_day(end), data_dir=data_dir,
            size=size, bar_minutes=bar_minutes, strategy_name="meanrev",
            lookback=lookback, entry_z=entry_z, exit_z=exit_z, er_threshold=er_threshold,
            exit_kind=exit_, stop_pips=stop_pips, tp_pips=tp_pips,
            atr_period=atr_period, stop_atr=stop_atr, tp_atr=tp_atr,
            daily_loss_limit=daily_loss_limit or float("inf"),
            max_position=max_position or float("inf"),
        )
        typer.echo(f"symbol={symbol}  bars={bar_minutes}m  meanrev(lookback={lookback} entry_z={entry_z} exit_z={exit_z} er<{er_threshold})  exit={exit_}")
        typer.echo(f"  trades        : {metrics['num_trades']}")
        typer.echo(f"  total return  : {metrics['total_return']:.2%}")
        typer.echo(f"  Sharpe        : {metrics['sharpe']:.2f}")
        typer.echo(f"  profit factor : {metrics['profit_factor']:.2f}")
        return
    if strategy == "trend":
        if not (symbol and start and end):
            raise typer.BadParameter("trend strategy needs --symbol/--start/--end")
        result, metrics = run_backtest(
            symbol=symbol, start=_parse_day(start), end=_parse_day(end), data_dir=data_dir,
            fast=fast, slow=slow, size=size, bar_minutes=bar_minutes,
            strategy_name="trend", trend=trend, exit_kind=exit_,
            stop_pips=stop_pips, tp_pips=tp_pips,
            atr_period=atr_period, stop_atr=stop_atr, tp_atr=tp_atr,
            daily_loss_limit=daily_loss_limit or float("inf"),
            max_position=max_position or float("inf"),
        )
        typer.echo(f"symbol={symbol}  bars={bar_minutes}m  trend MA({fast}/{slow}/{trend})  exit={exit_}")
        typer.echo(f"  trades        : {metrics['num_trades']}")
        typer.echo(f"  total return  : {metrics['total_return']:.2%}")
        typer.echo(f"  Sharpe        : {metrics['sharpe']:.2f}")
        typer.echo(f"  profit factor : {metrics['profit_factor']:.2f}")
        return
    if config is not None:
        cfg = load_config(config)
    else:
        if not (symbol and start and end):
            raise typer.BadParameter("provide --config, or --symbol/--start/--end")
        cfg = BacktestConfig(
            symbol=symbol, start=start, end=end, fast=fast, slow=slow, size=size,
            bar_minutes=bar_minutes,
            daily_loss_limit=daily_loss_limit or None,
            max_position=max_position or None,
        )

    result, metrics = run_backtest_config(cfg, data_dir=data_dir)

    typer.echo(f"symbol={cfg.symbol}  bars={cfg.bar_minutes}m  MA({cfg.fast}/{cfg.slow})")
    typer.echo(f"  trades        : {metrics['num_trades']}")
    typer.echo(f"  win rate      : {metrics['win_rate']:.1%}")
    typer.echo(f"  total return  : {metrics['total_return']:.2%}")
    typer.echo(f"  CAGR          : {metrics['cagr']:.2%}")
    typer.echo(f"  Sharpe        : {metrics['sharpe']:.2f}")
    typer.echo(f"  Sortino       : {metrics['sortino']:.2f}")
    typer.echo(f"  max drawdown  : {metrics['max_drawdown']:.2%}")
    typer.echo(f"  profit factor : {metrics['profit_factor']:.2f}")
    typer.echo(f"  ruined        : {metrics['ruined']}")
    typer.echo(f"  final equity  : {metrics['final_equity']:,.2f}")

    if json_out:
        write_results_json(json_out, cfg, result, metrics)
        typer.echo(f"wrote {json_out}")
    if html_out:
        write_html_report(html_out, cfg, result, metrics)
        typer.echo(f"wrote {html_out}")


def _int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


@app.command()
def optimize(
    symbol: str = typer.Option(..., help="Single symbol, e.g. EURUSD"),
    start: str = typer.Option(..., help="Inclusive start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Exclusive end date YYYY-MM-DD"),
    data_dir: str = typer.Option("data", help="Parquet store root"),
    fast: str = typer.Option("5,10,20", help="Comma-separated fast MA windows"),
    slow: str = typer.Option("30,50,100", help="Comma-separated slow MA windows"),
    n_splits: int = typer.Option(3, help="Walk-forward folds"),
    train_ratio: float = typer.Option(0.7, help="Train fraction of each fold"),
    bar_minutes: int = typer.Option(5, help="Bar interval in minutes"),
    objective: str = typer.Option("sharpe", help="Metric to maximize on train"),
    json_out: str = typer.Option(None, "--json", help="Write optimize results to this path"),
    strategy: str = typer.Option("ma", help="ma | trend | meanrev"),
    trend: str = typer.Option(None, "--trend", help="Comma-separated trend MA windows (trend strategy)"),
    exit_: str = typer.Option("none", "--exit", help="none | fixed | atr"),
    stop_pips: float = typer.Option(0.0, help="Stop in pips (exit=fixed)"),
    tp_pips: float = typer.Option(0.0, help="Take-profit in pips (exit=fixed)"),
    atr_period: int = typer.Option(14, help="ATR period (exit=atr)"),
    stop_atr: float = typer.Option(0.0, help="Stop = mult x ATR (exit=atr)"),
    tp_atr: float = typer.Option(0.0, help="TP = mult x ATR (exit=atr)"),
    lookback: str = typer.Option(None, "--lookback", help="Comma-separated lookback windows (meanrev)"),
    entry_z: str = typer.Option(None, "--entry-z", help="Comma-separated entry z-scores (meanrev)"),
    exit_z: str = typer.Option(None, "--exit-z", help="Comma-separated exit z-scores (meanrev)"),
    er_threshold: str = typer.Option(None, "--er-threshold", help="Comma-separated ER thresholds (meanrev)"),
) -> None:
    """Walk-forward optimize strategy params over stored ticks."""
    out = run_optimize(
        symbol=symbol, start=_parse_day(start), end=_parse_day(end), data_dir=data_dir,
        fast_values=_int_list(fast), slow_values=_int_list(slow),
        n_splits=n_splits, train_ratio=train_ratio, bar_minutes=bar_minutes, objective=objective,
        strategy=strategy,
        trend_values=_int_list(trend) if trend else None,
        exit_kind=exit_,
        stop_pips=stop_pips, tp_pips=tp_pips,
        atr_period=atr_period, stop_atr=stop_atr, tp_atr=tp_atr,
        lookback_values=_int_list(lookback) if lookback else None,
        entry_z_values=_float_list(entry_z) if entry_z else None,
        exit_z_values=_float_list(exit_z) if exit_z else None,
        er_threshold_values=_float_list(er_threshold) if er_threshold else None,
    )
    typer.echo(f"symbol={symbol}  objective={objective}  folds={len(out['folds'])}")
    for i, fr in enumerate(out["folds"]):
        oos = fr["test_metrics"].get(objective) if fr["test_metrics"] else None
        if oos is not None:
            typer.echo(f"  fold {i}: best={fr['best_params']}  OOS {objective}={oos:.2f}")
        else:
            typer.echo(f"  fold {i}: best={fr['best_params']}  OOS {objective}=n/a")
    typer.echo(f"  mean OOS {objective}: {out['mean_oos_objective']:.2f}")
    if json_out:
        Path(json_out).write_text(json.dumps(out, indent=2, allow_nan=False))
        typer.echo(f"wrote {json_out}")


@app.command()
def live(
    symbol: str = typer.Option(..., help="Single symbol, e.g. EURUSD"),
    mode: str = typer.Option("dry-run", help="dry-run | paper | live"),
    fast: int = typer.Option(20, help="Fast MA window (bars)"),
    slow: int = typer.Option(50, help="Slow MA window (bars)"),
    size: float = typer.Option(10_000, help="Position size (base units)"),
    host: str = typer.Option("127.0.0.1", help="IB Gateway/TWS host"),
    port: int = typer.Option(7497, help="IB port (7497 TWS paper, 4002 Gateway paper)"),
    client_id: int = typer.Option(1, help="IB client id"),
    daily_loss_limit: float = typer.Option(0.0, help="Daily-loss kill switch (0 = disabled)"),
    max_position: float = typer.Option(0.0, help="Max abs position (0 = unlimited)"),
    i_understand_live: bool = typer.Option(False, "--i-understand-live",
                                           help="Required to enable LIVE (real money)."),
) -> None:
    """Trade the MA-crossover strategy live on Interactive Brokers (paper by default)."""
    resolved = _resolve_mode(mode, confirm_live=i_understand_live)
    engine = build_live_engine(
        symbol=symbol, mode=resolved, fast=fast, slow=slow, size=size,
        host=host, port=port, client_id=client_id,
        daily_loss_limit=daily_loss_limit or float("inf"),
        max_position=max_position or float("inf"),
    )
    typer.echo(f"connecting to IB {host}:{port} (client {client_id}) in {resolved.value} mode...")
    engine.start()
    typer.echo("connected. running — Ctrl-C to stop.")
    try:
        while True:
            engine.broker._client().sleep(1)  # type: ignore[attr-defined]
    except KeyboardInterrupt:
        typer.echo("\nstopping...")
    finally:
        engine.stop()
        typer.echo("stopped; state persisted.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
