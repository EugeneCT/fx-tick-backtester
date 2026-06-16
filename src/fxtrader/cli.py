"""fxtrader command-line interface."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import typer

from fxtrader.broker.simulated import SimulatedBroker
from fxtrader.core.portfolio import Portfolio
from fxtrader.data.backfill import CoverageLedger, backfill as _backfill
from fxtrader.data.dukascopy import Downloader, _http_download, download_range
from fxtrader.data.feed import HistoricalTickFeed
from fxtrader.data.store import TickStore
from fxtrader.engine.backtest import BacktestEngine
from fxtrader.metrics.performance import compute_metrics
from fxtrader.risk.manager import RiskLimits, RiskManager
from fxtrader.strategy.examples.ma_crossover import MaCrossover

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
):
    """Run a MaCrossover backtest over stored ticks. Returns (BacktestResult, metrics)."""
    store = TickStore(data_dir)
    feed = HistoricalTickFeed(store, [symbol], start, end)
    strategy = MaCrossover(symbol=symbol, fast=fast, slow=slow, size=size)
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
    symbol: str = typer.Option(..., help="Single symbol, e.g. EURUSD"),
    start: str = typer.Option(..., help="Inclusive start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="Exclusive end date YYYY-MM-DD"),
    data_dir: str = typer.Option("data", help="Parquet store root"),
    fast: int = typer.Option(20, help="Fast MA window (bars)"),
    slow: int = typer.Option(50, help="Slow MA window (bars)"),
    size: float = typer.Option(10_000, help="Position size (base units)"),
    bar_minutes: int = typer.Option(1, help="Bar interval in minutes"),
    max_position: float = typer.Option(0.0, help="Max abs position per symbol (0 = unlimited)"),
    daily_loss_limit: float = typer.Option(0.0, help="Daily-loss kill switch (0 = disabled)"),
) -> None:
    """Backtest the MA-crossover strategy over stored ticks."""
    _result, metrics = run_backtest(
        symbol=symbol, start=_parse_day(start), end=_parse_day(end),
        data_dir=data_dir, fast=fast, slow=slow, size=size, bar_minutes=bar_minutes,
        max_position=max_position or float("inf"),
        daily_loss_limit=daily_loss_limit or float("inf"),
    )
    typer.echo(f"symbol={symbol}  bars={bar_minutes}m  MA({fast}/{slow})")
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
