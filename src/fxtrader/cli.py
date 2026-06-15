"""fxtrader command-line interface."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer

from fxtrader.data.backfill import CoverageLedger, backfill as _backfill
from fxtrader.data.dukascopy import Downloader, _http_download, download_range
from fxtrader.data.store import TickStore

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
