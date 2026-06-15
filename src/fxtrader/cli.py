"""fxtrader command-line interface."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
