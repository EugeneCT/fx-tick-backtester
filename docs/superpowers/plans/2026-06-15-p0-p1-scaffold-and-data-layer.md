# FX Trader — Phase 0 + Phase 1 (Scaffold + Data Layer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `fxtrader` Python package with its core domain types, and a data layer that downloads Dukascopy tick data into a partitioned Parquet store and replays it as time-ordered events (with optional bar resampling).

**Architecture:** A `src/`-layout package. `core/` holds pure, I/O-free domain types (`Tick`, `Bar`, `Order`, `Fill`, `Position`), a `Clock` abstraction, and a `Portfolio`. `data/` holds the Dukascopy `.bi5` decoder + fetcher, a `TickStore` over Parquet, a `HistoricalTickFeed` that merge-sorts ticks across symbols, and a `Resampler` that builds bars. A `typer` CLI exposes `download`. Everything is TDD'd with `pytest`.

**Tech Stack:** Python 3.12 (pyenv venv), polars (data), typer (CLI), pytest (tests), stdlib `lzma`/`struct`/`urllib` (Dukascopy decode/fetch). No network in unit tests — the `.bi5` decoder is tested against a synthetic buffer.

**Conventions used throughout this plan:**
- All timestamps are timezone-aware UTC `datetime` objects.
- Run tests with `pytest` from the repo root.
- Commit after every task with the message shown in the task's final step.
- Prices: Dukascopy stores integer prices scaled by a per-symbol point value (`1e-5` for EUR/USD & GBP/USD, `1e-3` for USD/JPY). Months in Dukascopy URLs are **0-indexed**.

---

## File structure (created across this plan)

```
pyproject.toml                         # Task 1
README.md                              # Task 1
.env.example                           # Task 1
src/fxtrader/__init__.py               # Task 1
src/fxtrader/core/__init__.py          # Task 2
src/fxtrader/core/types.py             # Task 2  — Tick, Bar, Order, Fill, Position, Side, OrderType
src/fxtrader/core/clock.py             # Task 3  — Clock, SimClock, WallClock
src/fxtrader/core/portfolio.py         # Task 4  — Portfolio
src/fxtrader/data/__init__.py          # Task 5
src/fxtrader/data/dukascopy.py         # Task 5/6 — decode_bi5, build_url, fetch_hour, download_range
src/fxtrader/data/store.py             # Task 7  — TickStore (Parquet + manifest)
src/fxtrader/data/feed.py              # Task 8/9 — HistoricalTickFeed, Resampler
src/fxtrader/cli.py                    # Task 10 — typer app: download
tests/...                              # one test module per source module
```

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `.env.example`
- Create: `src/fxtrader/__init__.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Create the Python 3.12 venv**

The system Python is 3.14; quant wheels lag, so pin 3.12 for this project.

Run:
```bash
pyenv install -s 3.12.7
pyenv local 3.12.7
python -m venv .venv
. .venv/bin/activate
python --version
```
Expected: `Python 3.12.7`

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "fxtrader"
version = "0.1.0"
description = "Tick-driven FX backtesting and live trading platform"
requires-python = ">=3.12"
dependencies = [
    "polars>=1.0",
    "typer>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.6"]

[project.scripts]
fxtrader = "fxtrader.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/fxtrader"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]
```

- [ ] **Step 3: Write `src/fxtrader/__init__.py`**

```python
"""fxtrader — tick-driven FX backtesting and live trading."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Write `.env.example`**

```bash
# Interactive Brokers (used in Phase 3)
IB_HOST=127.0.0.1
IB_PORT=7497          # 7497 = TWS paper, 4002 = Gateway paper
IB_CLIENT_ID=1

# Data layer
FXTRADER_DATA_DIR=./data
```

- [ ] **Step 5: Write `README.md`**

```markdown
# fxtrader

Tick-driven FX backtesting and live trading platform.

## Setup

```bash
pyenv install -s 3.12.7
pyenv local 3.12.7
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
fxtrader download --symbols EURUSD,GBPUSD,USDJPY --start 2024-01-01 --end 2024-01-31
```

Data is written to a partitioned Parquet store under `./data` (override with `--data-dir`).
```

- [ ] **Step 6: Install the package editable**

Run:
```bash
pip install -e ".[dev]"
```
Expected: ends with `Successfully installed fxtrader-0.1.0 ...`

- [ ] **Step 7: Write the smoke test `tests/test_smoke.py`**

```python
import fxtrader


def test_package_imports_and_has_version():
    assert fxtrader.__version__ == "0.1.0"
```

- [ ] **Step 8: Run the smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS (1 passed)

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml README.md .env.example src/fxtrader/__init__.py tests/test_smoke.py .python-version
git commit -m "chore: scaffold fxtrader package (P0)"
```

---

## Task 2: Core domain types

**Files:**
- Create: `src/fxtrader/core/__init__.py`
- Create: `src/fxtrader/core/types.py`
- Create: `tests/core/test_types.py`

- [ ] **Step 1: Write the failing test `tests/core/test_types.py`**

```python
from datetime import datetime, timezone

from fxtrader.core.types import Bar, Fill, Order, OrderType, Position, Side, Tick


def _ts() -> datetime:
    return datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)


def test_tick_mid_and_spread():
    tick = Tick(symbol="EURUSD", ts=_ts(), bid=1.1000, ask=1.1002)
    assert tick.mid == 1.1001
    assert round(tick.spread, 5) == 0.0002


def test_tick_is_immutable():
    tick = Tick(symbol="EURUSD", ts=_ts(), bid=1.1, ask=1.1002)
    try:
        tick.bid = 2.0  # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("Tick should be frozen")


def test_bar_fields():
    bar = Bar(symbol="EURUSD", ts=_ts(), open=1.1, high=1.2, low=1.05, close=1.15, volume=10.0)
    assert (bar.open, bar.high, bar.low, bar.close) == (1.1, 1.2, 1.05, 1.15)


def test_order_defaults_to_market():
    order = Order(symbol="EURUSD", side=Side.BUY, quantity=10_000)
    assert order.order_type is OrderType.MARKET
    assert order.limit_price is None


def test_fill_records_commission():
    fill = Fill(order_id="o1", symbol="EURUSD", side=Side.SELL,
                quantity=10_000, price=1.1001, ts=_ts(), commission=0.2)
    assert fill.commission == 0.2


def test_position_defaults_flat():
    pos = Position(symbol="EURUSD")
    assert pos.quantity == 0.0
    assert pos.avg_price == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.core'`

- [ ] **Step 3: Create `src/fxtrader/core/__init__.py`**

```python
"""Core, I/O-free domain types and primitives."""
```

- [ ] **Step 4: Write `src/fxtrader/core/types.py`**

```python
"""Pure domain types shared by every engine. No I/O, no broker imports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    ts: datetime  # tz-aware UTC
    bid: float
    ask: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    ts: datetime  # bar OPEN time, tz-aware UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class Order:
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    order_id: str | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    ts: datetime  # tz-aware UTC
    commission: float = 0.0


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: float = 0.0  # signed: + long, - short
    avg_price: float = 0.0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/core/test_types.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add src/fxtrader/core/__init__.py src/fxtrader/core/types.py tests/core/test_types.py
git commit -m "feat: core domain types (Tick, Bar, Order, Fill, Position)"
```

---

## Task 3: Clock abstraction

**Files:**
- Create: `src/fxtrader/core/clock.py`
- Create: `tests/core/test_clock.py`

- [ ] **Step 1: Write the failing test `tests/core/test_clock.py`**

```python
from datetime import datetime, timezone

import pytest

from fxtrader.core.clock import SimClock, WallClock


def test_simclock_returns_set_time():
    clock = SimClock()
    ts = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    clock.set_time(ts)
    assert clock.now() == ts


def test_simclock_raises_before_time_set():
    clock = SimClock()
    with pytest.raises(RuntimeError):
        clock.now()


def test_wallclock_is_utc_and_advances():
    clock = WallClock()
    first = clock.now()
    assert first.tzinfo is timezone.utc
    assert clock.now() >= first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_clock.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.core.clock'`

- [ ] **Step 3: Write `src/fxtrader/core/clock.py`**

```python
"""Time source abstraction: simulated (backtest) vs wall-clock (live).

Strategies read time ONLY via a Clock so the same code works in both engines.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


class SimClock:
    """Driven by event timestamps during a backtest."""

    def __init__(self) -> None:
        self._now: datetime | None = None

    def set_time(self, ts: datetime) -> None:
        self._now = ts

    def now(self) -> datetime:
        if self._now is None:
            raise RuntimeError("SimClock time not set; advance an event first")
        return self._now


class WallClock:
    """Real time, used live."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/core/test_clock.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/core/clock.py tests/core/test_clock.py
git commit -m "feat: Clock abstraction (SimClock, WallClock)"
```

---

## Task 4: Portfolio

**Files:**
- Create: `src/fxtrader/core/portfolio.py`
- Create: `tests/core/test_portfolio.py`

Accounting model: **cash-flow**. A BUY of `q` at `p` does `cash -= q*p`; a SELL does `cash += q*p`; commission always reduces cash. `equity(prices) = cash + Σ position.quantity * price`. `avg_price` and `realized_pnl` are tracked for later trade reporting.

- [ ] **Step 1: Write the failing test `tests/core/test_portfolio.py`**

```python
from datetime import datetime, timezone

from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Fill, Side


def _fill(side, qty, price, commission=0.0):
    return Fill(order_id="o", symbol="EURUSD", side=side, quantity=qty,
                price=price, ts=datetime(2024, 1, 2, tzinfo=timezone.utc),
                commission=commission)


def test_starting_cash_and_flat_equity():
    p = Portfolio(starting_cash=100_000)
    assert p.cash == 100_000
    assert p.equity({"EURUSD": 1.10}) == 100_000


def test_buy_reduces_cash_and_opens_long():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1000))
    assert p.cash == 100_000 - 11_000
    assert p.position("EURUSD").quantity == 10_000
    assert p.position("EURUSD").avg_price == 1.1000


def test_equity_marks_to_market():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1000))
    # price rises to 1.1010 -> +10 quote-ccy on 10k units
    assert round(p.equity({"EURUSD": 1.1010}), 2) == 100_010.00


def test_commission_reduces_cash():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1000, commission=0.5))
    assert p.cash == 100_000 - 11_000 - 0.5


def test_close_realizes_pnl():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1000))
    p.apply_fill(_fill(Side.SELL, 10_000, 1.1010))
    assert p.position("EURUSD").quantity == 0
    assert round(p.realized_pnl, 2) == 10.00
    assert round(p.equity({"EURUSD": 1.1010}), 2) == 100_010.00


def test_adding_to_long_updates_avg_price():
    p = Portfolio(starting_cash=100_000)
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1000))
    p.apply_fill(_fill(Side.BUY, 10_000, 1.1020))
    assert p.position("EURUSD").quantity == 20_000
    assert round(p.position("EURUSD").avg_price, 5) == 1.1010
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_portfolio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.core.portfolio'`

- [ ] **Step 3: Write `src/fxtrader/core/portfolio.py`**

```python
"""Position/cash bookkeeping via cash-flow accounting."""

from __future__ import annotations

from fxtrader.core.types import Fill, Position, Side


class Portfolio:
    def __init__(self, starting_cash: float) -> None:
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.realized_pnl = 0.0

    def position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol))

    def apply_fill(self, fill: Fill) -> None:
        signed = fill.quantity if fill.side is Side.BUY else -fill.quantity
        pos = self.position(fill.symbol)

        # Realize PnL on the portion that reduces an existing opposite position.
        if pos.quantity != 0 and (pos.quantity > 0) != (signed > 0):
            closing = min(abs(signed), abs(pos.quantity))
            direction = 1.0 if pos.quantity > 0 else -1.0
            self.realized_pnl += (fill.price - pos.avg_price) * closing * direction

        new_qty = pos.quantity + signed
        if new_qty == 0:
            pos.avg_price = 0.0
        elif pos.quantity == 0 or (pos.quantity > 0) == (signed > 0):
            # Opening or adding in the same direction -> weighted average.
            pos.avg_price = (
                pos.avg_price * abs(pos.quantity) + fill.price * abs(signed)
            ) / abs(new_qty)
        elif abs(signed) > abs(pos.quantity):
            # Flipped through zero -> remaining lot is at the fill price.
            pos.avg_price = fill.price
        # else: partial close, avg_price of the remaining lot is unchanged.

        pos.quantity = new_qty
        self.cash -= signed * fill.price
        self.cash -= fill.commission
        self.fills.append(fill)

    def equity(self, prices: dict[str, float]) -> float:
        mark = sum(p.quantity * prices[p.symbol] for p in self.positions.values() if p.quantity)
        return self.cash + mark
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/core/test_portfolio.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/core/portfolio.py tests/core/test_portfolio.py
git commit -m "feat: Portfolio with cash-flow accounting and mark-to-market equity"
```

---

## Task 5: Dukascopy `.bi5` decoder

**Files:**
- Create: `src/fxtrader/data/__init__.py`
- Create: `src/fxtrader/data/dukascopy.py`
- Create: `tests/data/test_dukascopy_decode.py`

A `.bi5` file is LZMA-compressed; decompressed it is a sequence of 20-byte big-endian records `(ms_offset: uint32, ask: uint32, bid: uint32, ask_vol: float32, bid_vol: float32)`. Integer prices are scaled by the symbol's point value. The test builds a synthetic buffer — **no network**.

- [ ] **Step 1: Write the failing test `tests/data/test_dukascopy_decode.py`**

```python
import lzma
import struct
from datetime import datetime, timezone

from fxtrader.data.dukascopy import POINT_VALUE, build_url, decode_bi5


def _make_bi5(records):
    raw = b"".join(struct.pack(">IIIff", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def test_decode_bi5_eurusd():
    hour = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    # (ms_offset, ask_int, bid_int, ask_vol, bid_vol)
    records = [
        (0, 110020, 110000, 1.5, 2.0),
        (1500, 110030, 110010, 0.5, 0.7),
    ]
    ticks = decode_bi5(_make_bi5(records), hour, "EURUSD")

    assert len(ticks) == 2
    assert ticks[0].symbol == "EURUSD"
    assert ticks[0].ts == datetime(2024, 1, 2, 9, 0, 0, tzinfo=timezone.utc)
    assert round(ticks[0].ask, 5) == 1.10020
    assert round(ticks[0].bid, 5) == 1.10000
    assert ticks[0].ask_volume == 1.5
    assert ticks[1].ts == datetime(2024, 1, 2, 9, 0, 1, 500_000, tzinfo=timezone.utc)


def test_decode_bi5_usdjpy_uses_3_decimal_scaling():
    hour = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    ticks = decode_bi5(_make_bi5([(0, 145123, 145100, 1.0, 1.0)]), hour, "USDJPY")
    assert round(ticks[0].ask, 3) == 145.123
    assert round(ticks[0].bid, 3) == 145.100


def test_decode_empty_buffer_returns_no_ticks():
    assert decode_bi5(lzma.compress(b"", format=lzma.FORMAT_ALONE), datetime.now(timezone.utc), "EURUSD") == []


def test_point_values_present():
    assert POINT_VALUE["EURUSD"] == 1e-5
    assert POINT_VALUE["USDJPY"] == 1e-3


def test_build_url_uses_zero_indexed_month():
    url = build_url("EURUSD", datetime(2024, 3, 5, 9, tzinfo=timezone.utc))
    # March -> month index 02
    assert url == (
        "https://datafeed.dukascopy.com/datafeed/EURUSD/2024/02/05/09h_ticks.bi5"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_dukascopy_decode.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.data'`

- [ ] **Step 3: Create `src/fxtrader/data/__init__.py`**

```python
"""Data layer: Dukascopy download, Parquet store, historical feed."""
```

- [ ] **Step 4: Write `src/fxtrader/data/dukascopy.py` (decoder portion)**

```python
"""Dukascopy tick data: .bi5 decoding and URL construction.

.bi5 = LZMA-compressed stream of 20-byte big-endian records:
    (ms_offset: uint32, ask: uint32, bid: uint32, ask_vol: float32, bid_vol: float32)
Integer prices are scaled by the symbol's point value. Dukascopy month is 0-indexed.
"""

from __future__ import annotations

import lzma
import struct
from datetime import datetime, timedelta

from fxtrader.core.types import Tick

POINT_VALUE: dict[str, float] = {
    "EURUSD": 1e-5,
    "GBPUSD": 1e-5,
    "USDJPY": 1e-3,
}

_RECORD = struct.Struct(">IIIff")
_BASE_URL = "https://datafeed.dukascopy.com/datafeed"


def build_url(symbol: str, hour: datetime) -> str:
    """Dukascopy datafeed URL for a single hour. Month is 0-indexed."""
    return (
        f"{_BASE_URL}/{symbol}/{hour.year:04d}/{hour.month - 1:02d}/"
        f"{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"
    )


def decode_bi5(raw: bytes, hour_start: datetime, symbol: str) -> list[Tick]:
    """Decode one hour's compressed .bi5 payload into Ticks."""
    if not raw:
        return []
    point = POINT_VALUE[symbol]
    data = lzma.decompress(raw)
    ticks: list[Tick] = []
    for ms, ask_i, bid_i, ask_v, bid_v in _RECORD.iter_unpack(data):
        ticks.append(
            Tick(
                symbol=symbol,
                ts=hour_start + timedelta(milliseconds=ms),
                bid=bid_i * point,
                ask=ask_i * point,
                bid_volume=bid_v,
                ask_volume=ask_v,
            )
        )
    return ticks
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/data/test_dukascopy_decode.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add src/fxtrader/data/__init__.py src/fxtrader/data/dukascopy.py tests/data/test_dukascopy_decode.py
git commit -m "feat: Dukascopy .bi5 decoder and URL builder"
```

---

## Task 6: Dukascopy hour fetch + range download

**Files:**
- Modify: `src/fxtrader/data/dukascopy.py` (add `fetch_hour`, `download_range`)
- Create: `tests/data/test_dukascopy_fetch.py`

`fetch_hour` downloads one hour and decodes it; a 404/empty body means "no ticks" (weekend/holiday), not an error. The downloader is injected so tests never touch the network.

- [ ] **Step 1: Write the failing test `tests/data/test_dukascopy_fetch.py`**

```python
import lzma
import struct
from datetime import datetime, timezone

from fxtrader.data.dukascopy import download_range, fetch_hour


def _payload(records):
    raw = b"".join(struct.pack(">IIIff", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def test_fetch_hour_decodes_downloaded_payload():
    payload = _payload([(0, 110020, 110000, 1.0, 1.0)])
    calls = []

    def fake_download(url: str) -> bytes:
        calls.append(url)
        return payload

    hour = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    ticks = fetch_hour("EURUSD", hour, downloader=fake_download)

    assert len(ticks) == 1
    assert calls[0].endswith("/EURUSD/2024/00/02/09h_ticks.bi5")


def test_fetch_hour_treats_empty_as_no_ticks():
    ticks = fetch_hour(
        "EURUSD",
        datetime(2024, 1, 6, 9, 0, tzinfo=timezone.utc),  # a Saturday
        downloader=lambda url: b"",
    )
    assert ticks == []


def test_download_range_iterates_each_hour():
    seen = []

    def fake_download(url: str) -> bytes:
        seen.append(url)
        return _payload([(0, 110020, 110000, 1.0, 1.0)])

    start = datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)  # exclusive
    ticks = download_range("EURUSD", start, end, downloader=fake_download)

    assert len(seen) == 3            # hours 09, 10, 11
    assert len(ticks) == 3           # one tick per hour
    assert ticks == sorted(ticks, key=lambda t: t.ts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_dukascopy_fetch.py -v`
Expected: FAIL with `ImportError: cannot import name 'fetch_hour'`

- [ ] **Step 3: Append fetch logic to `src/fxtrader/data/dukascopy.py`**

Add these imports at the top of the file (merge with existing imports):

```python
import urllib.error
import urllib.request
from collections.abc import Callable
```

Append at the end of the file:

```python
Downloader = Callable[[str], bytes]


def _http_download(url: str, *, timeout: float = 30.0) -> bytes:
    """Default downloader. Returns empty bytes for a 404 (no data that hour)."""
    req = urllib.request.Request(url, headers={"User-Agent": "fxtrader/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return b""
        raise


def fetch_hour(
    symbol: str,
    hour: datetime,
    *,
    downloader: Downloader = _http_download,
) -> list[Tick]:
    """Download and decode a single hour of ticks."""
    raw = downloader(build_url(symbol, hour))
    return decode_bi5(raw, hour, symbol)


def download_range(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    downloader: Downloader = _http_download,
) -> list[Tick]:
    """Download ticks for [start, end) one hour at a time, in time order."""
    ticks: list[Tick] = []
    cursor = start.replace(minute=0, second=0, microsecond=0)
    while cursor < end:
        ticks.extend(fetch_hour(symbol, cursor, downloader=downloader))
        cursor += timedelta(hours=1)
    return ticks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_dukascopy_fetch.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the whole suite**

Run: `pytest -q`
Expected: PASS (all tests so far)

- [ ] **Step 6: Commit**

```bash
git add src/fxtrader/data/dukascopy.py tests/data/test_dukascopy_fetch.py
git commit -m "feat: Dukascopy hour fetch and range download (injectable downloader)"
```

---

## Task 7: TickStore (Parquet + manifest)

**Files:**
- Create: `src/fxtrader/data/store.py`
- Create: `tests/data/test_store.py`

Layout: `data/<SYMBOL>/<YEAR>/<MONTH>.parquet`, columns `[symbol, ts, bid, ask, bid_volume, ask_volume]`. `write_ticks` merges with any existing month file (concat, sort by `ts`, dedup). A `manifest.json` at the store root records, per symbol, the min/max stored timestamp.

- [ ] **Step 1: Write the failing test `tests/data/test_store.py`**

```python
import json
from datetime import datetime, timezone

from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore


def _tick(ms_min, bid, ask):
    return Tick(
        symbol="EURUSD",
        ts=datetime(2024, 1, 2, 9, ms_min, tzinfo=timezone.utc),
        bid=bid,
        ask=ask,
        bid_volume=1.0,
        ask_volume=1.0,
    )


def test_write_then_read_roundtrip(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [_tick(0, 1.1000, 1.1002), _tick(1, 1.1001, 1.1003)])

    df = store.read_ticks(
        "EURUSD",
        datetime(2024, 1, 2, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 3, 0, tzinfo=timezone.utc),
    )
    assert df.height == 2
    assert df["bid"].to_list() == [1.1000, 1.1001]


def test_read_filters_by_time_range(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [_tick(0, 1.1, 1.1002), _tick(30, 1.2, 1.2002)])

    df = store.read_ticks(
        "EURUSD",
        datetime(2024, 1, 2, 9, 15, tzinfo=timezone.utc),
        datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
    )
    assert df.height == 1
    assert df["bid"].to_list() == [1.2]


def test_write_is_idempotent_and_sorted(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [_tick(1, 1.1001, 1.1003)])
    store.write_ticks("EURUSD", [_tick(0, 1.1000, 1.1002), _tick(1, 1.1001, 1.1003)])

    df = store.read_ticks(
        "EURUSD",
        datetime(2024, 1, 2, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 3, 0, tzinfo=timezone.utc),
    )
    assert df.height == 2  # duplicate dropped
    assert df["ts"].is_sorted()


def test_manifest_tracks_coverage(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [_tick(0, 1.1, 1.1002), _tick(30, 1.2, 1.2002)])

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["EURUSD"]["min_ts"] == "2024-01-02T09:00:00+00:00"
    assert manifest["EURUSD"]["max_ts"] == "2024-01-02T09:30:00+00:00"


def test_read_missing_symbol_returns_empty(tmp_path):
    store = TickStore(tmp_path)
    df = store.read_ticks(
        "GBPUSD",
        datetime(2024, 1, 2, tzinfo=timezone.utc),
        datetime(2024, 1, 3, tzinfo=timezone.utc),
    )
    assert df.height == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.data.store'`

- [ ] **Step 3: Write `src/fxtrader/data/store.py`**

```python
"""Parquet-backed tick store, partitioned by symbol/year/month."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from fxtrader.core.types import Tick

_COLUMNS = ["symbol", "ts", "bid", "ask", "bid_volume", "ask_volume"]
_EMPTY = pl.DataFrame(
    schema={
        "symbol": pl.String,
        "ts": pl.Datetime(time_unit="us", time_zone="UTC"),
        "bid": pl.Float64,
        "ask": pl.Float64,
        "bid_volume": pl.Float64,
        "ask_volume": pl.Float64,
    }
)


class TickStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"

    def _month_path(self, symbol: str, year: int, month: int) -> Path:
        return self.root / symbol / f"{year:04d}" / f"{month:02d}.parquet"

    def _to_df(self, ticks: list[Tick]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "symbol": [t.symbol for t in ticks],
                "ts": [t.ts for t in ticks],
                "bid": [t.bid for t in ticks],
                "ask": [t.ask for t in ticks],
                "bid_volume": [t.bid_volume for t in ticks],
                "ask_volume": [t.ask_volume for t in ticks],
            },
            schema=_EMPTY.schema,
        )

    def write_ticks(self, symbol: str, ticks: list[Tick]) -> None:
        if not ticks:
            return
        df = self._to_df(ticks)
        for (year, month), group in df.group_by(
            df["ts"].dt.year().alias("y"), df["ts"].dt.month().alias("m")
        ):
            path = self._month_path(symbol, year, month)
            path.parent.mkdir(parents=True, exist_ok=True)
            merged = group.drop("y", "m") if "y" in group.columns else group
            if path.exists():
                merged = pl.concat([pl.read_parquet(path), merged])
            merged = merged.unique(subset=["ts"], keep="last").sort("ts")
            merged.select(_COLUMNS).write_parquet(path)
        self._update_manifest(symbol)

    def read_ticks(self, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
        symbol_dir = self.root / symbol
        if not symbol_dir.exists():
            return _EMPTY.clone()
        files = sorted(symbol_dir.glob("*/*.parquet"))
        if not files:
            return _EMPTY.clone()
        return (
            pl.read_parquet(files)
            .filter((pl.col("ts") >= start) & (pl.col("ts") < end))
            .sort("ts")
        )

    def _update_manifest(self, symbol: str) -> None:
        manifest: dict = {}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
        files = sorted((self.root / symbol).glob("*/*.parquet"))
        df = pl.read_parquet(files)
        manifest[symbol] = {
            "min_ts": df["ts"].min().isoformat(),
            "max_ts": df["ts"].max().isoformat(),
            "rows": df.height,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_store.py -v`
Expected: PASS (5 passed)

Note: if `group_by` tuple-unpacking raises on your polars version, replace the loop with `for keys, group in df.group_by(...)` and unpack `year, month = keys`. Verify the test passes either way.

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/data/store.py tests/data/test_store.py
git commit -m "feat: TickStore (partitioned Parquet write/read + manifest)"
```

---

## Task 8: HistoricalTickFeed

**Files:**
- Create: `src/fxtrader/data/feed.py`
- Create: `tests/data/test_feed.py`

The feed reads one or more symbols from a `TickStore` and yields `Tick` events in global time order (merge across symbols by `ts`).

- [ ] **Step 1: Write the failing test `tests/data/test_feed.py`**

```python
from datetime import datetime, timezone

from fxtrader.core.types import Tick
from fxtrader.data.feed import HistoricalTickFeed
from fxtrader.data.store import TickStore


def _tick(symbol, minute, bid, ask):
    return Tick(symbol=symbol, ts=datetime(2024, 1, 2, 9, minute, tzinfo=timezone.utc),
                bid=bid, ask=ask)


def test_feed_yields_single_symbol_in_order(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [_tick("EURUSD", 1, 1.1001, 1.1003),
                                 _tick("EURUSD", 0, 1.1000, 1.1002)])

    feed = HistoricalTickFeed(store, ["EURUSD"],
                              datetime(2024, 1, 2, tzinfo=timezone.utc),
                              datetime(2024, 1, 3, tzinfo=timezone.utc))
    out = list(feed)
    assert [t.ts.minute for t in out] == [0, 1]
    assert all(isinstance(t, Tick) for t in out)


def test_feed_merges_symbols_by_time(tmp_path):
    store = TickStore(tmp_path)
    store.write_ticks("EURUSD", [_tick("EURUSD", 0, 1.10, 1.1002),
                                 _tick("EURUSD", 2, 1.10, 1.1002)])
    store.write_ticks("GBPUSD", [_tick("GBPUSD", 1, 1.27, 1.2702)])

    feed = HistoricalTickFeed(store, ["EURUSD", "GBPUSD"],
                              datetime(2024, 1, 2, tzinfo=timezone.utc),
                              datetime(2024, 1, 3, tzinfo=timezone.utc))
    out = [(t.symbol, t.ts.minute) for t in feed]
    assert out == [("EURUSD", 0), ("GBPUSD", 1), ("EURUSD", 2)]


def test_feed_empty_store_yields_nothing(tmp_path):
    feed = HistoricalTickFeed(TickStore(tmp_path), ["EURUSD"],
                              datetime(2024, 1, 2, tzinfo=timezone.utc),
                              datetime(2024, 1, 3, tzinfo=timezone.utc))
    assert list(feed) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_feed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.data.feed'`

- [ ] **Step 3: Write `src/fxtrader/data/feed.py` (feed portion)**

```python
"""Historical tick replay and on-the-fly bar resampling."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import polars as pl

from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore


class HistoricalTickFeed:
    """Yields Ticks from a TickStore in global time order across symbols."""

    def __init__(
        self,
        store: TickStore,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> None:
        self._store = store
        self._symbols = symbols
        self._start = start
        self._end = end

    def __iter__(self) -> Iterator[Tick]:
        frames = [
            self._store.read_ticks(sym, self._start, self._end)
            for sym in self._symbols
        ]
        non_empty = [f for f in frames if f.height]
        if not non_empty:
            return iter(())
        merged = pl.concat(non_empty).sort("ts")
        return (
            Tick(
                symbol=row["symbol"],
                ts=row["ts"],
                bid=row["bid"],
                ask=row["ask"],
                bid_volume=row["bid_volume"],
                ask_volume=row["ask_volume"],
            )
            for row in merged.iter_rows(named=True)
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_feed.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/data/feed.py tests/data/test_feed.py
git commit -m "feat: HistoricalTickFeed (time-ordered multi-symbol replay)"
```

---

## Task 9: Resampler (ticks → bars)

**Files:**
- Modify: `src/fxtrader/data/feed.py` (add `Resampler`)
- Create: `tests/data/test_resampler.py`

`Resampler(interval)` consumes ticks for one symbol and emits a completed `Bar` (built from mid prices) whenever a tick crosses into a new time bucket. `flush()` returns the final in-progress bar. Bars are stamped with the bucket **open** time.

- [ ] **Step 1: Write the failing test `tests/data/test_resampler.py`**

```python
from datetime import datetime, timedelta, timezone

from fxtrader.core.types import Tick
from fxtrader.data.feed import Resampler


def _tick(second, bid, ask):
    return Tick(symbol="EURUSD",
                ts=datetime(2024, 1, 2, 9, 0, second, tzinfo=timezone.utc),
                bid=bid, ask=ask)


def test_resampler_emits_bar_on_bucket_rollover():
    r = Resampler(timedelta(minutes=1))
    # all within minute 09:00
    assert r.update(_tick(0, 1.1000, 1.1002)) is None   # mid 1.1001
    assert r.update(_tick(20, 1.1010, 1.1012)) is None   # mid 1.1011 (high)
    assert r.update(_tick(40, 1.0990, 1.0992)) is None   # mid 1.0991 (low)
    # crossing into 09:01 closes the 09:00 bar
    bar = r.update(Tick(symbol="EURUSD",
                        ts=datetime(2024, 1, 2, 9, 1, 0, tzinfo=timezone.utc),
                        bid=1.1000, ask=1.1002))
    assert bar is not None
    assert bar.ts == datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc)
    assert round(bar.open, 5) == 1.1001
    assert round(bar.high, 5) == 1.1011
    assert round(bar.low, 5) == 1.0991
    assert round(bar.close, 5) == 1.0991


def test_resampler_flush_returns_last_partial_bar():
    r = Resampler(timedelta(minutes=1))
    r.update(_tick(0, 1.1000, 1.1002))
    r.update(_tick(30, 1.1010, 1.1012))
    bar = r.flush()
    assert bar is not None
    assert round(bar.open, 5) == 1.1001
    assert round(bar.close, 5) == 1.1011
    assert r.flush() is None  # nothing left


def test_resampler_counts_ticks_as_volume():
    r = Resampler(timedelta(minutes=1))
    r.update(_tick(0, 1.1, 1.1002))
    r.update(_tick(10, 1.1, 1.1002))
    bar = r.flush()
    assert bar.volume == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_resampler.py -v`
Expected: FAIL with `ImportError: cannot import name 'Resampler'`

- [ ] **Step 3: Append `Resampler` to `src/fxtrader/data/feed.py`**

Add to the imports at the top of the file (merge with existing):

```python
from datetime import timedelta

from fxtrader.core.types import Bar
```

Append at the end of the file:

```python
class Resampler:
    """Builds time bars from a single symbol's ticks, using mid price."""

    def __init__(self, interval: timedelta) -> None:
        self._interval = interval
        self._bucket: datetime | None = None
        self._symbol: str | None = None
        self._o = self._h = self._l = self._c = 0.0
        self._volume = 0

    def _floor(self, ts: datetime) -> datetime:
        epoch = ts.timestamp()
        step = self._interval.total_seconds()
        floored = epoch - (epoch % step)
        return datetime.fromtimestamp(floored, tz=ts.tzinfo)

    def update(self, tick: Tick) -> Bar | None:
        bucket = self._floor(tick.ts)
        emitted: Bar | None = None
        if self._bucket is not None and bucket != self._bucket:
            emitted = self._build()
            self._bucket = None
        mid = tick.mid
        if self._bucket is None:
            self._bucket = bucket
            self._symbol = tick.symbol
            self._o = self._h = self._l = self._c = mid
            self._volume = 1
        else:
            self._h = max(self._h, mid)
            self._l = min(self._l, mid)
            self._c = mid
            self._volume += 1
        return emitted

    def flush(self) -> Bar | None:
        if self._bucket is None:
            return None
        bar = self._build()
        self._bucket = None
        return bar

    def _build(self) -> Bar:
        assert self._bucket is not None and self._symbol is not None
        return Bar(
            symbol=self._symbol,
            ts=self._bucket,
            open=self._o,
            high=self._h,
            low=self._l,
            close=self._c,
            volume=float(self._volume),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/data/test_resampler.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/data/feed.py tests/data/test_resampler.py
git commit -m "feat: Resampler (ticks -> OHLC bars on mid price)"
```

---

## Task 10: CLI `download` command

**Files:**
- Create: `src/fxtrader/cli.py`
- Create: `tests/test_cli.py`

`fxtrader download` parses `--symbols`, `--start`, `--end`, `--data-dir`, downloads each symbol via `download_range`, and writes to a `TickStore`. A `--downloader` seam is not exposed on the CLI; instead the test calls the underlying `run_download` helper with a fake downloader so no network is hit.

- [ ] **Step 1: Write the failing test `tests/test_cli.py`**

```python
import lzma
import struct
from datetime import datetime, timezone

from fxtrader.cli import run_download
from fxtrader.data.store import TickStore


def _payload(records):
    raw = b"".join(struct.pack(">IIIff", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def test_run_download_populates_store(tmp_path):
    payload = _payload([(0, 110020, 110000, 1.0, 1.0)])

    summary = run_download(
        symbols=["EURUSD"],
        start=datetime(2024, 1, 2, 9, tzinfo=timezone.utc),
        end=datetime(2024, 1, 2, 11, tzinfo=timezone.utc),  # 2 hours
        data_dir=tmp_path,
        downloader=lambda url: payload,
    )

    assert summary["EURUSD"] == 2  # one tick per hour
    store = TickStore(tmp_path)
    df = store.read_ticks("EURUSD",
                          datetime(2024, 1, 2, tzinfo=timezone.utc),
                          datetime(2024, 1, 3, tzinfo=timezone.utc))
    assert df.height == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fxtrader.cli'`

- [ ] **Step 3: Write `src/fxtrader/cli.py`**

```python
"""fxtrader command-line interface."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer

from fxtrader.data.dukascopy import Downloader, _http_download, download_range
from fxtrader.data.store import TickStore

app = typer.Typer(add_completion=False, help="fxtrader CLI")


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Verify the CLI is wired (help renders)**

Run: `fxtrader download --help`
Expected: usage text listing `--symbols`, `--start`, `--end`, `--data-dir`.

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: PASS (all tests, ~26 passing)

- [ ] **Step 7: Commit**

```bash
git add src/fxtrader/cli.py tests/test_cli.py
git commit -m "feat: CLI download command wired to TickStore"
```

---

## Task 11: End-to-end smoke against real Dukascopy (manual, network)

**Files:**
- Create: `tests/integration/test_dukascopy_live.py`

This is a **network** test — opt-in, skipped by default so CI/unit runs stay offline.

- [ ] **Step 1: Write `tests/integration/test_dukascopy_live.py`**

```python
import os
from datetime import datetime, timezone

import pytest

from fxtrader.data.dukascopy import fetch_hour

pytestmark = pytest.mark.skipif(
    os.environ.get("FXTRADER_NETWORK_TESTS") != "1",
    reason="set FXTRADER_NETWORK_TESTS=1 to run live Dukascopy download",
)


def test_fetch_one_real_eurusd_hour():
    # A Tuesday mid-session hour that reliably has ticks.
    hour = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
    ticks = fetch_hour("EURUSD", hour)
    assert len(ticks) > 100
    assert all(t.ask >= t.bid for t in ticks)
    assert all(1.0 < t.bid < 1.3 for t in ticks)  # sane EUR/USD range
```

- [ ] **Step 2: Run it manually to confirm the real feed works**

Run: `FXTRADER_NETWORK_TESTS=1 pytest tests/integration/test_dukascopy_live.py -v`
Expected: PASS (downloads one real hour; asserts sane ticks). Without the env var it is skipped.

- [ ] **Step 3: Confirm the default suite still skips it**

Run: `pytest -q`
Expected: PASS with 1 skipped.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_dukascopy_live.py
git commit -m "test: opt-in live Dukascopy download smoke test"
```

---

## Self-review

**Spec coverage (P0 + P1 portion of the design spec):**
- §3 Tech stack — Python 3.12 venv (T1), polars (T7–T9), typer (T10), stdlib lzma/struct/urllib (T5–T6). ✓
- §4.2 Core abstractions — `Tick`/`Bar`/`Order`/`Fill`/`Position` (T2), `Clock` (T3), `Portfolio` (T4). ✓ (`DataFeed`/`Broker`/`RiskManager` interfaces are P2 — out of scope here, by design.)
- §5.1 Data layer — `.bi5` decoder (T5), fetch + range download (T6), `TickStore` Parquet + manifest (T7), `HistoricalTickFeed` (T8), `Resampler` (T9). ✓
- §5.7 CLI `download` (T10). ✓
- §9 risks — month 0-indexing tested (T5), empty/weekend handled (T6), Python 3.12 pin (T1), real-feed validation (T11). ✓
- §8 testing — unit per module + round-trip + opt-in live smoke. ✓ (golden-path backtest integration test is P2.)

**Placeholder scan:** none — every code and test step contains complete, runnable content.

**Type/signature consistency:** `Tick(symbol, ts, bid, ask, bid_volume, ask_volume)`, `Bar(symbol, ts, open, high, low, close, volume)`, `Fill(order_id, symbol, side, quantity, price, ts, commission)`, `Side.BUY/SELL`, `decode_bi5(raw, hour_start, symbol)`, `build_url(symbol, hour)`, `fetch_hour(symbol, hour, *, downloader)`, `download_range(symbol, start, end, *, downloader)`, `TickStore.write_ticks/read_ticks`, `HistoricalTickFeed(store, symbols, start, end)`, `Resampler(interval).update/flush`, `run_download(symbols, start, end, data_dir, downloader)` — used consistently across all tasks. ✓

**Scope:** This plan stops at a working, tested data layer + CLI. P2 (backtester), P3 (live), P4 (research agent) are deliberately separate plans.
