# Price-Action + Trend strategy (`pa`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 6th strategy `pa` — four famous price-action triggers (pin bar, engulfing, inside-bar break, swing-structure break) gated by a same-timeframe long-MA trend filter, with stops/targets supplied by the existing `AtrMultiple` exit (calibrated later, in Phase-B research).

**Architecture:** Pure pattern detectors (`patterns.py`) → a thin bar-based strategy (`price_action.py`) that combines the active detector with the long-MA trend gate and the EXIT-FIRST stop/TP/time block copied from `orb`/`trend` → registry + CLI plumbing following the existing per-strategy pattern. No CLI generalization (consolidation cycle 2 stays deferred).

**Tech Stack:** Python 3.12 (`.venv`), pytest, Typer CLI, Polars-backed tick store. Invoke tooling as `.venv/bin/python` / `.venv/bin/pytest` / `.venv/bin/fxtrader`.

## Global Constraints

- Python pinned to 3.12 in `.venv`; ALWAYS invoke as `.venv/bin/pytest` / `.venv/bin/python` (shell `source` does not persist across tool calls).
- Baseline test suite is **221 passed, 2 skipped** — keep green; only add tests.
- Strategy code is pure logic: no I/O, no direct portfolio mutation; read position via `self.portfolio.position(symbol).quantity`; trade via `self.order(...)`.
- EXIT-FIRST, bar-granular, **stop-first same-bar rule** (check stop before TP) — copy the exact structure from `OpeningRangeBreakout._enter/_flatten` and `TrendFilteredCrossover.on_bar`.
- Single-symbol strategy: ignore bars where `bar.symbol != self.symbol`.
- Patterns are tested **independently**; `pattern` is fixed per backtest (chosen, not gridded).
- `Bar(symbol, ts, open, high, low, close, volume)` — frozen dataclass, positional fields in that order. `Side` is `Side.BUY` / `Side.SELL`.
- Do NOT touch `src/` during Phase B; do NOT wire `--json`/`--html` into `pa` (those are MA-only by design).

---

### Task 1: Pattern detectors (`strategy/patterns.py`)

Pure functions, no state, no I/O — mirror `strategy/indicators.py`. Each returns `-1` / `0` / `+1` (or `bool` for the inside-bar predicate).

**Files:**
- Create: `src/fxtrader/strategy/patterns.py`
- Test: `tests/strategy/test_patterns.py`

**Interfaces:**
- Consumes: `fxtrader.core.types.Bar`.
- Produces:
  - `pin_bar(bar: Bar, *, wick_ratio: float, body_frac: float) -> int`
  - `engulfing(prev: Bar, bar: Bar) -> int`
  - `is_inside_bar(prev: Bar, bar: Bar) -> bool`
  - `swing_break(bars: Sequence[Bar], lookback: int) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategy/test_patterns.py
from datetime import datetime, timezone

from fxtrader.core.types import Bar
from fxtrader.strategy.patterns import pin_bar, engulfing, is_inside_bar, swing_break

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _bar(o, h, l, c):
    return Bar("EURUSD", T0, o, h, l, c, 1.0)


# --- pin_bar ---
def test_bullish_pin_long_lower_wick():
    # rng=0.01, tiny body near the top, long lower wick → +1
    b = _bar(1.0090, 1.0100, 1.0000, 1.0095)
    assert pin_bar(b, wick_ratio=2.0, body_frac=0.34) == 1


def test_bearish_pin_long_upper_wick():
    b = _bar(1.0010, 1.0100, 1.0000, 1.0005)
    assert pin_bar(b, wick_ratio=2.0, body_frac=0.34) == -1


def test_big_body_is_not_a_pin():
    b = _bar(1.0000, 1.0100, 0.9990, 1.0090)  # body dominates the range
    assert pin_bar(b, wick_ratio=2.0, body_frac=0.34) == 0


def test_zero_range_bar_is_not_a_pin():
    b = _bar(1.0000, 1.0000, 1.0000, 1.0000)
    assert pin_bar(b, wick_ratio=2.0, body_frac=0.34) == 0


# --- engulfing ---
def test_bullish_engulfing():
    prev = _bar(1.0050, 1.0055, 0.9995, 1.0000)  # bearish
    cur = _bar(0.9990, 1.0065, 0.9985, 1.0060)   # bullish, body engulfs prev body
    assert engulfing(prev, cur) == 1


def test_bearish_engulfing():
    prev = _bar(1.0000, 1.0055, 0.9995, 1.0050)  # bullish
    cur = _bar(1.0060, 1.0065, 0.9985, 0.9990)   # bearish, body engulfs prev body
    assert engulfing(prev, cur) == -1


def test_non_engulfing_returns_zero():
    prev = _bar(1.0050, 1.0055, 0.9995, 1.0000)  # bearish
    cur = _bar(1.0010, 1.0045, 1.0005, 1.0040)   # bullish but does not exceed prev.open
    assert engulfing(prev, cur) == 0


# --- is_inside_bar ---
def test_inside_bar_true():
    prev = _bar(1.0000, 1.0100, 1.0000, 1.0050)
    cur = _bar(1.0030, 1.0080, 1.0020, 1.0040)   # contained within prev range
    assert is_inside_bar(prev, cur) is True


def test_inside_bar_false_when_breaks_high():
    prev = _bar(1.0000, 1.0100, 1.0000, 1.0050)
    cur = _bar(1.0030, 1.0120, 1.0020, 1.0040)   # high exceeds prev high
    assert is_inside_bar(prev, cur) is False


# --- swing_break ---
def test_swing_break_up():
    bars = [_bar(1.00, 1.01, 1.00, 1.005) for _ in range(3)]
    bars.append(_bar(1.01, 1.025, 1.01, 1.02))   # close above swing high (1.01)
    assert swing_break(bars, lookback=3) == 1


def test_swing_break_down():
    bars = [_bar(1.00, 1.01, 1.00, 1.005) for _ in range(3)]
    bars.append(_bar(1.00, 1.00, 0.985, 0.99))   # close below swing low (1.00)
    assert swing_break(bars, lookback=3) == -1


def test_swing_break_none_inside_range():
    bars = [_bar(1.00, 1.01, 1.00, 1.005) for _ in range(3)]
    bars.append(_bar(1.005, 1.008, 1.002, 1.006))
    assert swing_break(bars, lookback=3) == 0


def test_swing_break_insufficient_history():
    bars = [_bar(1.00, 1.01, 1.00, 1.005) for _ in range(2)]
    assert swing_break(bars, lookback=3) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/strategy/test_patterns.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fxtrader.strategy.patterns'`.

- [ ] **Step 3: Write the implementation**

```python
# src/fxtrader/strategy/patterns.py
"""Pure price-action pattern detectors. No I/O, no state. Each returns -1 / 0 / +1."""

from __future__ import annotations

from collections.abc import Sequence

from fxtrader.core.types import Bar


def pin_bar(bar: Bar, *, wick_ratio: float, body_frac: float) -> int:
    """Rejection candle: small body, one dominant wick, close in the far third.

    +1 bullish (long lower wick, close in upper third); -1 bearish (mirror); else 0.
    """
    rng = bar.high - bar.low
    if rng <= 0:
        return 0
    body = abs(bar.close - bar.open)
    if body > body_frac * rng:
        return 0  # body too large to be a pin
    upper = bar.high - max(bar.open, bar.close)
    lower = min(bar.open, bar.close) - bar.low
    eps = rng * 1e-9
    if lower >= wick_ratio * max(body, eps) and bar.close >= bar.low + (2.0 / 3.0) * rng:
        return 1
    if upper >= wick_ratio * max(body, eps) and bar.close <= bar.low + (1.0 / 3.0) * rng:
        return -1
    return 0


def engulfing(prev: Bar, bar: Bar) -> int:
    """Current real body fully engulfs the prior real body and flips direction."""
    prev_bear = prev.close < prev.open
    prev_bull = prev.close > prev.open
    cur_bull = bar.close > bar.open
    cur_bear = bar.close < bar.open
    if prev_bear and cur_bull and bar.close >= prev.open and bar.open <= prev.close:
        return 1
    if prev_bull and cur_bear and bar.close <= prev.open and bar.open >= prev.close:
        return -1
    return 0


def is_inside_bar(prev: Bar, bar: Bar) -> bool:
    """True when `bar`'s range is fully contained within `prev`'s range."""
    return bar.high <= prev.high and bar.low >= prev.low


def swing_break(bars: Sequence[Bar], lookback: int) -> int:
    """Close beyond the swing high/low of the `lookback` bars preceding the current bar.

    +1 close > prior swing high; -1 close < prior swing low; else 0.
    """
    if len(bars) < lookback + 1:
        return 0
    window = bars[-(lookback + 1):-1]
    cur = bars[-1]
    swing_high = max(b.high for b in window)
    swing_low = min(b.low for b in window)
    if cur.close > swing_high:
        return 1
    if cur.close < swing_low:
        return -1
    return 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/strategy/test_patterns.py -q`
Expected: PASS (13 passed).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/patterns.py tests/strategy/test_patterns.py
git commit -m "feat(strategy): price-action pattern detectors (pin/engulf/inside/swing-break)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `PriceActionTrend` strategy (`strategy/examples/price_action.py`)

**Files:**
- Create: `src/fxtrader/strategy/examples/price_action.py`
- Test: `tests/strategy/test_price_action.py`

**Interfaces:**
- Consumes: `pin_bar`, `engulfing`, `is_inside_bar`, `swing_break` (Task 1); `ExitPolicy`/`NoExit` (`strategy/exits.py`); `Strategy` base.
- Produces:
  - `PriceActionTrend(symbol, pattern, trend, size, wick_ratio=2.0, body_frac=0.34, swing_lookback=20, inside_break_bars=3, max_hold_bars=0, exit_policy=None)`
  - Attribute `_last_exit_reason: str | None` ∈ {`"stop"`, `"tp"`, `"time"`, `None`} (parity with `orb`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategy/test_price_action.py
from datetime import datetime, timedelta, timezone

import pytest

from fxtrader.core.clock import SimClock
from fxtrader.core.portfolio import Portfolio
from fxtrader.core.types import Bar, Fill, Order, Side
from fxtrader.strategy.examples.price_action import PriceActionTrend
from fxtrader.strategy.exits import FixedPips

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _drive(strategy, ohlc, *, symbol="EURUSD"):
    """Feed (open, high, low, close) bars; fills execute at bar.close into the portfolio."""
    pf = Portfolio(100_000.0)
    clock = SimClock()
    orders: list[Order] = []

    def sink(order: Order) -> None:
        orders.append(order)
        pf.apply_fill(Fill("x", order.symbol, order.side, order.quantity,
                           _drive.price, clock.now(), 0.0))

    strategy.bind(clock, pf, sink)
    strategy.on_start()
    for i, (o, h, l, c) in enumerate(ohlc):
        ts = T0 + timedelta(minutes=i)
        clock.set_time(ts)
        _drive.price = c
        strategy.on_bar(Bar(symbol, ts, o, h, l, c, 1.0))
    return pf, orders


# Uptrend warmup (4 bars, rising closes) then a bullish PIN on the 5th bar.
_UPTREND_PIN = [
    (1.0000, 1.0010, 0.9990, 1.0000),
    (1.0000, 1.0020, 1.0000, 1.0010),
    (1.0010, 1.0030, 1.0010, 1.0020),
    (1.0020, 1.0040, 1.0020, 1.0030),
    (1.0035, 1.0040, 0.9990, 1.0038),  # bullish pin; close 1.0038 > trend MA → long
]


def test_rejects_bad_pattern():
    with pytest.raises(ValueError):
        PriceActionTrend("EURUSD", pattern="nope", trend=4, size=10_000)


def test_no_trades_before_warmup():
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000)
    _pf, orders = _drive(s, _UPTREND_PIN[:3])  # < trend bars
    assert orders == []


def test_pin_in_uptrend_goes_long():
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000)
    pf, orders = _drive(s, _UPTREND_PIN)
    assert pf.position("EURUSD").quantity > 0
    assert len(orders) == 1 and orders[0].side == Side.BUY


def test_bullish_pin_against_downtrend_blocked():
    # Falling closes → trend down; a bullish pin must be suppressed by the gate.
    series = [
        (1.0030, 1.0040, 1.0020, 1.0030),
        (1.0020, 1.0030, 1.0010, 1.0020),
        (1.0010, 1.0020, 1.0000, 1.0010),
        (1.0000, 1.0010, 0.9990, 1.0000),
        (0.9995, 1.0000, 0.9950, 0.9998),  # bullish pin, but close < trend MA → blocked
    ]
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000)
    pf, orders = _drive(s, series)
    assert pf.position("EURUSD").quantity == 0
    assert orders == []


def test_exit_first_stop_flattens():
    # Enter long on bar 5 at close 1.0038; FixedPips stop_dist=0.0020 → stop level 1.0018.
    s = PriceActionTrend(
        "EURUSD", pattern="pin", trend=4, size=10_000,
        exit_policy=FixedPips(stop_pips=20, tp_pips=200, pip_size=0.0001),
    )
    stop_bar = (1.0030, 1.0035, 1.0010, 1.0020)  # low 1.0010 <= 1.0018 → stop
    pf, orders = _drive(s, _UPTREND_PIN + [stop_bar])
    assert pf.position("EURUSD").quantity == 0
    assert orders[-1].side == Side.SELL
    assert s._last_exit_reason == "stop"


def test_max_hold_bars_flattens():
    # No exit policy; max_hold_bars=2 → time exit two bars after entry.
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000, max_hold_bars=2)
    hold1 = (1.0038, 1.0045, 1.0030, 1.0040)
    hold2 = (1.0040, 1.0048, 1.0035, 1.0044)  # 2nd bar held → time exit
    pf, orders = _drive(s, _UPTREND_PIN + [hold1, hold2])
    assert pf.position("EURUSD").quantity == 0
    assert s._last_exit_reason == "time"


def test_ignores_other_symbols():
    s = PriceActionTrend("EURUSD", pattern="pin", trend=4, size=10_000)
    _pf, orders = _drive(s, _UPTREND_PIN, symbol="GBPUSD")
    assert orders == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/strategy/test_price_action.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fxtrader.strategy.examples.price_action'`.

- [ ] **Step 3: Write the implementation**

```python
# src/fxtrader/strategy/examples/price_action.py
"""Trend-gated price-action strategy: fire a famous PA pattern only with the long-MA trend."""

from __future__ import annotations

from collections import deque

from fxtrader.core.types import Bar, Side
from fxtrader.strategy.base import Strategy
from fxtrader.strategy.exits import ExitPolicy, NoExit
from fxtrader.strategy.patterns import engulfing, is_inside_bar, pin_bar, swing_break

_PATTERNS = ("pin", "engulf", "inside", "swingbreak")


class PriceActionTrend(Strategy):
    def __init__(
        self,
        symbol: str,
        pattern: str,
        trend: int,
        size: float,
        wick_ratio: float = 2.0,
        body_frac: float = 0.34,
        swing_lookback: int = 20,
        inside_break_bars: int = 3,
        max_hold_bars: int = 0,
        exit_policy: ExitPolicy | None = None,
    ) -> None:
        if pattern not in _PATTERNS:
            raise ValueError(f"pattern must be one of {_PATTERNS}")
        if trend < 2:
            raise ValueError("trend window must be >= 2")
        if swing_lookback < 1 or inside_break_bars < 1 or max_hold_bars < 0:
            raise ValueError("swing_lookback/inside_break_bars >= 1, max_hold_bars >= 0")
        self.symbol = symbol
        self.pattern = pattern
        self.trend = trend
        self.size = size
        self.wick_ratio = wick_ratio
        self.body_frac = body_frac
        self.swing_lookback = swing_lookback
        self.inside_break_bars = inside_break_bars
        self.max_hold_bars = max_hold_bars
        self.exit_policy: ExitPolicy = exit_policy or NoExit()
        self._closes: deque[float] = deque(maxlen=trend)
        self._bars: deque[Bar] = deque(maxlen=max(swing_lookback + 1, 2))
        self._entry_price: float | None = None
        self._stop_dist: float | None = None
        self._tp_dist: float | None = None
        self._pos_dir = 0
        self._hold = 0
        self._mother: tuple[float, float, int] | None = None  # (hi, lo, bars_left)
        self._last_exit_reason: str | None = None

    def on_bar(self, bar: Bar) -> None:
        if bar.symbol != self.symbol:
            return
        self.exit_policy.update(bar)
        self._closes.append(bar.close)
        self._bars.append(bar)

        # 1. EXIT FIRST — stop, then TP (stop-first same-bar), then time.
        position = self.portfolio.position(self.symbol).quantity
        if position != 0 and self._entry_price is not None:
            entry = self._entry_price
            reason: str | None = None
            if position > 0:
                if self._stop_dist is not None and bar.low <= entry - self._stop_dist:
                    reason = "stop"
                elif self._tp_dist is not None and bar.high >= entry + self._tp_dist:
                    reason = "tp"
            else:
                if self._stop_dist is not None and bar.high >= entry + self._stop_dist:
                    reason = "stop"
                elif self._tp_dist is not None and bar.low <= entry - self._tp_dist:
                    reason = "tp"
            if reason is None and self.max_hold_bars > 0:
                self._hold += 1
                if self._hold >= self.max_hold_bars:
                    reason = "time"
            if reason:
                self._flatten(position, reason)
                return

        if len(self._closes) < self.trend:
            return

        trend_ma = sum(self._closes) / len(self._closes)
        trend_dir = 1 if bar.close > trend_ma else (-1 if bar.close < trend_ma else 0)
        if trend_dir == 0:
            return

        # 2. ENTRY — only when flat and the pattern agrees with the trend.
        if self.portfolio.position(self.symbol).quantity != 0:
            return
        signal = self._detect(bar)
        if signal != 0 and signal == trend_dir:
            self._enter(bar, signal * self.size)

    def _detect(self, bar: Bar) -> int:
        if self.pattern == "pin":
            return pin_bar(bar, wick_ratio=self.wick_ratio, body_frac=self.body_frac)
        if self.pattern == "engulf":
            if len(self._bars) < 2:
                return 0
            return engulfing(self._bars[-2], bar)
        if self.pattern == "swingbreak":
            return swing_break(list(self._bars), self.swing_lookback)
        if self.pattern == "inside":
            return self._inside_break(bar)
        return 0

    def _inside_break(self, bar: Bar) -> int:
        sig = 0
        if self._mother is not None:
            hi, lo, left = self._mother
            if bar.close > hi:
                sig, self._mother = 1, None
            elif bar.close < lo:
                sig, self._mother = -1, None
            else:
                left -= 1
                self._mother = (hi, lo, left) if left > 0 else None
        if self._mother is None and len(self._bars) >= 2:
            prev = self._bars[-2]
            if is_inside_bar(prev, bar):
                self._mother = (prev.high, prev.low, self.inside_break_bars)
        return sig

    def _enter(self, bar: Bar, desired: float) -> None:
        side = Side.BUY if desired > 0 else Side.SELL
        self.order(self.symbol, side, abs(desired))
        self._pos_dir = 1 if desired > 0 else -1
        self._entry_price = bar.close
        self._stop_dist, self._tp_dist = self.exit_policy.distances()
        self._hold = 0

    def _flatten(self, position: float, reason: str) -> None:
        side = Side.SELL if position > 0 else Side.BUY
        self.order(self.symbol, side, abs(position))
        self._last_exit_reason = reason
        self._pos_dir = 0
        self._entry_price = None
        self._stop_dist = self._tp_dist = None
        self._hold = 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/strategy/test_price_action.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/examples/price_action.py tests/strategy/test_price_action.py
git commit -m "feat(strategy): PriceActionTrend (pa) — trend-gated PA patterns + exit-first stop/TP/time

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Registry wiring (`strategy/registry.py`)

**Files:**
- Modify: `src/fxtrader/strategy/registry.py` (add `STRATEGY_PARAMS["pa"]`; add `"pa"` branch to `make_strategy`)
- Test: `tests/strategy/test_registry.py` (append)

**Interfaces:**
- Consumes: `PriceActionTrend` (Task 2).
- Produces: `STRATEGY_PARAMS["pa"] = ("pattern", "trend", "wick_ratio", "body_frac", "swing_lookback", "inside_break_bars", "max_hold_bars")`; `make_strategy("pa", ...)` returns a `PriceActionTrend`.

- [ ] **Step 1: Write the failing tests** (append to `tests/strategy/test_registry.py`)

```python
from fxtrader.strategy.examples.price_action import PriceActionTrend  # add near other imports


def test_make_strategy_pa():
    s = make_strategy(
        "pa", symbol="EURUSD",
        params={"pattern": "pin", "trend": 50, "size": 10_000, "wick_ratio": 2.0,
                "body_frac": 0.34, "swing_lookback": 20, "inside_break_bars": 3,
                "max_hold_bars": 0},
        exit_policy=AtrMultiple(14, 1.5, 2.0),
    )
    assert isinstance(s, PriceActionTrend)


def test_strategy_params_has_pa():
    assert STRATEGY_PARAMS["pa"] == (
        "pattern", "trend", "wick_ratio", "body_frac",
        "swing_lookback", "inside_break_bars", "max_hold_bars",
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/strategy/test_registry.py::test_make_strategy_pa tests/strategy/test_registry.py::test_strategy_params_has_pa -q`
Expected: FAIL — `KeyError: 'pa'` / `ValueError: unknown strategy 'pa'`.

- [ ] **Step 3: Write the implementation**

In `src/fxtrader/strategy/registry.py`, add the import near the other example imports:

```python
from fxtrader.strategy.examples.price_action import PriceActionTrend
```

Add the params entry to `STRATEGY_PARAMS` (after the `"xpair"` line):

```python
    "pa": ("pattern", "trend", "wick_ratio", "body_frac",
           "swing_lookback", "inside_break_bars", "max_hold_bars"),
```

Add the branch in `make_strategy` (before the final `raise ValueError`):

```python
    if name == "pa":
        return PriceActionTrend(
            symbol=symbol, pattern=params["pattern"], trend=params["trend"], size=params["size"],
            wick_ratio=params.get("wick_ratio", 2.0), body_frac=params.get("body_frac", 0.34),
            swing_lookback=params.get("swing_lookback", 20),
            inside_break_bars=params.get("inside_break_bars", 3),
            max_hold_bars=params.get("max_hold_bars", 0),
            exit_policy=exit_policy,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/strategy/test_registry.py -q`
Expected: PASS (all registry tests, including the two new ones).

- [ ] **Step 5: Commit**

```bash
git add src/fxtrader/strategy/registry.py tests/strategy/test_registry.py
git commit -m "feat(registry): wire pa strategy + STRATEGY_PARAMS

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `run_backtest` + `_strategy_run_fn` plumbing (`cli.py`)

Enables single-backtest research (Phase-B Pass 1/2/OOS) and the WFO `run_fn`. Follows the existing per-strategy keyword-plumbing pattern.

**Files:**
- Modify: `src/fxtrader/cli.py` (`run_backtest` signature + `make_strategy` params dict; `_strategy_run_fn` validation + call)
- Test: `tests/test_cli_pa.py`

**Interfaces:**
- Consumes: `run_backtest`, `make_strategy` (`"pa"` branch from Task 3).
- Produces: `run_backtest(..., strategy_name="pa", pattern=..., trend=..., wick_ratio=..., body_frac=..., swing_lookback=..., inside_break_bars=..., max_hold_bars=...)` returns `(BacktestResult, metrics)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_pa.py
from datetime import datetime, timezone, timedelta

from fxtrader.cli import run_backtest
from fxtrader.core.types import Tick
from fxtrader.data.store import TickStore


def _seed_uptrend(tmp_path, days=8):
    """Gently rising 30-min ticks (a clean uptrend) so the trend gate is satisfiable."""
    store = TickStore(tmp_path)
    ticks = []
    base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    px = 1.1000
    for slot in range(days * 48):
        ts = base + timedelta(minutes=30 * slot)
        px += 0.00005
        ticks.append(Tick("EURUSD", ts, px, px + 0.00002))
    store.write_ticks("EURUSD", ticks)


def test_run_backtest_pa_end_to_end(tmp_path):
    _seed_uptrend(tmp_path)
    result, metrics = run_backtest(
        symbol="EURUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 9, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy_name="pa", pattern="pin", trend=5,
        bar_minutes=30, exit_kind="atr", atr_period=5, stop_atr=1.5, tp_atr=2.0,
        max_position=50_000,
    )
    # Smoke: completes, returns the standard metrics dict, account not ruined.
    assert "num_trades" in metrics and metrics["num_trades"] >= 0
    assert metrics["ruined"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli_pa.py -q`
Expected: FAIL — `TypeError: run_backtest() got an unexpected keyword argument 'pattern'` (the strategy is wired by Task 3, but `run_backtest` does not yet declare the pa kwargs).

- [ ] **Step 3: Add the kwargs to `run_backtest`**

In `src/fxtrader/cli.py`, extend the `run_backtest` signature (after `stop_z: float = 0.0,`):

```python
    pattern: str = "pin",
    wick_ratio: float = 2.0,
    body_frac: float = 0.34,
    swing_lookback: int = 20,
    inside_break_bars: int = 3,
    max_hold_bars: int = 0,
```

In the `make_strategy(...)` call inside `run_backtest`, add these keys to the `params={...}` dict:

```python
            "pattern": pattern, "wick_ratio": wick_ratio, "body_frac": body_frac,
            "swing_lookback": swing_lookback, "inside_break_bars": inside_break_bars,
            "max_hold_bars": max_hold_bars,
```

- [ ] **Step 4: Add the `"pa"` branch to `_strategy_run_fn`**

In `_strategy_run_fn`'s `run_fn`, add a validation branch alongside the others:

```python
        elif strategy_name == "pa":
            if p.get("trend", 0) < 2 or p.get("pattern", "pin") not in (
                "pin", "engulf", "inside", "swingbreak"
            ):
                return None
```

And add the pa kwargs to the `run_backtest(...)` call inside `run_fn` (after `stop_z=p.get("stop_z", 0.0),`):

```python
            pattern=p.get("pattern", "pin"), wick_ratio=p.get("wick_ratio", 2.0),
            body_frac=p.get("body_frac", 0.34), swing_lookback=p.get("swing_lookback", 20),
            inside_break_bars=p.get("inside_break_bars", 3),
            max_hold_bars=p.get("max_hold_bars", 0),
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_cli_pa.py -q`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add src/fxtrader/cli.py tests/test_cli_pa.py
git commit -m "feat(cli): plumb pa params through run_backtest + _strategy_run_fn

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `run_optimize` grid + `backtest` CLI subcommand (`cli.py`)

Enables walk-forward optimization (the rigor guardrail against single-fold flukes) and CLI use consistent with the other five strategies.

**Files:**
- Modify: `src/fxtrader/cli.py` (`run_optimize` signature + grid/base_params for `"pa"`; `backtest` command options + `"pa"` branch + guard string)
- Test: `tests/test_cli_pa.py` (append)

**Interfaces:**
- Consumes: `_strategy_run_fn` `"pa"` branch (Task 4).
- Produces: `run_optimize(..., strategy="pa", pattern="pin", trend_values=[...], wick_ratio_values=[...], body_frac_values=[...])` → strict-JSON dict with `param_grid` / `folds` / `mean_oos_objective`; `fxtrader backtest --strategy pa --pattern pin ...`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cli_pa.py`)

```python
from typer.testing import CliRunner

from fxtrader.cli import app, run_optimize

runner = CliRunner()


def test_backtest_pa_cli_runs_end_to_end(tmp_path):
    _seed_uptrend(tmp_path)
    res = runner.invoke(app, [
        "backtest", "--strategy", "pa", "--symbol", "EURUSD",
        "--start", "2024-01-01", "--end", "2024-01-09",
        "--pattern", "pin", "--trend", "5", "--bar-minutes", "30",
        "--exit", "atr", "--atr-period", "5", "--stop-atr", "1.5", "--tp-atr", "2.0",
        "--max-position", "50000", "--data-dir", str(tmp_path),
    ])
    assert res.exit_code == 0, res.output
    assert "pa" in res.output.lower()


def test_optimize_pa_grid(tmp_path):
    _seed_uptrend(tmp_path, days=20)
    out = run_optimize(
        symbol="EURUSD",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 20, tzinfo=timezone.utc),
        data_dir=tmp_path, strategy="pa", pattern="pin",
        trend_values=[5], wick_ratio_values=[2.0], body_frac_values=[0.34],
        n_splits=2, bar_minutes=30, objective="sharpe",
    )
    assert out["param_grid"] == {"trend": [5], "wick_ratio": [2.0], "body_frac": [0.34]}
    assert "folds" in out and "mean_oos_objective" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_cli_pa.py::test_backtest_pa_cli_runs_end_to_end tests/test_cli_pa.py::test_optimize_pa_grid -q`
Expected: FAIL — CLI raises `BadParameter` (guard lists only `ma|trend|meanrev|orb|xpair`); `run_optimize` has no `pattern`/`*_values` kwargs → `TypeError`.

- [ ] **Step 3: Extend `run_optimize`**

In `src/fxtrader/cli.py`, add to the `run_optimize` signature (after `stop_z_values: list[float] | None = None,`):

```python
    pattern: str = "pin",
    wick_ratio_values: list[float] | None = None,
    body_frac_values: list[float] | None = None,
    swing_lookback_values: list[int] | None = None,
    inside_break_bars_values: list[int] | None = None,
```

In the grid-building if/elif chain, add a `"pa"` branch (before the final `else`):

```python
    elif strategy == "pa":
        grid = {"trend": trend_values or []}
        if pattern == "pin":
            grid["wick_ratio"] = wick_ratio_values or []
            grid["body_frac"] = body_frac_values or []
        elif pattern == "swingbreak":
            grid["swing_lookback"] = swing_lookback_values or []
        elif pattern == "inside":
            grid["inside_break_bars"] = inside_break_bars_values or []
        # "engulf": trend-only grid
```

After `base_params` is constructed, pin the fixed `pattern` so `run_fn` passes it through:

```python
    if strategy == "pa":
        base_params["pattern"] = pattern
```

- [ ] **Step 4: Add CLI options + the `"pa"` branch to the `backtest` command**

In the `backtest` command signature, add options (after `stop_z` option):

```python
    pattern: str = typer.Option("pin", "--pattern", help="pin|engulf|inside|swingbreak (pa)"),
    wick_ratio: float = typer.Option(2.0, "--wick-ratio", help="Pin wick:body ratio (pa pin)"),
    body_frac: float = typer.Option(0.34, "--body-frac", help="Pin max body/range (pa pin)"),
    swing_lookback: int = typer.Option(20, "--swing-lookback", help="Swing window bars (pa swingbreak)"),
    inside_break_bars: int = typer.Option(3, "--inside-break-bars", help="Bars to arm inside-break (pa inside)"),
    max_hold_bars: int = typer.Option(0, "--max-hold-bars", help="Hard time-exit in bars (0 = off)"),
```

Update the guard line:

```python
    if strategy not in ("ma", "trend", "meanrev", "orb", "xpair", "pa"):
        raise typer.BadParameter("strategy must be ma | trend | meanrev | orb | xpair | pa")
```

Add the branch (place it alongside the other strategy branches, e.g. after the `orb` branch):

```python
    if strategy == "pa":
        if not (symbol and start and end):
            raise typer.BadParameter("pa strategy needs --symbol/--start/--end")
        result, metrics = run_backtest(
            symbol=symbol, start=_parse_day(start), end=_parse_day(end), data_dir=data_dir,
            size=size, bar_minutes=bar_minutes, strategy_name="pa",
            pattern=pattern, trend=trend, wick_ratio=wick_ratio, body_frac=body_frac,
            swing_lookback=swing_lookback, inside_break_bars=inside_break_bars,
            max_hold_bars=max_hold_bars,
            exit_kind=exit_, stop_pips=stop_pips, tp_pips=tp_pips,
            atr_period=atr_period, stop_atr=stop_atr, tp_atr=tp_atr,
            daily_loss_limit=daily_loss_limit or float("inf"),
            max_position=max_position or float("inf"),
        )
        typer.echo(f"symbol={symbol}  bars={bar_minutes}m  pa({pattern} trend={trend})  exit={exit_}")
        typer.echo(f"  trades        : {metrics['num_trades']}")
        typer.echo(f"  total return  : {metrics['total_return']:.2%}")
        typer.echo(f"  Sharpe        : {metrics['sharpe']:.2f}")
        typer.echo(f"  profit factor : {metrics['profit_factor']:.2f}")
        return
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_cli_pa.py -q`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add src/fxtrader/cli.py tests/test_cli_pa.py
git commit -m "feat(cli): pa walk-forward grid + backtest subcommand

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Integration — full suite green + docs

**Files:**
- Modify: `CLAUDE.md` (add the `pa` row to the strategy table; mark it under research)

**Interfaces:**
- Consumes: all prior tasks.
- Produces: green suite (≥ 221 + new tests passed, 2 skipped); honest doc state.

- [ ] **Step 1: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS — previous baseline (221 passed, 2 skipped) **plus** the new tests (13 patterns + 7 strategy + 2 registry + 4 cli = 26). No failures, no new skips.

- [ ] **Step 2: Lint**

Run: `.venv/bin/ruff check src/fxtrader/strategy/patterns.py src/fxtrader/strategy/examples/price_action.py src/fxtrader/strategy/registry.py src/fxtrader/cli.py`
Expected: no errors (matches repo style). Fix any reported issues and re-run.

- [ ] **Step 3: Add the `pa` row to the CLAUDE.md strategy table**

In `CLAUDE.md`, under the "five strategies" table, append a row (and adjust the surrounding "five" wording to "six" where it refers to the built strategies — do NOT alter the Research Conclusion, which still describes the five evaluated strategies):

```markdown
| `pa` | `price_action.py` | trend-gated price-action patterns (pin/engulf/inside/swing-break) + MAE-calibrated ATR exit | **under research (Phase B, 2026-06-24)** |
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: register pa strategy (under research) in CLAUDE.md

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phase B — Research (NOT part of this code plan)

After Phase A merges, dispatch the `quant-researcher` agent (never edits `src/`) to run the 2-pass
MAE/MFE calibration protocol from the spec: for each pattern × {EURUSD, GBPUSD, USDJPY-bounded} ×
{4h=240m, 8h=480m}, run Pass-1 (wide stop + `max_hold_bars`) → read winner/loser MAE/MFE percentiles
via `metrics/excursions.py` → set `stop_atr`/`tp_atr` → Pass-2 IS confirm → OOS once. Drive via
`run_backtest` / `optimize --json` from `research/` scripts (mirror `run_orb_exit.py`). Deliverable:
`research/findings-2026-06-24-*.md` with per-cell trade counts, gross vs net profit factor, Sharpe,
OOS behavior, calibrated exits, and an honest verdict. See the design spec for survival criteria.

## Self-Review

- **Spec coverage:** strategy (Task 2), four detectors (Task 1), long-MA trend gate (Task 2), ATR
  exit reuse (existing `AtrMultiple`, exposed via Tasks 4–5), `max_hold_bars` safety (Task 2),
  registry + CLI + WFO wiring (Tasks 3–5), USDJPY bounding via existing `max_position` (Task 4/5
  tests pass it), tests + green suite + docs (Task 6). Phase-B protocol documented (not coded). ✓
- **Placeholder scan:** no TBD/TODO; every code step shows full code. ✓
- **Type consistency:** detector signatures, `PriceActionTrend(...)` params, `STRATEGY_PARAMS["pa"]`
  tuple, and `run_backtest`/`run_optimize` kwargs are identical everywhere they appear across tasks. ✓
