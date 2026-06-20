"""Pluggable exit policies. Pure logic; distances are in price terms."""

from __future__ import annotations

from collections import deque
from typing import Protocol, runtime_checkable

from fxtrader.core.types import Bar


@runtime_checkable
class ExitPolicy(Protocol):
    def update(self, bar: Bar) -> None: ...
    def distances(self) -> tuple[float | None, float | None]: ...


class NoExit:
    """No stop/TP — the pure trend-gate strategy."""

    def update(self, bar: Bar) -> None:  # noqa: D401
        return None

    def distances(self) -> tuple[float | None, float | None]:
        return (None, None)


class FixedPips:
    """Constant stop/TP distances expressed in pips."""

    def __init__(self, stop_pips: float, tp_pips: float, pip_size: float) -> None:
        if stop_pips < 0 or tp_pips < 0 or pip_size <= 0:
            raise ValueError("pips must be >= 0 and pip_size > 0")
        self._stop = stop_pips * pip_size if stop_pips > 0 else None
        self._tp = tp_pips * pip_size if tp_pips > 0 else None

    def update(self, bar: Bar) -> None:
        return None

    def distances(self) -> tuple[float | None, float | None]:
        return (self._stop, self._tp)


class AtrMultiple:
    """Stop/TP as multiples of ATR (volatility-scaled), frozen at entry by the caller."""

    def __init__(self, period: int, stop_mult: float, tp_mult: float) -> None:
        if period <= 0 or stop_mult < 0 or tp_mult < 0:
            raise ValueError("period must be > 0 and mults >= 0")
        self._period = period
        self._stop_mult = stop_mult
        self._tp_mult = tp_mult
        self._trs: deque[float] = deque(maxlen=period)
        self._prev_close: float | None = None

    def update(self, bar: Bar) -> None:
        if self._prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - self._prev_close),
                abs(bar.low - self._prev_close),
            )
        self._trs.append(tr)
        self._prev_close = bar.close

    def distances(self) -> tuple[float | None, float | None]:
        if len(self._trs) < self._period:
            return (None, None)
        atr = sum(self._trs) / len(self._trs)
        stop = self._stop_mult * atr if self._stop_mult > 0 else None
        tp = self._tp_mult * atr if self._tp_mult > 0 else None
        return (stop, tp)
