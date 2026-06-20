"""Pluggable exit policies. Pure logic; distances are in price terms."""

from __future__ import annotations

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
