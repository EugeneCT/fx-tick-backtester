"""Instrument conventions. Pure, no I/O."""

from __future__ import annotations


def pip_size(symbol: str) -> float:
    """One pip in price terms: 0.01 for JPY-quoted pairs, else 0.0001."""
    return 0.01 if symbol.upper().endswith("JPY") else 0.0001
