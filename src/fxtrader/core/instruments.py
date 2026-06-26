"""Instrument conventions. Pure, no I/O."""

from __future__ import annotations


def pip_size(symbol: str) -> float:
    """One pip in price terms: 0.01 for JPY-quoted pairs, 0.1 for gold (XAU), else 0.0001."""
    s = symbol.upper()
    if s.endswith("JPY"):
        return 0.01
    if s.startswith("XAU"):
        return 0.1
    return 0.0001
