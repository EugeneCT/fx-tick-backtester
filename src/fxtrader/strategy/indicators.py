"""Pure technical indicators used by strategies. No I/O, no state."""

from __future__ import annotations

import statistics
from collections.abc import Sequence


def zscore(closes: Sequence[float]) -> float:
    """(last − mean) / population stdev over the window; 0.0 when the window is flat."""
    std = statistics.pstdev(closes)
    if std == 0:
        return 0.0
    return (closes[-1] - statistics.fmean(closes)) / std


def efficiency_ratio(closes: Sequence[float]) -> float:
    """Kaufman Efficiency Ratio: |net move| / sum(|bar-to-bar moves|), in [0, 1]; 0.0 if flat."""
    gross = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
    if gross == 0:
        return 0.0
    return abs(closes[-1] - closes[0]) / gross
