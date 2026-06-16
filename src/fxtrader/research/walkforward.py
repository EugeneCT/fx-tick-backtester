"""Walk-forward train/test splitting for out-of-sample strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Fold:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


def walk_forward_splits(
    start: datetime,
    end: datetime,
    n_splits: int,
    train_ratio: float = 0.7,
) -> list[Fold]:
    """Split [start, end) into n_splits contiguous windows, each train_ratio train / rest test."""
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if end <= start:
        raise ValueError("end must be after start")

    total = (end - start) / n_splits
    folds: list[Fold] = []
    for i in range(n_splits):
        win_start = start + total * i
        win_end = start + total * (i + 1)
        split = win_start + (win_end - win_start) * train_ratio
        folds.append(Fold(win_start, split, split, win_end))
    return folds
