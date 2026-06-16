from datetime import datetime, timezone

import pytest

from fxtrader.research.walkforward import Fold, walk_forward_splits


def _dt(month, day):
    return datetime(2024, month, day, tzinfo=timezone.utc)


def test_single_split_70_30():
    start, end = _dt(1, 1), _dt(1, 11)  # 10 days
    folds = walk_forward_splits(start, end, n_splits=1, train_ratio=0.7)
    assert len(folds) == 1
    f = folds[0]
    assert f.train_start == start
    assert f.train_end == _dt(1, 8)   # 7 days in
    assert f.test_start == _dt(1, 8)
    assert f.test_end == end


def test_two_splits_are_contiguous_and_cover_the_range():
    start, end = _dt(1, 1), _dt(1, 11)  # 10 days -> two 5-day windows
    folds = walk_forward_splits(start, end, n_splits=2, train_ratio=0.6)
    assert len(folds) == 2
    # window 1: [01-01, 01-06); train 3 days, test 2 days
    assert folds[0].train_start == _dt(1, 1)
    assert folds[0].train_end == _dt(1, 4)
    assert folds[0].test_end == _dt(1, 6)
    # window 2 starts where window 1 ended
    assert folds[1].train_start == _dt(1, 6)
    assert folds[1].test_end == end


def test_returns_fold_dataclass():
    folds = walk_forward_splits(_dt(1, 1), _dt(1, 11), n_splits=1, train_ratio=0.5)
    assert isinstance(folds[0], Fold)


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        walk_forward_splits(_dt(1, 1), _dt(1, 11), n_splits=0, train_ratio=0.7)
    with pytest.raises(ValueError):
        walk_forward_splits(_dt(1, 1), _dt(1, 11), n_splits=1, train_ratio=1.0)
    with pytest.raises(ValueError):
        walk_forward_splits(_dt(1, 11), _dt(1, 1), n_splits=1, train_ratio=0.7)  # end <= start
