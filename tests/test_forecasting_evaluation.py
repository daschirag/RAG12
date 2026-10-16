"""A5: rolling-origin folds (chronology, no overlap) and the paired day bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backend.forecasting.evaluation import (
    FoldSpec,
    daily_clearness,
    paired_day_bootstrap,
    rolling_origin_folds,
    target_days,
)
from src.backend.forecasting.features import build_features
from src.backend.forecasting.preprocess import load_plant_raw, preprocess


@pytest.fixture(scope="module")
def data():
    gen, weather = load_plant_raw(1)
    return build_features(preprocess(gen, weather, plant=1).frame)


def test_folds_cover_the_last_days_in_order(data):
    folds = list(rolling_origin_folds(data))
    days = target_days(data)
    assert len(days) == 34
    assert [d for d, _ in folds] == list(days[22:])          # 14 + 4 + 4 = 22 days before the first test day
    assert len(folds) == 12


def test_each_fold_is_chronological_disjoint_and_expanding(data):
    prev_train = 0
    for test_day, parts in rolling_origin_folds(data):
        order = ["train", "val", "calib", "test"]
        for a, b in zip(order, order[1:]):
            assert parts[a].meta["target_time"].max() < parts[b].meta["target_time"].min()
        assert (parts["test"].meta["target_time"].dt.normalize() == test_day).all()
        assert parts["val"].meta["target_time"].dt.normalize().nunique() == 4
        assert parts["calib"].meta["target_time"].dt.normalize().nunique() == 4
        n_train = len(parts["train"].y)
        assert n_train > prev_train                            # window expands
        prev_train = n_train
        idx = [set(p.X.index) for p in parts.values()]
        assert sum(len(s) for s in idx) == len(set().union(*idx))


def test_last_fold_matches_fixed_split_segments(data):
    """The final fold's val/calib days equal the A2 fixed split's last val/calib days shifted to test day 34."""
    test_day, parts = list(rolling_origin_folds(data))[-1]
    assert test_day == pd.Timestamp("2020-06-17")
    assert parts["calib"].meta["target_time"].min().normalize() == pd.Timestamp("2020-06-13")
    assert parts["val"].meta["target_time"].min().normalize() == pd.Timestamp("2020-06-09")


def test_fold_spec_too_large_rejected(data):
    with pytest.raises(ValueError):
        list(rolling_origin_folds(data, FoldSpec(min_train_days=30)))


def _days(n):
    return pd.date_range("2020-06-01", periods=n, freq="D")


def test_bootstrap_detects_a_consistent_difference():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.uniform(1000, 2000, 12), index=_days(12))
    better = base - 100 + rng.normal(0, 10, 12)
    res = paired_day_bootstrap(better, base, n_resamples=5000, seed=0)
    assert res.mean_diff == pytest.approx(-100, abs=10)
    assert res.ci_high < 0 and res.share_a_better == 1.0 and res.days_a_better == 12


def test_bootstrap_no_difference_ci_contains_zero():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 50, 12), index=_days(12))
    b = pd.Series(rng.normal(0, 50, 12), index=_days(12))
    res = paired_day_bootstrap(a, b, n_resamples=5000, seed=0)
    assert res.ci_low < 0 < res.ci_high


def test_bootstrap_is_paired_and_deterministic():
    a = pd.Series([1.0, 2.0, 3.0], index=_days(3))
    b = pd.Series([0.0, 5.0, 3.0, 99.0], index=_days(4))       # extra day ignored (paired on shared days)
    r1 = paired_day_bootstrap(a, b, n_resamples=2000, seed=7)
    r2 = paired_day_bootstrap(a, b, n_resamples=2000, seed=7)
    assert r1 == r2 and r1.n_days == 3
    assert r1.mean_diff == pytest.approx((1 - 3 + 0) / 3)
    with pytest.raises(ValueError):
        paired_day_bootstrap(a.iloc[:1], b.iloc[:1])


def test_daily_clearness_known_value():
    idx = pd.date_range("2020-06-01", periods=96, freq="15min")
    env = pd.Series(0.0, index=idx)
    env.iloc[40:56] = 1.0
    frame = pd.DataFrame({"irradiation": env * 0.5}, index=idx)
    assert daily_clearness(frame, env).iloc[0] == pytest.approx(0.5)


def test_day_bootstrap_ratio_pooled_and_ci():
    from src.backend.forecasting.evaluation import day_bootstrap_ratio

    covered = pd.Series([45, 40, 47, 44], index=_days(4))
    rows = pd.Series([50, 50, 50, 50], index=_days(4))
    point, lo, hi = day_bootstrap_ratio(covered, rows, n_resamples=4000, seed=0)
    assert point == pytest.approx(176 / 200)
    assert 40 / 50 <= lo <= point <= hi <= 47 / 50
    assert day_bootstrap_ratio(covered, rows, n_resamples=4000, seed=0) == (point, lo, hi)
