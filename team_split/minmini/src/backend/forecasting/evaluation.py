"""A5 - Rolling-origin folds and paired day-level bootstrap.

Rolling origin (expanding window): for every test day d after a minimum
training window, the days before d are split, in time order, into
train (all earlier days) | val (``val_days``) | calib (``calib_days``), exactly
as in the fixed A2 split. Rows are assigned by the day of their *target* time,
so no training target lies in a later segment.

Paired day-level bootstrap: errors within a day are strongly autocorrelated,
so days (not 15-min rows) are the resampling unit. Both models are scored on
the same days; each resample draws days with replacement and averages the
per-day loss difference.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import SupervisedSet


@dataclass(frozen=True)
class FoldSpec:
    min_train_days: int = 14
    val_days: int = 4
    calib_days: int = 4


def _subset(data: SupervisedSet, mask: np.ndarray) -> SupervisedSet:
    return SupervisedSet(X=data.X[mask], y=data.y[mask], meta=data.meta[mask], config=data.config)


def target_days(data: SupervisedSet) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(data.meta["target_time"].dt.normalize().unique()).sort_values()


def rolling_origin_folds(
    data: SupervisedSet, spec: FoldSpec = FoldSpec()
) -> Iterator[tuple[pd.Timestamp, dict[str, SupervisedSet]]]:
    """Yield (test_day, {"train", "val", "calib", "test"}) for each eligible test day."""
    days = target_days(data)
    day_of_row = data.meta["target_time"].dt.normalize()
    first = spec.min_train_days + spec.val_days + spec.calib_days
    if first >= len(days):
        raise ValueError(f"Need more than {first} days; have {len(days)}")
    for i in range(first, len(days)):
        cal_start = i - spec.calib_days
        val_start = cal_start - spec.val_days
        segments = {
            "train": days[:val_start],
            "val": days[val_start:cal_start],
            "calib": days[cal_start:i],
            "test": days[i:i + 1],
        }
        yield days[i], {name: _subset(data, day_of_row.isin(d).to_numpy()) for name, d in segments.items()}


@dataclass(frozen=True)
class BootstrapResult:
    n_days: int
    mean_diff: float          # mean over days of loss(A) - loss(B); negative -> A better
    ci_low: float
    ci_high: float
    share_a_better: float     # share of resamples with mean diff < 0
    days_a_better: int        # observed days where A had the lower loss

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def paired_day_bootstrap(
    loss_a: pd.Series, loss_b: pd.Series, n_resamples: int = 10_000, seed: int = 0, ci: float = 0.95
) -> BootstrapResult:
    """Bootstrap the mean per-day loss difference A - B over the days both models share."""
    both = pd.concat([loss_a.rename("a"), loss_b.rename("b")], axis=1, join="inner").dropna()
    if len(both) < 2:
        raise ValueError("need at least 2 shared days")
    diff = (both["a"] - both["b"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return BootstrapResult(
        n_days=len(diff), mean_diff=float(diff.mean()), ci_low=float(lo), ci_high=float(hi),
        share_a_better=float(np.mean(means < 0)), days_a_better=int((diff < 0).sum()),
    )


def daily_clearness(frame: pd.DataFrame, envelope: pd.Series) -> pd.Series:
    """Per day: sum of measured irradiation / sum of the prior-days envelope (daylight slots).

    Near 1 = as clear as the clearest previous days; low = cloudy. Descriptive only.
    """
    irr = frame["irradiation"]
    day = envelope > 0.05
    num = irr.where(day).groupby(irr.index.normalize()).sum(min_count=1)
    den = envelope.where(day).groupby(envelope.index.normalize()).sum(min_count=1)
    return (num / den).rename("clearness")


def day_bootstrap_ratio(
    num_by_day: pd.Series, den_by_day: pd.Series, n_resamples: int = 10_000, seed: int = 0, ci: float = 0.95
) -> tuple[float, float, float]:
    """Pooled ratio sum(num)/sum(den) with a day-resampled percentile CI.

    Used for coverage: num = covered rows per day, den = scored rows per day.
    """
    both = pd.concat([num_by_day.rename("n"), den_by_day.rename("d")], axis=1, join="inner").dropna()
    if len(both) < 2 or (both["d"] <= 0).any():
        raise ValueError("need at least 2 days with positive denominators")
    n, d = both["n"].to_numpy(float), both["d"].to_numpy(float)
    idx = np.random.default_rng(seed).integers(0, len(n), size=(n_resamples, len(n)))
    ratios = n[idx].sum(axis=1) / d[idx].sum(axis=1)
    lo, hi = np.quantile(ratios, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(n.sum() / d.sum()), float(lo), float(hi)
