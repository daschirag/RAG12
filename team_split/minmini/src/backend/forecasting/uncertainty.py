"""A4 - Prediction intervals for 1-hour-ahead plant AC power.

Meaning of "confidence" everywhere in this project: the **nominal coverage
level** of the interval, e.g. 0.90 means the method targets containing the
observed value in 90% of cases. It is NOT the probability that the point
prediction is correct, and it is not a per-forecast probability.

Methods (all share the same XGBoost base models; see docs/forecasting.md):

- ``qr_raw``          quantile regression bounds, uncalibrated              (baseline)
- ``gaussian``        point +/- z * sigma, sigma from calibration residuals (baseline)
- ``split_conformal`` point +/- conformal quantile of |residual|            (baseline)
- ``cqr_pooled``      conformalized quantile regression, one calibration group
- ``cqr_mondrian``    CQR calibrated separately for day and night slots     (proposed)

Calibration uses the ``calib`` split only. Hyperparameters of the base models
are chosen on ``val`` only. Test data never enters any fit or calibration.

Split-conformal guarantee: if calibration and test scores are exchangeable,
P(y in interval) >= level (marginally). Time series are not exchangeable; see
the docs for what this means here.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import SupervisedSet
from .models import (
    DEFAULT_NIGHT_THRESHOLD,
    PLANT_CAPACITY_KW,
    TrainOnlyImputer,
    eval_mask,
    night_mask,
)

LEVELS: tuple[float, ...] = (0.80, 0.90, 0.95)
METHODS: tuple[str, ...] = ("qr_raw", "gaussian", "split_conformal", "cqr_pooled", "cqr_mondrian")
PROPOSED = "cqr_mondrian"


def quantiles_for(levels: tuple[float, ...] = LEVELS) -> list[float]:
    """Lower/upper quantiles needed for central intervals at ``levels`` plus the median."""
    qs = {0.5}
    for level in levels:
        a = 1 - level
        qs |= {round(a / 2, 6), round(1 - a / 2, 6)}
    return sorted(qs)


# ------------------------------------------------------------------- conformal

def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample split-conformal quantile: the ceil((n+1)*level)-th smallest score.

    Returns +inf when n is too small for the requested level.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * level)
    return math.inf if k > n else float(s[k - 1])


def _per_group(fn, groups_cal: np.ndarray | None, groups_te: np.ndarray | None, n_te: int, *cal_arrays):
    """Apply ``fn(*cal_arrays_subset) -> value`` per group; return per-test-row values."""
    out = np.empty(n_te, dtype=float)
    if groups_cal is None:
        out[:] = fn(*cal_arrays)
        return out
    for g in np.unique(groups_te):
        sel = groups_cal == g
        out[groups_te == g] = fn(*(a[sel] for a in cal_arrays))
    return out


def gaussian_interval(point_cal, y_cal, point_te, level, groups_cal=None, groups_te=None):
    z = norm.ppf(0.5 + level / 2)
    sigma = _per_group(lambda p, y: np.std(y - p, ddof=1) if len(y) > 1 else math.inf,
                       groups_cal, groups_te, len(point_te), point_cal, y_cal)
    return point_te - z * sigma, point_te + z * sigma


def split_conformal_interval(point_cal, y_cal, point_te, level, groups_cal=None, groups_te=None):
    q = _per_group(lambda p, y: conformal_quantile(np.abs(y - p), level),
                   groups_cal, groups_te, len(point_te), point_cal, y_cal)
    return point_te - q, point_te + q


def cqr_interval(lo_cal, hi_cal, y_cal, lo_te, hi_te, level, groups_cal=None, groups_te=None):
    q = _per_group(lambda lo, hi, y: conformal_quantile(np.maximum(lo - y, y - hi), level),
                   groups_cal, groups_te, len(lo_te), lo_cal, hi_cal, y_cal)
    return lo_te - q, hi_te + q


def cqr_offsets(lo_cal, hi_cal, y_cal, groups_cal, level: float, groups=(True, False)) -> dict:
    """Per-group CQR correction q (same score and rank as ``cqr_interval``), for storing in a model.

    A group with no calibration rows gets +inf, as in ``cqr_interval``.
    """
    scores = np.maximum(np.asarray(lo_cal) - y_cal, np.asarray(y_cal) - hi_cal)
    groups_cal = np.asarray(groups_cal)
    return {g: conformal_quantile(scores[groups_cal == g], level) for g in groups}


def finalize_interval(lo, hi, point, night: np.ndarray, capacity: float = PLANT_CAPACITY_KW):
    """Clip to [0, capacity], keep lo <= point <= hi, apply the night rule."""
    lo = np.clip(np.minimum(lo, point), 0.0, capacity)
    hi = np.clip(np.maximum(hi, point), 0.0, capacity)
    lo = np.where(night, 0.0, lo)
    hi = np.where(night, 0.0, hi)
    return lo, hi


# --------------------------------------------------------------------- metrics

def coverage(y, lo, hi) -> float:
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))


def winkler_score(y, lo, hi, level) -> float:
    """Interval (Winkler) score, lower is better: width + (2/alpha) * miss distance."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    a = 1 - level
    return float(np.mean((hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)))


def pinball_loss(y, q, tau) -> float:
    y, q = np.asarray(y), np.asarray(q)
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def interval_pinball(y, lo, hi, level) -> float:
    """Mean pinball loss of the two bounds, read as the alpha/2 and 1-alpha/2 quantiles."""
    a = 1 - level
    return 0.5 * (pinball_loss(y, lo, a / 2) + pinball_loss(y, hi, 1 - a / 2))


def interval_metrics(y, lo, hi, level) -> dict:
    return {
        "rows": int(len(y)),
        "coverage": coverage(y, lo, hi),
        "mean_width": mean_width(lo, hi),
        "winkler": winkler_score(y, lo, hi, level),
        "pinball": interval_pinball(y, lo, hi, level),
    }


# ---------------------------------------------------------- quantile regression

class QuantileXGBoost:
    """One multi-quantile XGBoost model; hyperparameters chosen on val mean pinball loss."""

    grid = {"max_depth": [3, 5], "learning_rate": [0.03, 0.1]}

    def __init__(self, quantiles: list[float], seed: int = 0) -> None:
        self.quantiles = list(quantiles)
        self.seed = seed

    def _make(self, params: dict):
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=np.array(self.quantiles),
            n_estimators=2000, subsample=0.8, colsample_bytree=0.8, random_state=self.seed,
            early_stopping_rounds=50, tree_method="hist", n_jobs=4, **params,
        )

    def fit(self, train: SupervisedSet, val: SupervisedSet, frame: pd.DataFrame | None = None):
        self.imputer_ = TrainOnlyImputer().fit(train.X)
        Xtr = self.imputer_.transform(train.X)
        Xva = self.imputer_.transform(val.X)
        m = eval_mask(val)
        best = None
        keys = sorted(self.grid)
        for values in itertools.product(*(self.grid[k] for k in keys)):
            params = dict(zip(keys, values))
            model = self._make(params)
            model.fit(Xtr, train.y, eval_set=[(Xva, val.y)], verbose=False)
            q = self._sorted(model.predict(Xva))
            score = np.mean([pinball_loss(val.y.to_numpy()[m], q[m, j], tau)
                             for j, tau in enumerate(self.quantiles)])
            if best is None or score < best[0]:
                best = (score, params, model)
        self.val_pinball_, self.params_, self.model_ = best
        self.params_ = {**self.params_, "best_iteration": int(self.model_.best_iteration)}
        return self

    @staticmethod
    def _sorted(pred: np.ndarray) -> np.ndarray:
        # Rearrangement removes quantile crossing (Chernozhukov et al. 2010).
        return np.sort(np.asarray(pred).reshape(len(pred), -1), axis=1)

    def predict(self, data: SupervisedSet) -> pd.DataFrame:
        pred = self._sorted(self.model_.predict(self.imputer_.transform(data.X)))
        return pd.DataFrame(pred, index=data.X.index, columns=self.quantiles)


# --------------------------------------------------------------- full pipeline

@dataclass
class IntervalSet:
    """Intervals for one split: {(method, level): (lo, hi)} plus the point forecast."""

    point: np.ndarray
    bounds: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]]


def build_intervals(
    point_cal: np.ndarray,
    q_cal: pd.DataFrame,
    calib: SupervisedSet,
    point_te: np.ndarray,
    q_te: pd.DataFrame,
    target: SupervisedSet,
    levels: tuple[float, ...] = LEVELS,
    night_threshold: float | None = DEFAULT_NIGHT_THRESHOLD,
    groups_cal: np.ndarray | None = None,
    groups_te: np.ndarray | None = None,
    capacity: float = PLANT_CAPACITY_KW,
) -> IntervalSet:
    """Calibrate every method on ``calib`` (scored rows only) and apply to ``target``.

    ``point_*`` are post-processed point forecasts; ``q_*`` are raw quantile
    predictions. Only ``calib.y`` is read; ``target.y`` is never used here.
    Mondrian groups default to the solar day/night flag (``target_daytime``);
    other domains pass their own per-row groups (full length of calib/target).
    """
    m = eval_mask(calib)
    y_cal = calib.y.to_numpy()[m]
    pc, pt = point_cal[m], point_te
    g_cal = (calib.meta["target_daytime"].to_numpy() if groups_cal is None else np.asarray(groups_cal))[m]
    g_te = target.meta["target_daytime"].to_numpy() if groups_te is None else np.asarray(groups_te)
    night = night_mask(target, night_threshold)
    bounds = {}
    for level in levels:
        a = 1 - level
        lo_tau, hi_tau = round(a / 2, 6), round(1 - a / 2, 6)
        lo_c, hi_c = q_cal[lo_tau].to_numpy()[m], q_cal[hi_tau].to_numpy()[m]
        lo_t, hi_t = q_te[lo_tau].to_numpy(), q_te[hi_tau].to_numpy()
        raw = {
            "qr_raw": (lo_t, hi_t),
            "gaussian": gaussian_interval(pc, y_cal, pt, level, g_cal, g_te),
            "split_conformal": split_conformal_interval(pc, y_cal, pt, level, g_cal, g_te),
            "cqr_pooled": cqr_interval(lo_c, hi_c, y_cal, lo_t, hi_t, level),
            "cqr_mondrian": cqr_interval(lo_c, hi_c, y_cal, lo_t, hi_t, level, g_cal, g_te),
        }
        for method, (lo, hi) in raw.items():
            bounds[(method, level)] = finalize_interval(lo, hi, pt, night, capacity)
    return IntervalSet(point=pt, bounds=bounds)


TIME_OF_DAY_BINS = (
    ("night_early", 0, 5 * 60),
    ("dawn", 5 * 60, 9 * 60),
    ("midday", 9 * 60, 16 * 60),
    ("dusk", 16 * 60, 20 * 60),
    ("night_late", 20 * 60, 24 * 60),
)


def time_of_day_bin(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """dawn 05:00-08:59, midday 09:00-15:59, dusk 16:00-19:59, else night (by target time)."""
    t = pd.DatetimeIndex(times)
    minutes = t.hour * 60 + t.minute
    out = np.empty(len(t), dtype=object)
    for name, lo, hi in TIME_OF_DAY_BINS:
        out[(minutes >= lo) & (minutes < hi)] = "night" if name.startswith("night") else name
    return out
