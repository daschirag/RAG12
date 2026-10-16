"""A4: conformal quantile, known-coverage sanity checks, calibration/test separation, determinism."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backend.forecasting.features import build_features, split_by_target_day
from src.backend.forecasting.models import PLANT_CAPACITY_KW, XGBoost, night_mask, postprocess
from src.backend.forecasting.preprocess import load_plant_raw, preprocess
from src.backend.forecasting.uncertainty import (
    LEVELS,
    METHODS,
    QuantileXGBoost,
    build_intervals,
    conformal_quantile,
    coverage,
    cqr_interval,
    gaussian_interval,
    interval_pinball,
    mean_width,
    pinball_loss,
    quantiles_for,
    split_conformal_interval,
    time_of_day_bin,
    winkler_score,
)

# ------------------------------------------------------------ unit pieces


def test_conformal_quantile_is_finite_sample_rank():
    scores = np.arange(1, 10, dtype=float)                   # n = 9
    assert conformal_quantile(scores, 0.9) == 9.0            # ceil(10*0.9) = 9th smallest
    assert conformal_quantile(scores, 0.5) == 5.0            # ceil(10*0.5) = 5th
    assert conformal_quantile(scores, 0.95) == math.inf      # ceil(10*0.95) = 10 > n
    assert conformal_quantile(np.array([]), 0.9) == math.inf


def test_quantiles_for_levels():
    assert quantiles_for((0.8, 0.9, 0.95)) == [0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975]


def test_interval_scores_known_values():
    y = np.array([0.0, 5.0, 10.0])
    lo, hi = np.array([1.0, 4.0, 4.0]), np.array([3.0, 6.0, 6.0])
    assert coverage(y, lo, hi) == pytest.approx(1 / 3)
    assert mean_width(lo, hi) == 2.0
    # level 0.9 -> alpha 0.1 -> penalty factor 20; misses: 1 below (row 0), 4 above (row 2).
    assert winkler_score(y, lo, hi, 0.9) == pytest.approx((2 + 2 + 2 + 20 * 1 + 20 * 4) / 3)
    # Under-prediction by 2 at tau 0.9 costs 0.9 * 2; over-prediction by 2 costs 0.1 * 2.
    assert pinball_loss(np.array([10.0]), np.array([8.0]), 0.9) == pytest.approx(1.8)
    assert pinball_loss(np.array([10.0]), np.array([12.0]), 0.9) == pytest.approx(0.2)
    assert interval_pinball(y, lo, hi, 0.9) > 0


def test_time_of_day_bins():
    t = pd.to_datetime(["2020-06-13 04:45", "2020-06-13 05:00", "2020-06-13 12:00",
                        "2020-06-13 16:00", "2020-06-13 19:45", "2020-06-13 20:00"])
    assert list(time_of_day_bin(t)) == ["night", "dawn", "midday", "dusk", "dusk", "night"]


# ------------------------------------------- coverage on synthetic data (known truth)

@pytest.mark.parametrize("level", LEVELS)
def test_split_conformal_hits_nominal_coverage_on_exchangeable_data(level):
    """Averaged over many i.i.d. repetitions, coverage is within a few MC s.e. of nominal
    and never materially below it (finite-sample guarantee: >= level)."""
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(400):
        y_cal, y_te = rng.standard_t(4, 200), rng.standard_t(4, 200)
        lo, hi = split_conformal_interval(np.zeros(200), y_cal, np.zeros(200), level)
        covs.append(coverage(y_te, lo, hi))
    mean_cov = float(np.mean(covs))
    se = float(np.std(covs) / np.sqrt(len(covs)))
    upper = level + 1 / 201                          # exact expectation bound for n = 200
    assert level - 3 * se <= mean_cov <= upper + 3 * se


@pytest.mark.parametrize("level", LEVELS)
def test_gaussian_interval_nominal_when_residuals_are_gaussian(level):
    rng = np.random.default_rng(1)
    y_cal, y_te = rng.normal(0, 2, 5000), rng.normal(0, 2, 20000)
    lo, hi = gaussian_interval(np.zeros(5000), y_cal, np.zeros(20000), level)
    assert coverage(y_te, lo, hi) == pytest.approx(level, abs=0.01)


def test_gaussian_interval_undercovers_heavy_tails():
    """Sanity check in the other direction: the Gaussian baseline is not distribution-free."""
    rng = np.random.default_rng(2)
    y_cal, y_te = rng.standard_t(2, 5000), rng.standard_t(2, 20000)
    lo, hi = gaussian_interval(np.zeros(5000), y_cal, np.zeros(20000), 0.8)
    assert coverage(y_te, lo, hi) > 0.8 + 0.03       # sigma inflated by tails -> over-covers at 80%
    lo, hi = gaussian_interval(np.zeros(5000), y_cal, np.zeros(20000), 0.99)
    assert coverage(y_te, lo, hi) >= 0.95            # still not a guarantee; documented, not asserted exact


@pytest.mark.parametrize("level", LEVELS)
def test_cqr_repairs_miscalibrated_quantiles(level):
    """Heteroscedastic data, quantile model that is too narrow by 50%: CQR restores coverage."""
    rng = np.random.default_rng(3)

    def draw(n):
        x = rng.uniform(1, 5, n)
        return x, rng.normal(0, x)                   # sd grows with x

    z = 0.5 * float(np.quantile(rng.normal(size=200000), 0.5 + level / 2))  # 50% too narrow
    x_cal, y_cal = draw(3000)
    x_te, y_te = draw(20000)
    lo_raw, hi_raw = -z * x_te, z * x_te
    assert coverage(y_te, lo_raw, hi_raw) < level - 0.1
    lo, hi = cqr_interval(-z * x_cal, z * x_cal, y_cal, lo_raw, hi_raw, level)
    assert coverage(y_te, lo, hi) == pytest.approx(level, abs=0.015)


def test_mondrian_groups_calibrate_separately():
    """Two groups with very different noise: pooled calibration mis-covers each group,
    per-group calibration covers both."""
    rng = np.random.default_rng(4)
    n = 4000
    g_cal, g_te = rng.random(n) < 0.5, rng.random(n) < 0.5
    y_cal = np.where(g_cal, rng.normal(0, 10, n), rng.normal(0, 0.1, n))
    y_te = np.where(g_te, rng.normal(0, 10, n), rng.normal(0, 0.1, n))
    zeros = np.zeros(n)
    lo_p, hi_p = split_conformal_interval(zeros, y_cal, zeros, 0.9)
    lo_m, hi_m = split_conformal_interval(zeros, y_cal, zeros, 0.9, g_cal, g_te)
    assert coverage(y_te[g_te], lo_p[g_te], hi_p[g_te]) < 0.85
    for grp in (g_te, ~g_te):
        assert coverage(y_te[grp], lo_m[grp], hi_m[grp]) == pytest.approx(0.9, abs=0.03)


# ------------------------------------------------ real pipeline: separation + determinism

@pytest.fixture(scope="module")
def pipeline_inputs():
    gen, weather = load_plant_raw(1)
    frame = preprocess(gen, weather, plant=1).frame
    parts = split_by_target_day(build_features(frame))
    return frame, parts


def _run(frame, parts, seed, grid_small=True):
    if grid_small:
        XGBoost.grid = {"max_depth": [3], "learning_rate": [0.1]}
        QuantileXGBoost.grid = {"max_depth": [3], "learning_rate": [0.1]}
    train, val, calib, test = (parts[k] for k in ("train", "val", "calib", "test"))
    point = XGBoost(seed=seed).fit(train, val, frame)
    qr = QuantileXGBoost(quantiles_for(), seed=seed).fit(train, val)
    pc = postprocess(point.predict(calib, frame), calib)
    pt = postprocess(point.predict(test, frame), test)
    return build_intervals(pc, qr.predict(calib), calib, pt, qr.predict(test), test)


@pytest.fixture(scope="module")
def run_seed0(pipeline_inputs):
    frame, parts = pipeline_inputs
    grids = (dict(XGBoost.grid), dict(QuantileXGBoost.grid))
    try:
        yield _run(frame, parts, seed=0)
    finally:
        XGBoost.grid, QuantileXGBoost.grid = grids


def test_intervals_do_not_depend_on_test_targets(pipeline_inputs, run_seed0):
    """build_intervals must never read test y: scrambling it changes nothing."""
    frame, parts = pipeline_inputs
    test = parts["test"]
    scrambled = type(test)(X=test.X, y=test.y * 0 + 12345.0, meta=test.meta, config=test.config)
    calib = parts["calib"]
    # Re-use the seed-0 point/quantile predictions via a tiny wrapper around the stored result.
    base = run_seed0
    q_dummy = pd.DataFrame({q: base.point for q in quantiles_for()}, index=test.X.index)
    qc_dummy = pd.DataFrame({q: np.zeros(len(calib.y)) for q in quantiles_for()}, index=calib.X.index)
    a = build_intervals(np.zeros(len(calib.y)), qc_dummy, calib, base.point, q_dummy, test)
    b = build_intervals(np.zeros(len(calib.y)), qc_dummy, calib, base.point, q_dummy, scrambled)
    for key in a.bounds:
        np.testing.assert_array_equal(a.bounds[key][0], b.bounds[key][0])
        np.testing.assert_array_equal(a.bounds[key][1], b.bounds[key][1])


def test_intervals_do_depend_on_calibration_targets(pipeline_inputs, run_seed0):
    """Positive control for the separation test: calibration y is actually used."""
    frame, parts = pipeline_inputs
    calib, test = parts["calib"], parts["test"]
    shifted = type(calib)(X=calib.X, y=calib.y + 3000.0, meta=calib.meta, config=calib.config)
    q_t = pd.DataFrame({q: run_seed0.point for q in quantiles_for()}, index=test.X.index)
    q_c = pd.DataFrame({q: np.zeros(len(calib.y)) for q in quantiles_for()}, index=calib.X.index)
    a = build_intervals(np.zeros(len(calib.y)), q_c, calib, run_seed0.point, q_t, test)
    b = build_intervals(np.zeros(len(calib.y)), q_c, shifted, run_seed0.point, q_t, test)
    lo_a, hi_a = a.bounds[("split_conformal", 0.9)]
    lo_b, hi_b = b.bounds[("split_conformal", 0.9)]
    assert (hi_b - lo_b).mean() > (hi_a - lo_a).mean()


def test_calibration_split_is_disjoint_from_test(pipeline_inputs):
    _, parts = pipeline_inputs
    assert parts["calib"].meta["target_time"].max() < parts["test"].meta["target_time"].min()
    assert parts["val"].meta["target_time"].max() < parts["calib"].meta["target_time"].min()
    assert not set(parts["calib"].X.index) & set(parts["test"].X.index)


def test_interval_shapes_bounds_and_night(pipeline_inputs, run_seed0):
    _, parts = pipeline_inputs
    test = parts["test"]
    night = night_mask(test)
    assert set(run_seed0.bounds) == {(m, lv) for m in METHODS for lv in LEVELS}
    for (method, level), (lo, hi) in run_seed0.bounds.items():
        assert lo.shape == hi.shape == (len(test.y),)
        assert (lo <= run_seed0.point + 1e-9).all() and (run_seed0.point <= hi + 1e-9).all()
        assert lo.min() >= 0 and hi.max() <= PLANT_CAPACITY_KW
        assert (lo[night] == 0).all() and (hi[night] == 0).all()
    # Wider nominal level -> interval at least as wide (per method, same calibration).
    for method in METHODS:
        w = [np.mean(run_seed0.bounds[(method, lv)][1] - run_seed0.bounds[(method, lv)][0]) for lv in LEVELS]
        assert w[0] <= w[1] + 1e-6 <= w[2] + 2e-6


def test_same_seed_same_intervals_and_seeds_differ(pipeline_inputs, run_seed0):
    frame, parts = pipeline_inputs
    grids = (dict(XGBoost.grid), dict(QuantileXGBoost.grid))
    try:
        again = _run(frame, parts, seed=0)
        other = _run(frame, parts, seed=1)
    finally:
        XGBoost.grid, QuantileXGBoost.grid = grids
    for key, (lo, hi) in run_seed0.bounds.items():
        np.testing.assert_array_equal(lo, again.bounds[key][0])
        np.testing.assert_array_equal(hi, again.bounds[key][1])
    assert not np.array_equal(run_seed0.point, other.point)   # seed actually matters (subsampling)


def test_confidence_definition_in_schema():
    from src.backend.forecasting.schemas import ForecastResponse

    desc = ForecastResponse.model_fields["confidence"].description.lower()
    assert "nominal coverage" in desc
    assert "not the probability" in desc
