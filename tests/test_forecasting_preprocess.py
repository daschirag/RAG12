"""A1 preprocessing: each step on small synthetic frames, causality, split, scaler, real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backend.forecasting.preprocess import (
    TARGET,
    PreprocessConfig,
    TrainOnlyScaler,
    causal_smooth,
    chronological_split,
    load_plant_raw,
    preprocess,
    remove_duplicates,
    robust_trailing_z,
    snap_to_grid,
)

CFG = PreprocessConfig(n_inverters=2)


def _gen(rows):
    """rows: (timestamp, inverter, ac_kw)."""
    df = pd.DataFrame(rows, columns=["timestamp", "SOURCE_KEY", "AC_POWER"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["DC_POWER"] = df["AC_POWER"] * 1.02
    return df


def _weather(rows):
    """rows: (timestamp, irradiation, ambient, module)."""
    df = pd.DataFrame(
        rows, columns=["timestamp", "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _one_day(irr_fn=None, ac_per_irr=500.0, day="2020-05-15"):
    """Two inverters over one day with a clean irradiation bell curve."""
    ts = pd.date_range(day, periods=96, freq="15min")
    hours = ts.hour + ts.minute / 60
    irr = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None) if irr_fn is None else irr_fn(hours)
    gen = _gen([(t, inv, float(i * ac_per_irr)) for t, i in zip(ts, irr) for inv in ("a", "b")])
    weather = _weather([(t, float(i), 25.0 + 5 * i, 25.0 + 20 * i) for t, i in zip(ts, irr)])
    return gen, weather


# ----------------------------------------------------------------- steps

def test_remove_duplicates_exact_and_conflicting():
    df = _gen([
        ("2020-05-15 10:00", "a", 1.0),
        ("2020-05-15 10:00", "a", 1.0),   # exact duplicate -> one kept
        ("2020-05-15 10:15", "a", 2.0),
        ("2020-05-15 10:15", "a", 3.0),   # conflicting -> both dropped
        ("2020-05-15 10:30", "a", 4.0),
    ])
    out, exact, conflicting = remove_duplicates(df, ["timestamp", "SOURCE_KEY"])
    assert exact == 1
    assert conflicting == 2
    assert out["AC_POWER"].tolist() == [1.0, 4.0]


def test_snap_to_grid_tolerance():
    df = pd.DataFrame({"timestamp": pd.to_datetime(
        ["2020-05-15 10:00", "2020-05-15 10:16", "2020-05-15 10:37"])})
    out, snapped, dropped = snap_to_grid(df, pd.Timedelta("2min"))
    assert out["timestamp"].tolist() == list(pd.to_datetime(["2020-05-15 10:00", "2020-05-15 10:15"]))
    assert (snapped, dropped) == (1, 1)


def test_output_is_regular_grid_and_joins_weather():
    gen, weather = _one_day()
    gen = gen[gen["timestamp"] != "2020-05-15 12:00"]       # drop a whole slot
    frame = preprocess(gen, weather, CFG).frame
    assert len(frame) == 96
    assert (frame.index.to_series().diff().dropna() == pd.Timedelta("15min")).all()
    assert frame.loc["2020-05-15 12:00", "n_reporting"] == 0
    assert np.isnan(frame.loc["2020-05-15 12:00", TARGET])  # daytime gap stays NaN


def test_aggregation_scales_mean_by_inverter_count():
    gen = _gen([
        ("2020-05-15 12:00", "a", 100.0), ("2020-05-15 12:00", "b", 300.0),
        ("2020-05-15 12:15", "a", 200.0),                     # b missing -> 200 * 2
    ])
    weather = _weather([("2020-05-15 12:00", 0.5, 30, 40), ("2020-05-15 12:15", 0.5, 30, 40)])
    frame = preprocess(gen, weather, CFG).frame
    assert frame.loc["2020-05-15 12:00", TARGET] == 400.0
    assert frame.loc["2020-05-15 12:15", TARGET] == 400.0
    assert frame.loc["2020-05-15 12:15", "inverter_coverage"] == 0.5


def test_low_coverage_target_set_nan():
    cfg = PreprocessConfig(n_inverters=4, min_inverter_coverage=0.5)
    gen = _gen([("2020-05-15 12:00", "a", 100.0)])            # 1 of 4 reporting
    weather = _weather([("2020-05-15 12:00", 0.5, 30, 40)])
    frame = preprocess(gen, weather, cfg).frame
    assert np.isnan(frame.loc["2020-05-15 12:00", TARGET])
    assert frame.loc["2020-05-15 12:00", "flag_low_coverage"]


def test_physical_bounds_and_night_inconsistency():
    gen = _gen([
        ("2020-05-15 12:00", "a", -5.0), ("2020-05-15 12:00", "b", -5.0),    # negative
        ("2020-05-15 12:15", "a", 1e6), ("2020-05-15 12:15", "b", 1e6),      # above capacity
        ("2020-05-15 23:00", "a", 50.0), ("2020-05-15 23:00", "b", 50.0),    # output at night
    ])
    weather = _weather([
        ("2020-05-15 12:00", 0.5, 30, 40),
        ("2020-05-15 12:15", 9.0, 30, 40),                   # irradiation out of range
        ("2020-05-15 23:00", 0.0, 22, 21),
    ])
    res = preprocess(gen, weather, CFG)
    f = res.frame
    assert np.isnan(f.loc["2020-05-15 12:00", TARGET])
    assert np.isnan(f.loc["2020-05-15 12:15", TARGET])
    assert f.loc["2020-05-15 12:15", "flag_bound_irradiation"]
    assert np.isnan(f.loc["2020-05-15 23:00", TARGET])
    assert f.loc["2020-05-15 23:00", "flag_night_inconsistent"]
    assert res.report.bound_violations[TARGET] == 2
    assert res.report.night_inconsistent == 1


def test_weather_ffill_limit_and_night_target_fill():
    gen, weather = _one_day()
    # Remove 4 consecutive weather slots at midday: 2 are filled, 2 stay NaN.
    gap = pd.date_range("2020-05-15 12:00", periods=4, freq="15min")
    weather = weather[~weather["timestamp"].isin(gap)]
    # Remove generation for a night slot (irradiation observed as 0) -> filled with 0.
    gen = gen[gen["timestamp"] != "2020-05-15 02:00"]
    res = preprocess(gen, weather, CFG)
    f = res.frame
    assert f.loc[gap[:2], "flag_imputed_irradiation"].all()
    assert f.loc[gap[:2], "irradiation"].tolist() == [f.loc["2020-05-15 11:45", "irradiation"]] * 2
    assert f.loc[gap[2:], "irradiation"].isna().all()
    assert f.loc["2020-05-15 02:00", TARGET] == 0.0
    assert f.loc["2020-05-15 02:00", "flag_imputed_target"]
    assert res.report.target_night_filled == 1


def test_stat_outlier_flagged_but_value_kept():
    days = [_one_day(day=f"2020-05-{d:02d}") for d in range(15, 23)]
    gen = pd.concat([g for g, _ in days], ignore_index=True)
    weather = pd.concat([w for _, w in days], ignore_index=True)
    # Small multiplicative noise so the trailing MAD is non-zero.
    rng = np.random.default_rng(0)
    gen["AC_POWER"] *= rng.uniform(0.98, 1.02, len(gen))
    target_ts = pd.Timestamp("2020-05-22 12:00")
    gen.loc[gen["timestamp"] == target_ts, "AC_POWER"] *= 0.2      # sudden 80% drop
    f = preprocess(gen, weather, CFG).frame
    assert f.loc[target_ts, "flag_stat_outlier"]
    assert f.loc[target_ts, TARGET] > 0                                # kept, not deleted
    assert f["flag_stat_outlier"].sum() == 1


def test_robust_trailing_z_excludes_current_value():
    idx = pd.date_range("2020-05-15", periods=30, freq="h")
    s = pd.Series(np.r_[np.tile([1.0, 1.1, 0.9], 9), 1.0, 1.0, 50.0], index=idx)
    z = robust_trailing_z(s, "7D", min_periods=5)
    assert z.iloc[-1] > 100            # the spike is not part of its own baseline
    assert z.iloc[:5].isna().all()     # not enough history yet


def test_causal_smooth_keeps_raw_and_is_trailing():
    s = pd.Series([0.0, 0.0, 10.0, 0.0, 0.0])
    sm = causal_smooth(s, 3)
    assert sm.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    s2 = s.copy()
    s2.iloc[4] = 99.0                  # changing the future must not change the past
    assert causal_smooth(s2, 3).iloc[:4].tolist() == sm.iloc[:4].tolist()


# ------------------------------------------------------------- causality

def test_pipeline_is_causal_prefix_invariant():
    """Output up to T must be identical whether or not data after T exists."""
    gen, weather = load_plant_raw(1)
    cutoff = pd.Timestamp("2020-06-01 13:30")
    full = preprocess(gen, weather).frame
    part = preprocess(
        gen[gen["timestamp"] <= cutoff], weather[weather["timestamp"] <= cutoff]
    ).frame
    assert part.index[-1] == cutoff
    pd.testing.assert_frame_equal(part, full.loc[:cutoff], check_freq=False)


# ------------------------------------------------------- split + scaler

def _hourly_frame(days=10):
    idx = pd.date_range("2020-05-15", periods=days * 24, freq="h", name="timestamp")
    return pd.DataFrame({"x": np.arange(len(idx), dtype=float)}, index=idx)


def test_chronological_split_whole_days_in_order():
    f = _hourly_frame(10)
    train, val, test = chronological_split(f, train_days=6, val_days=2)
    assert len(train) == 6 * 24 and len(val) == 2 * 24 and len(test) == 2 * 24
    assert train.index.max() < val.index.min() and val.index.max() < test.index.min()
    assert train.index[-1].hour == 23 and val.index[0].hour == 0
    pd.testing.assert_frame_equal(pd.concat([train, val, test]), f)


def test_chronological_split_rejects_bad_sizes():
    with pytest.raises(ValueError):
        chronological_split(_hourly_frame(5), train_days=4, val_days=1)


@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_scaler_uses_train_statistics_only(method):
    f = _hourly_frame(10)
    train, val, _ = chronological_split(f, 6, 2)
    scaler = TrainOnlyScaler(method).fit(train, ["x"])
    t = scaler.transform(train)["x"]
    if method == "standard":
        assert abs(t.mean()) < 1e-9 and abs(t.std(ddof=0) - 1) < 1e-9
    else:
        assert t.min() == 0 and t.max() == 1
    # Validation values are larger than anything in train: they must fall
    # outside the train-fitted range, proving val did not influence the fit.
    assert scaler.transform(val)["x"].min() > t.max()
    back = scaler.inverse_transform(scaler.transform(val))
    pd.testing.assert_frame_equal(back, val)


def test_scaler_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        TrainOnlyScaler().transform(_hourly_frame(1))


# ------------------------------------------------------------- real data

@pytest.mark.parametrize("plant", [1, 2])
def test_real_plant_loads_with_explicit_formats(plant):
    gen, weather = load_plant_raw(plant)
    assert gen["timestamp"].min() == pd.Timestamp("2020-05-15 00:00")
    assert gen["timestamp"].max() == pd.Timestamp("2020-06-17 23:45")
    assert weather["timestamp"].max() == pd.Timestamp("2020-06-17 23:45")
    assert gen["SOURCE_KEY"].nunique() == 22
    assert "HUMIDITY" not in {c.upper() for c in weather.columns}


def test_real_plant1_report_matches_profile():
    gen, weather = load_plant_raw(1)
    res = preprocess(gen, weather, plant=1)
    rep, f = res.report, res.frame
    assert rep.grid_slots == 3264                      # 34 days x 96
    assert rep.slots_without_generation == 106
    assert rep.slots_without_weather == 82
    assert rep.dc_scale_suspect and 10 < rep.dc_ac_ratio_median < 10.5
    assert "DC_POWER" not in f.columns                  # DC is not carried forward
    day = f[f["irradiation"] > 0.1]
    assert day[TARGET].corr(day["irradiation"]) > 0.95
