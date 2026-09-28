"""A2 features: exact alignment, no future information, split by target day."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backend.forecasting.features import (
    FEATURE_DESCRIPTIONS,
    FeatureConfig,
    build_features,
    irradiation_envelope,
    split_by_target_day,
)
from src.backend.forecasting.preprocess import TARGET, load_plant_raw, preprocess


@pytest.fixture(scope="module")
def plant1_frame() -> pd.DataFrame:
    gen, weather = load_plant_raw(1)
    return preprocess(gen, weather, plant=1).frame


@pytest.fixture(scope="module")
def plant1_set(plant1_frame):
    return build_features(plant1_frame)


def _corrupt_after(frame: pd.DataFrame, t: pd.Timestamp, seed: int) -> pd.DataFrame:
    """Replace every numeric value strictly after t with random garbage."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    future = out.index > t
    for col in out.columns:
        if out[col].dtype == bool:
            out.loc[future, col] = rng.random(future.sum()) > 0.5
        else:
            out[col] = out[col].astype(float)
            out.loc[future, col] = rng.uniform(0, 5000, future.sum())
    return out


def test_default_features_match_documentation(plant1_set):
    assert plant1_set.feature_names == list(FEATURE_DESCRIPTIONS)
    assert not any(n.startswith("oracle_") for n in plant1_set.feature_names)


def test_target_is_exactly_one_hour_ahead(plant1_frame, plant1_set):
    assert plant1_set.config.horizon == 4
    meta, y = plant1_set.meta, plant1_set.y
    assert ((meta["target_time"] - meta.index) == pd.Timedelta("1h")).all()
    np.testing.assert_array_equal(y.to_numpy(), plant1_frame.loc[meta["target_time"], TARGET].to_numpy())


def test_lag_and_rolling_values_are_exact(plant1_frame, plant1_set):
    y = plant1_frame[TARGET]
    t = pd.Timestamp("2020-06-03 11:00")
    row = plant1_set.X.loc[t]
    assert row["ac_now"] == y.loc[t]
    for k in (1, 3, 6, 24):
        assert row[f"ac_lag_{k}"] == y.loc[t - k * pd.Timedelta("15min")]
    window = y.loc[t - pd.Timedelta("45min"): t]
    assert len(window) == 4
    assert row["ac_roll_mean_4"] == pytest.approx(window.mean())
    assert row["ac_roll_std_4"] == pytest.approx(window.std())
    assert row["ac_same_slot_yesterday"] == y.loc[t + pd.Timedelta("1h") - pd.Timedelta("1D")]
    assert row["hour"] == 12  # calendar of the target time, not the origin


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_future_information_in_features(plant1_frame, seed):
    """Corrupting everything after origin t must leave row t's features unchanged."""
    rng = np.random.default_rng(seed)
    daytime = plant1_frame.index[(plant1_frame.index.hour >= 7) & (plant1_frame.index.hour <= 17)]
    t = daytime[rng.integers(200, len(daytime) - 200)]
    clean = build_features(plant1_frame).X
    dirty = build_features(_corrupt_after(plant1_frame, t, seed)).X
    pd.testing.assert_series_equal(clean.loc[t], dirty.loc[t])
    # Earlier origins are untouched as well.
    pd.testing.assert_frame_equal(clean.loc[:t], dirty.loc[:t])


def test_oracle_weather_is_opt_in_and_does_use_the_future(plant1_frame):
    t = pd.Timestamp("2020-06-03 11:00")
    cfg = FeatureConfig(include_oracle_weather=True)
    clean = build_features(plant1_frame, cfg).X
    dirty = build_features(_corrupt_after(plant1_frame, t, 0), cfg).X
    # Sanity check that the leakage test can detect future information.
    assert clean.loc[t, "oracle_irr_target"] != dirty.loc[t, "oracle_irr_target"]
    shared = [c for c in clean.columns if not c.startswith("oracle_")]
    pd.testing.assert_series_equal(clean.loc[t, shared], dirty.loc[t, shared])


def test_irradiation_envelope_uses_previous_days_only():
    idx = pd.date_range("2020-05-15", periods=3 * 96, freq="15min")
    irr = pd.Series(0.0, index=idx)
    noon = idx[(idx.hour == 12) & (idx.minute == 0)]
    irr.loc[noon] = [0.5, 0.9, 0.7]
    env = irradiation_envelope(irr)
    assert np.isnan(env.loc[noon[0]])        # no previous day
    assert env.loc[noon[1]] == 0.5           # excludes today's 0.9
    assert env.loc[noon[2]] == 0.9


def test_split_by_target_day_is_chronological_and_disjoint(plant1_set):
    parts = split_by_target_day(plant1_set)
    assert list(parts) == ["train", "val", "calib", "test"]
    days = {k: v.meta["target_time"].dt.normalize().unique() for k, v in parts.items()}
    assert [len(d) for d in days.values()] == [21, 4, 4, 5]
    order = ["train", "val", "calib", "test"]
    for a, b in zip(order, order[1:]):
        assert parts[a].meta["target_time"].max() < parts[b].meta["target_time"].min()
    # No training target lies in a later segment.
    assert parts["train"].meta["target_time"].max() < pd.Timestamp("2020-06-05")
    assert sum(len(p.y) for p in parts.values()) == len(plant1_set.y)


def test_split_rejects_too_many_days(plant1_set):
    with pytest.raises(ValueError):
        split_by_target_day(plant1_set, (("train", 30), ("val", 4)))


def test_rows_with_missing_target_are_dropped(plant1_set):
    assert plant1_set.y.notna().all()


def test_irregular_grid_rejected(plant1_frame):
    with pytest.raises(ValueError):
        build_features(plant1_frame.drop(plant1_frame.index[10]))
