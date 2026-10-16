"""Wind T1 replication: preprocessing, features, folds, generalized models/intervals, API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.backend import main
from src.backend.forecasting.models import (
    SequenceEncoder,
    XGBoost,
    eval_mask,
    postprocess,
)
from src.backend.forecasting.preprocess import TARGET
from src.backend.forecasting.service import ForecastService, build_wind_model, save_plant_model
from src.backend.forecasting.uncertainty import (
    LEVELS,
    QuantileXGBoost,
    build_intervals,
    quantiles_for,
    split_conformal_interval,
)
from src.backend.forecasting.wind import (
    RATED_KW,
    WIND_CAPACITY_KW,
    WIND_CONTEXT_COLS,
    WIND_FEATURES,
    WIND_SEQ_COLUMNS,
    WindFeatureConfig,
    build_wind_features,
    deployment_month_split,
    load_wind_raw,
    power_regime,
    preprocess_wind,
    rolling_origin_month_folds,
)

SMALL = {"max_depth": [3], "learning_rate": [0.1]}


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_wind_raw()


@pytest.fixture(scope="module")
def prep(raw):
    return preprocess_wind(raw)


@pytest.fixture(scope="module")
def frame(prep) -> pd.DataFrame:
    return prep[0]


@pytest.fixture(scope="module")
def data(frame):
    return build_wind_features(frame)


# ------------------------------------------------------------- preprocessing

def test_load_parses_bom_and_dates(raw):
    assert len(raw) == 50530
    assert raw["timestamp"].min() == pd.Timestamp("2018-01-01 00:00")
    assert raw["timestamp"].max() == pd.Timestamp("2018-12-31 23:50")
    assert set(raw.columns) == {"timestamp", TARGET, "wind_speed", "wind_dir", "theoretical_power"}


def test_preprocess_grid_and_counts(prep):
    frame, rep = prep
    assert rep.grid_slots == 52560 == len(frame)                     # 365 days x 144
    assert (frame.index.to_series().diff().dropna() == pd.Timedelta("10min")).all()
    assert rep.slots_missing == 2030
    assert rep.negative_clipped == 57 and (frame[TARGET].dropna() >= 0).all()
    assert rep.longest_gap == "4 days 08:20:00"


def test_preprocess_rules_on_synthetic_rows():
    ts = pd.date_range("2018-01-01", periods=8, freq="10min")
    raw = pd.DataFrame({"timestamp": ts,
                        TARGET: [-5.0, -50.0, 100.0, 5000.0, 0.0, 200.0, 300.0, 400.0],
                        "wind_speed": [5, 5, 6, 12, 12, np.nan, np.nan, np.nan],
                        "wind_dir": [10.0] * 8,
                        "theoretical_power": [100, 100, 150, 3600, 3000, 200, 200, 200]})
    frame, rep = preprocess_wind(raw.dropna(subset=[TARGET]))
    f = frame.loc[ts]
    assert f[TARGET].iloc[0] == 0.0 and f["flag_negative_clipped"].iloc[0]       # small negative -> 0
    assert np.isnan(f[TARGET].iloc[1]) and f["flag_bound_ac_power_kw"].iloc[1]   # large negative -> NaN
    assert np.isnan(f[TARGET].iloc[3])                                           # above capacity -> NaN
    assert f["flag_turbine_down"].iloc[4]                                        # 0 kW while curve says 3000
    assert f["wind_speed"].iloc[5:7].tolist() == [12.0, 12.0] and np.isnan(f["wind_speed"].iloc[7])  # ffill <= 2


def test_preprocess_is_causal_prefix_invariant(raw):
    cutoff = pd.Timestamp("2018-06-15 13:30")
    full, _ = preprocess_wind(raw)
    part, _ = preprocess_wind(raw[raw["timestamp"] <= cutoff])
    pd.testing.assert_frame_equal(part, full.loc[:cutoff], check_freq=False)


# ------------------------------------------------------------------ features

def test_feature_names_and_one_hour_target(frame, data):
    assert data.feature_names == list(WIND_FEATURES)
    assert data.config.horizon == 6
    assert ((data.meta["target_time"] - data.meta.index) == pd.Timedelta("1h")).all()
    np.testing.assert_array_equal(data.y.to_numpy(), frame.loc[data.meta["target_time"], TARGET].to_numpy())
    t = pd.Timestamp("2018-07-10 11:00")
    row = data.X.loc[t]
    for k in (1, 3, 6, 24):
        assert row[f"ac_lag_{k}"] == frame.loc[t - k * pd.Timedelta("10min"), TARGET]
    assert row["ws_now"] == frame.loc[t, "wind_speed"]
    assert row["hour"] == 12


@pytest.mark.parametrize("seed", [0, 1])
def test_no_future_information_with_positive_control(frame, seed):
    rng = np.random.default_rng(seed)
    t = frame.index[rng.integers(5000, len(frame) - 5000)]
    dirty = frame.copy()
    after = dirty.index > t
    for col in (TARGET, "wind_speed", "wind_dir", "theoretical_power", "wind_speed_smooth"):
        dirty.loc[after, col] = rng.uniform(0, 3000, after.sum())
    cfg = WindFeatureConfig(include_oracle_weather=True)
    clean = build_wind_features(frame, cfg, drop_missing_target=False).X
    bad = build_wind_features(dirty, cfg, drop_missing_target=False).X
    shared = [c for c in clean.columns if not c.startswith("oracle_")]
    pd.testing.assert_frame_equal(clean.loc[:t, shared], bad.loc[:t, shared])
    assert clean.loc[t, "oracle_ws_target"] != bad.loc[t, "oracle_ws_target"]   # the test can see the future


def test_month_folds_chronological_disjoint(data):
    folds = list(rolling_origin_month_folds(data, ["2018-09", "2018-12"]))
    assert [str(m) for m, _ in folds] == ["2018-09", "2018-12"]
    for month, parts in folds:
        order = ["train", "val", "calib", "test"]
        for a, b in zip(order, order[1:]):
            assert parts[a].meta["target_time"].max() < parts[b].meta["target_time"].min()
        assert (parts["test"].meta["target_time"].dt.to_period("M") == month).all()
        assert parts["calib"].meta["target_time"].dt.to_period("M").nunique() == 1
    with pytest.raises(ValueError):
        list(rolling_origin_month_folds(data, ["2018-02"]))


def test_deployment_month_split(data):
    parts = deployment_month_split(data)
    months = {k: sorted(v.meta["target_time"].dt.to_period("M").astype(str).unique()) for k, v in parts.items()}
    assert months["calib"] == ["2018-12"] and months["val"] == ["2018-11"] and months["train"][-1] == "2018-10"


def test_power_regime_edges():
    r = power_regime(np.array([0, 359.9, 360, 3239.9, 3240, RATED_KW]))
    assert r.tolist() == ["low", "low", "mid", "mid", "high", "high"]


# ------------------------------------------------- generalized shared code

def test_postprocess_without_night_rule_only_clips(data):
    sub = type(data)(X=data.X.iloc[:500], y=data.y.iloc[:500], meta=data.meta.iloc[:500], config=data.config)
    raw_pred = np.linspace(-100, 5000, 500)
    out = postprocess(raw_pred, sub, None, WIND_CAPACITY_KW)
    np.testing.assert_allclose(out, np.clip(raw_pred, 0, WIND_CAPACITY_KW))


def test_sequence_encoder_wind_columns_causal(frame, data):
    parts = dict(rolling_origin_month_folds(data, ["2018-09"]))[pd.Period("2018-09", "M")]
    enc = SequenceEncoder(lookback=24, seq_columns=WIND_SEQ_COLUMNS, context_cols=WIND_CONTEXT_COLS)
    enc.fit(parts["train"], frame)
    test = parts["test"]
    seq, ctx = enc.transform(test, frame)
    assert seq.shape == (len(test.y), 24, len(WIND_SEQ_COLUMNS) + 4) and ctx.shape[1] == len(WIND_CONTEXT_COLS)
    i = 500
    t = test.X.index[i]
    dirty = frame.copy()
    dirty.loc[dirty.index > t, list(WIND_SEQ_COLUMNS)] = 9999.0
    np.testing.assert_array_equal(seq[i], enc.transform(test, dirty)[0][i])


def test_mondrian_groups_are_honoured():
    """Regime groups with different noise: per-group calibration gives each group its own width."""
    rng = np.random.default_rng(5)
    n = 3000
    groups_cal = np.where(rng.random(n) < 0.5, "low", "mid")
    groups_te = np.where(rng.random(n) < 0.5, "low", "mid")
    y_cal = np.where(groups_cal == "low", rng.normal(0, 5, n), rng.normal(0, 500, n))
    lo, hi = split_conformal_interval(np.zeros(n), y_cal, np.zeros(n), 0.9, groups_cal, groups_te)
    w = hi - lo
    assert w[groups_te == "low"].max() < 50 < w[groups_te == "mid"].min()


# ------------------------------------------------------------------- API

@pytest.fixture(scope="module")
def wind_service(tmp_path_factory, frame):
    path = tmp_path_factory.mktemp("wind_models")
    save_plant_model(build_wind_model(frame, point_grid=SMALL, quantile_grid=SMALL), path)
    return ForecastService(model_dir=path, frames={"wind_t1": frame})


@pytest.fixture
def client(wind_service):
    main.app.dependency_overrides[main.get_forecast_service] = lambda: wind_service
    try:
        yield TestClient(main.app)
    finally:
        main.app.dependency_overrides.pop(main.get_forecast_service, None)


def test_wind_forecast_endpoint(client, frame):
    resp = client.post("/forecast", json={"asset": "wind_t1", "origin": "2018-12-15T12:10", "confidence": 0.9})
    assert resp.status_code == 200
    b = resp.json()
    assert b["asset"] == "wind_t1" and b["plant"] is None and b["daylight_group"] is None
    assert b["calibration_group"].startswith("power regime ")
    assert 0 <= b["lower_bound"] <= b["prediction"] <= b["upper_bound"] <= WIND_CAPACITY_KW
    assert pd.Timestamp(b["target_time"]) == pd.Timestamp("2018-12-15 13:10")
    assert b["observed_kw"] == pytest.approx(frame.loc["2018-12-15 13:10", TARGET])
    assert b["in_sample"] is True and b["horizon_minutes"] == 60


def test_wind_last_origin_is_out_of_sample(client):
    b = client.post("/forecast", json={"asset": "wind_t1", "origin": "2018-12-31T23:50"}).json()
    assert b["observed_kw"] is None and b["in_sample"] is False


@pytest.mark.parametrize("payload", [
    {"asset": "wind_t1", "origin": "2018-12-15T12:05"},                     # not on the 10-min grid
    {"asset": "solar_plant1", "origin": "2020-06-16T11:10"},                # 10-min slot is off the solar grid
    {"plant": 1, "asset": "wind_t1", "origin": "2018-12-15T12:10"},         # both given
    {"asset": "wind_t2", "origin": "2018-12-15T12:10"},                     # unknown asset
])
def test_asset_validation_422(client, payload):
    assert client.post("/forecast", json=payload).status_code == 422


def test_wind_api_matches_offline_intervals_and_is_causal(wind_service, frame, data):
    model = wind_service._model("wind_t1")
    calib = deployment_month_split(data)["calib"]
    rows = build_wind_features(frame, drop_missing_target=False)
    origins = pd.DatetimeIndex(["2018-12-10 03:00", "2018-12-15 12:10", "2018-12-20 18:40"])
    target = type(calib)(X=rows.X.loc[origins], y=rows.y.loc[origins], meta=rows.meta.loc[origins],
                         config=rows.config)
    pc = postprocess(model.point.predict(calib, frame), calib, None, WIND_CAPACITY_KW)
    pt = postprocess(model.point.predict(target, frame), target, None, WIND_CAPACITY_KW)
    offline = build_intervals(pc, model.quantile.predict(calib), calib, pt, model.quantile.predict(target), target,
                              LEVELS, None, power_regime(pc), power_regime(pt), WIND_CAPACITY_KW)
    lo, hi = offline.bounds[("cqr_mondrian", 0.9)]
    for i, t in enumerate(origins):
        r = wind_service.forecast("wind_t1", t, 0.9)
        assert r["prediction"] == pytest.approx(pt[i])
        assert r["lower_bound"] == pytest.approx(lo[i]) and r["upper_bound"] == pytest.approx(hi[i])
    # Causality: corrupting data after the origin leaves the forecast unchanged.
    t = origins[1]
    dirty = frame.copy()
    dirty.loc[dirty.index > t, [TARGET, "wind_speed", "theoretical_power", "wind_speed_smooth"]] = 9999.0
    other = ForecastService(model_dir=wind_service.model_dir, frames={"wind_t1": dirty}).forecast("wind_t1", t, 0.9)
    assert other["prediction"] == wind_service.forecast("wind_t1", t, 0.9)["prediction"]


def test_quantile_and_point_models_fit_with_wind_settings(data, frame):
    parts = deployment_month_split(data)
    sub = {k: type(v)(X=v.X.iloc[::20], y=v.y.iloc[::20], meta=v.meta.iloc[::20], config=v.config)
           for k, v in parts.items()}
    xgb = XGBoost(seed=0, night_threshold=None, capacity=WIND_CAPACITY_KW)
    xgb.grid = SMALL
    xgb.fit(sub["train"], sub["val"], frame)
    qr = QuantileXGBoost(quantiles_for(LEVELS), seed=0)
    qr.grid = SMALL
    qr.fit(sub["train"], sub["val"])
    assert np.isfinite(xgb.predict(sub["calib"], frame)).all()
    assert (np.diff(qr.predict(sub["calib"]).to_numpy(), axis=1) >= 0).all()          # no crossing
    assert eval_mask(sub["calib"]).all()                                               # no imputed targets
