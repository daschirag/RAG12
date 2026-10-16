"""A6: POST /forecast (TestClient), request/response schemas, service consistency and causality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.backend import main
from src.backend.forecasting.features import build_features
from src.backend.forecasting.models import eval_mask, night_mask, postprocess
from src.backend.forecasting.preprocess import TARGET, load_plant_raw, preprocess
from src.backend.forecasting.schemas import ForecastRequest, ForecastResponse, ForecastResult
from src.backend.forecasting.service import (
    ForecastService,
    build_plant_model,
    deployment_split,
    save_plant_model,
)
from src.backend.forecasting.uncertainty import build_intervals

SMALL = {"max_depth": [3], "learning_rate": [0.1]}


@pytest.fixture(scope="module")
def frame1() -> pd.DataFrame:
    gen, weather = load_plant_raw(1)
    return preprocess(gen, weather, plant=1).frame


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, frame1):
    path = tmp_path_factory.mktemp("forecast_models")
    save_plant_model(build_plant_model(1, frame1, point_grid=SMALL, quantile_grid=SMALL), path)
    return path


@pytest.fixture(scope="module")
def service(model_dir, frame1) -> ForecastService:
    return ForecastService(model_dir=model_dir, frames={1: frame1})


@pytest.fixture
def client(service):
    main.app.dependency_overrides[main.get_forecast_service] = lambda: service
    try:
        yield TestClient(main.app)
    finally:
        main.app.dependency_overrides.pop(main.get_forecast_service, None)


# ------------------------------------------------------------------ schemas

def test_request_schema_validation():
    ok = ForecastRequest(plant=1, origin="2020-06-17T12:00")
    assert ok.confidence == 0.9
    assert ForecastRequest(plant=2, origin="2020-06-17T12:45", confidence=0.95).confidence == 0.95
    for bad in (
        {"plant": 3, "origin": "2020-06-17T12:00"},
        {"plant": 1, "origin": "2020-06-17T12:10"},             # off grid
        {"plant": 1, "origin": "2020-06-17T12:00:30"},          # seconds
        {"plant": 1, "origin": "2020-06-17T12:00+05:30"},       # timezone
        {"plant": 1, "origin": "2020-06-17T12:00", "confidence": 0.85},
        {"plant": 1, "origin": "2020-06-17T12:00", "confidence": 1.0},
    ):
        with pytest.raises(ValidationError):
            ForecastRequest(**bad)


def test_response_schema_ordering_and_confidence_description():
    ForecastResponse(prediction=5, lower_bound=1, upper_bound=9, confidence=0.9)
    with pytest.raises(ValidationError):
        ForecastResponse(prediction=10, lower_bound=1, upper_bound=9, confidence=0.9)
    with pytest.raises(ValidationError):
        ForecastResponse(prediction=5, lower_bound=1, upper_bound=9, confidence=1.2)
    desc = ForecastResult.model_fields["confidence"].description.lower()
    assert "nominal coverage" in desc and "not the probability" in desc


def test_openapi_documents_confidence_meaning(client):
    spec = client.get("/openapi.json").json()
    text = str(spec["components"]["schemas"]["ForecastResult"]).lower()
    assert "nominal coverage" in text


# ------------------------------------------------------------------ endpoint

def test_forecast_daytime_happy_path(client, frame1):
    resp = client.post("/forecast", json={"plant": 1, "origin": "2020-06-16T11:00", "confidence": 0.9})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"prediction", "lower_bound", "upper_bound", "confidence"}
    assert body["confidence"] == 0.9
    assert 0 <= body["lower_bound"] <= body["prediction"] <= body["upper_bound"]
    assert body["upper_bound"] > body["lower_bound"]                    # a real interval at midday
    assert pd.Timestamp(body["target_time"]) == pd.Timestamp("2020-06-16 12:00")
    assert body["observed_kw"] == pytest.approx(frame1.loc["2020-06-16 12:00", TARGET])
    assert body["in_sample"] is True and body["daylight_group"] is True
    assert body["method"] == "cqr_mondrian" and body["unit"] == "kW" and body["horizon_minutes"] == 60


def test_forecast_night_is_zero(client):
    body = client.post("/forecast", json={"plant": 1, "origin": "2020-06-16T01:00"}).json()
    assert body["prediction"] == body["lower_bound"] == body["upper_bound"] == 0.0


def test_forecast_beyond_the_record_is_out_of_sample(client):
    body = client.post("/forecast", json={"plant": 1, "origin": "2020-06-17T23:45"}).json()
    assert pd.Timestamp(body["target_time"]) == pd.Timestamp("2020-06-18 00:45")
    assert body["observed_kw"] is None and body["in_sample"] is False


def test_wider_confidence_gives_wider_interval(client):
    widths = []
    for level in (0.8, 0.9, 0.95):
        b = client.post("/forecast", json={"plant": 1, "origin": "2020-06-16T11:00", "confidence": level}).json()
        widths.append(b["upper_bound"] - b["lower_bound"])
    assert widths[0] <= widths[1] <= widths[2]


@pytest.mark.parametrize("payload", [
    {"plant": 1, "origin": "2020-06-16T11:10"},
    {"plant": 1, "origin": "2020-06-16T11:00", "confidence": 0.5},
    {"plant": 9, "origin": "2020-06-16T11:00"},
    {"origin": "2020-06-16T11:00"},
])
def test_invalid_requests_422(client, payload):
    assert client.post("/forecast", json=payload).status_code == 422


def test_origin_outside_record_422(client):
    resp = client.post("/forecast", json={"plant": 1, "origin": "2021-01-01T12:00"})
    assert resp.status_code == 422
    assert "between" in resp.json()["detail"]


def test_missing_model_returns_503(tmp_path, frame1):
    empty = ForecastService(model_dir=tmp_path, frames={1: frame1})
    main.app.dependency_overrides[main.get_forecast_service] = lambda: empty
    try:
        resp = TestClient(main.app).post("/forecast", json={"plant": 1, "origin": "2020-06-16T11:00"})
    finally:
        main.app.dependency_overrides.pop(main.get_forecast_service, None)
    assert resp.status_code == 503
    assert "build_forecast_model" in resp.json()["detail"]


def test_existing_endpoints_unaffected(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert "solar" in client.get("/domains").json()


# ------------------------------------------------- consistency and causality

def test_service_matches_offline_cqr_mondrian(service, frame1):
    """The API interval equals build_intervals' cqr_mondrian for the same models and calibration days."""
    model = service._model(1)
    parts = deployment_split(build_features(frame1))
    calib = parts["calib"]
    rows = build_features(frame1, drop_missing_target=False)
    origins = pd.DatetimeIndex(["2020-06-16 08:00", "2020-06-16 11:00", "2020-06-16 17:30"])
    target = type(calib)(X=rows.X.loc[origins], y=rows.y.loc[origins], meta=rows.meta.loc[origins],
                         config=rows.config)
    pc = postprocess(model.point.predict(calib, frame1), calib)
    pt = postprocess(model.point.predict(target, frame1), target)
    offline = build_intervals(pc, model.quantile.predict(calib), calib, pt, model.quantile.predict(target), target)
    lo, hi = offline.bounds[("cqr_mondrian", 0.9)]
    for i, t in enumerate(origins):
        r = service.forecast(1, t, 0.9)
        assert r["prediction"] == pytest.approx(pt[i])
        assert r["lower_bound"] == pytest.approx(lo[i]) and r["upper_bound"] == pytest.approx(hi[i])


def test_forecast_does_not_use_data_after_origin(model_dir, frame1):
    origin = pd.Timestamp("2020-06-16 11:00")
    dirty = frame1.copy()
    after = dirty.index > origin
    for col in (TARGET, "irradiation", "irradiation_smooth", "ambient_temp", "module_temp"):
        dirty.loc[after, col] = 9999.0
    clean_r = ForecastService(model_dir=model_dir, frames={1: frame1}).forecast(1, origin, 0.9)
    dirty_r = ForecastService(model_dir=model_dir, frames={1: dirty}).forecast(1, origin, 0.9)
    for key in ("prediction", "lower_bound", "upper_bound"):
        assert clean_r[key] == dirty_r[key]
    # Positive control: corrupting data up to and including the origin does change the forecast.
    past = frame1.copy()
    past.loc[(past.index > origin - pd.Timedelta("2h")) & (past.index <= origin), [TARGET, "irradiation"]] = 0.0
    past_r = ForecastService(model_dir=model_dir, frames={1: past}).forecast(1, origin, 0.9)
    assert past_r["prediction"] != clean_r["prediction"]


def test_deployment_split_ends_at_record(frame1):
    parts = deployment_split(build_features(frame1))
    last = {k: v.meta["target_time"].dt.normalize() for k, v in parts.items()}
    assert last["calib"].max() == pd.Timestamp("2020-06-17")
    assert last["calib"].nunique() == 4 and last["val"].nunique() == 4
    assert last["train"].max() < last["val"].min() and last["val"].max() < last["calib"].min()
    assert eval_mask(parts["calib"]).sum() > 300
    assert np.isfinite(night_mask(parts["calib"])).all()
