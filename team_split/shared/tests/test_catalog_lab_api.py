"""Catalog endpoints (events, forecast assets/series, suggestions, chunks) and the security lab."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.backend import main
from src.backend.forecasting.service import ForecastService
from src.backend.security.secure_pipeline import SecureRetrieval

RES = main._PROJECT_ROOT / "results"


@pytest.fixture
def client():
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def forecast_service():
    return ForecastService()


# ------------------------------------------------------------------ events / suggestions / chunks

def test_diagnose_events_list_farm_a_with_detector_scores(client):
    ev = client.get("/diagnose/events").json()
    assert len(ev) == 22 and {e["label"] for e in ev} == {"anomaly", "normal"}
    pred = pd.read_csv(RES / "faultdiag" / "b4_event_predictions.csv")
    lr = pred[(pred.model == "logistic_regression") & (pred.seed == 0)].set_index("event_id")
    e73 = next(e for e in ev if e["event_id"] == 73)
    assert e73["detector_score"] == pytest.approx(lr.loc[73, "event_prob"])
    assert all(e["component"] is None for e in ev if e["label"] == "normal")


def test_suggestions_are_labelled_kb_queries(client):
    s = client.get("/assistant/suggestions", params={"per_type": 2}).json()
    types = [x["query_type"] for x in s]
    assert all(types.count(t) <= 2 for t in types) and len(set(types)) >= 6
    assert client.get("/assistant/suggestions", params={"per_type": 9}).status_code == 422


def test_kb_chunks_returns_text_of_known_chunks(client):
    class Doc:
        doc_id, title, doc_type, synthetic = "fault-a-73", "Fault event A-73", "fault_description", False

    sec = SecureRetrieval(None, [{"chunk_id": "fault-a-73::0", "text": "Hydraulic group", "metadata": {"doc_id": "fault-a-73"}}],
                          [Doc()], None, {})
    main.app.dependency_overrides[main.get_secure_retrieval] = lambda: sec
    r = client.get("/kb/chunks", params={"ids": "fault-a-73::0,unknown::9"})
    assert r.status_code == 200 and r.json() == [{"chunk_id": "fault-a-73::0", "doc_id": "fault-a-73",
                                                  "title": "Fault event A-73", "doc_type": "fault_description",
                                                  "synthetic": False, "text": "Hydraulic group"}]
    assert client.get("/kb/chunks", params={"ids": "nope::1"}).status_code == 404


# ------------------------------------------------------------------ forecast assets / series (real models)

def test_forecast_assets(client, forecast_service):
    main.app.dependency_overrides[main.get_forecast_service] = lambda: forecast_service
    a = {x["asset"]: x for x in client.get("/forecast/assets").json()}
    assert set(a) == {"solar_plant1", "solar_plant2", "wind_t1"}
    assert a["wind_t1"]["step_minutes"] == 10 and a["solar_plant1"]["step_minutes"] == 15
    assert a["solar_plant1"]["segments"]["calib"][1] == "2020-06-17"


def test_forecast_series_matches_single_forecasts(client, forecast_service):
    main.app.dependency_overrides[main.get_forecast_service] = lambda: forecast_service
    r = client.get("/forecast/series", params={"asset": "solar_plant1", "start": "2020-06-16T06:00",
                                               "end": "2020-06-16T18:00", "confidence": 0.9}).json()
    pts = r["points"]
    assert len(pts) == 49 and 0 <= r["coverage_observed"] <= 1
    for p in pts[::6]:
        f = forecast_service.forecast("solar_plant1", pd.Timestamp(p["origin"]), 0.9)
        assert p["prediction"] == pytest.approx(f["prediction"], abs=1e-6)
        # batched float32 quantile prediction: bounds agree to well under 0.01 kW
        assert p["lower_bound"] == pytest.approx(f["lower_bound"], abs=0.01)
        assert p["upper_bound"] == pytest.approx(f["upper_bound"], abs=0.01)
        assert p["in_sample"] == f["in_sample"]


def test_forecast_series_validation(client, forecast_service):
    main.app.dependency_overrides[main.get_forecast_service] = lambda: forecast_service
    base = {"asset": "wind_t1", "start": "2018-12-03T00:00"}
    assert client.get("/forecast/series", params={**base, "end": "2018-12-10T00:00"}).status_code == 422  # > 3 days
    assert client.get("/forecast/series", params={**base, "end": "2018-12-04T00:00", "confidence": 0.5}).status_code == 422
    out = client.get("/forecast/series", params={"asset": "wind_t1", "start": "2030-01-01T00:00", "end": "2030-01-01T05:00"})
    assert out.status_code == 422 and "record" in out.json()["message"]
    assert client.get("/forecast/series", params={**base, "asset": "solar_plant9", "end": "2018-12-03T05:00"}).status_code == 422


# ------------------------------------------------------------------ security lab

def test_lab_catalogue_reports_the_recorded_weakness(client):
    c = client.get("/security/lab/catalogue").json()
    assert len(c["held_out_phrasings"]) == 15 and len(c["documents"]) == 9
    held = c["held_out_engine"]
    assert held["accepted_that_obeyed"]["k"] <= held["accepted"]["k"] <= held["delivered"]
    assert {t["attack"] for t in c["telemetry"]} == {"replay", "power_scale", "irradiation_bias",
                                                     "module_temp_spike", "wind_speed_bias"}


def test_lab_injection_blocks_and_rejects_the_c2_template(client):
    from src.backend.security.attacks import inject_query

    r = client.post("/security/lab/injection", json={"question": inject_query("Which checks apply to the gearbox?")}).json()
    assert r["blocked"] and r["injection_signals"] and r["engine"]["decision"] == "reject"
    clean = client.post("/security/lab/injection", json={"question": "Which checks apply to the gearbox?"}).json()
    assert clean["blocked"] is False and clean["engine"] is None and "run_pipeline" in clean["note"]
    assert client.post("/security/lab/injection", json={"question": ""}).status_code == 422


def test_lab_poisoning_needs_the_registry_for_the_unsafe_procedure(client):
    r = client.post("/security/lab/poisoning", json={"doc_id": "trouble-wind-pitch_system-rev2"}).json()
    assert r["full_check"]["trusted"] is False and r["content_only"]["trusted"] is True
    assert r["genuine_original"] == {**r["genuine_original"], "doc_id": "trouble-wind-pitch_system", "trusted": True}
    assert r["attack"] == "unsafe_procedure" and r["recorded_c6"]
    inj = client.post("/security/lab/poisoning", json={"doc_id": "fault-a-26-log"}).json()
    assert inj["attack"] == "prompt_injection_kb" and inj["content_only"]["trusted"] is False
    assert client.post("/security/lab/poisoning", json={"doc_id": "fault-a-73"}).status_code == 404


def test_lab_telemetry_validation(client):
    r = client.post("/security/lab/telemetry", json={"asset": "wind_t1", "attack": "power_scale"})
    assert r.status_code == 422 and "not defined for wind_t1" in r.json()["message"]


@pytest.mark.integration
def test_lab_telemetry_wind_replay_matches_the_recorded_c3_verdicts(client):
    r = client.post("/security/lab/telemetry", json={"asset": "wind_t1", "attack": "replay"}).json()
    c3 = pd.read_csv(RES / "security" / "c3_telemetry_days.csv")
    rec = c3[(c3.asset == "wind_t1") & (c3.attack == "replay") & (c3.seed == 0)]
    assert len(r["days"]) == len(rec) and r["all_match_recorded"]
    assert r["flagged_days"] == int(rec["any"].sum())
    assert all(d["engine"]["decision"] == "reject" for d in r["days"] if d["live_flagged"] and d["engine"])
    assert len(r["signal"]["clean"]) == len(r["signal"]["attacked"]) == len(r["signal"]["times"])
