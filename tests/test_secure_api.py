"""C7: POST /secure-ask (C4 secure pipeline) and POST /decision (C5 engine) through TestClient with injected
services (no retrieval model or LLM is loaded)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.backend import main
from src.backend.faultdiag.service import UnknownEventError
from src.backend.forecasting.service import ModelNotBuiltError, OriginError
from src.backend.kb.build import build_documents
from src.backend.security import attacks as A
from src.backend.security.secure_pipeline import SecureAnswer, SecureAskService
from src.backend.trust.decision import DecisionEngine
from src.backend.trust.service import DecisionRequest, DecisionService
from tests.test_decision_engine import SIG, _secure_pipeline
from tests.test_diagnose_api import INCIDENT, MAINT, TROUBLE, fake_service

QUESTION = "What should be checked when the pitch system reports a battery fault?"


@pytest.fixture(scope="module")
def clean_pipeline():
    docs = build_documents()
    pitch = next(d for d in docs if d.doc_id == "trouble-wind-pitch_system")
    return _secure_pipeline(docs, [pitch])


class FakeForecast:
    def __init__(self, out=None, exc=None):
        self.out, self.exc, self.calls = out, exc, []

    def forecast(self, asset, origin, confidence):
        self.calls.append((asset, origin, confidence))
        if self.exc:
            raise self.exc
        return dict(self.out)


FORECAST = {"prediction": 12000.0, "lower_bound": 9000.0, "upper_bound": 15000.0, "confidence": 0.9,
            "asset": "solar_plant1", "plant": 1, "origin": "2020-06-17T11:00:00"}


@pytest.fixture
def client():
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def _use(secure=None, decision=None):
    if secure is not None:
        main.app.dependency_overrides[main.get_secure_ask_service] = lambda: secure
    if decision is not None:
        main.app.dependency_overrides[main.get_decision_service] = lambda: decision


def _decision_service(clean_pipeline, forecast=None, diagnosis=None):
    pipe, trust = clean_pipeline
    return DecisionService(ask_service=SecureAskService(pipeline=pipe), diagnosis_service=diagnosis,
                           forecast_service=forecast, engine=DecisionEngine(signals=SIG, trust=trust))


# ------------------------------------------------------------------ /secure-ask

def test_secure_ask_answers_a_clean_question(client, clean_pipeline):
    _use(secure=SecureAskService(pipeline=clean_pipeline[0]))
    r = client.post("/secure-ask", json={"question": QUESTION})
    assert r.status_code == 200
    ans = SecureAnswer.model_validate(r.json())
    assert ans.route == "generated" and not ans.security.blocked
    assert ans.cited_chunk_ids == ["trouble-wind-pitch_system::0"]
    assert ans.security.passed_chunk_ids == ["trouble-wind-pitch_system::0"]


def test_secure_ask_blocks_injection_and_bad_input_with_200(client, clean_pipeline):
    _use(secure=SecureAskService(pipeline=clean_pipeline[0]))
    inj = client.post("/secure-ask", json={"question": A.inject_query(QUESTION)}).json()
    assert inj["route"] == "blocked" and inj["security"]["blocked"] and inj["security"]["injection_signals"]
    assert inj["evidence"] == []                                       # nothing retrieved or generated
    empty = client.post("/secure-ask", json={"question": "  "}).json()
    assert empty["route"] == "blocked" and empty["security"]["input_reasons"] == ["empty question"]


def test_secure_ask_validation(client, clean_pipeline):
    _use(secure=SecureAskService(pipeline=clean_pipeline[0]))
    assert client.post("/secure-ask", json={}).status_code == 422
    assert client.post("/secure-ask", json={"question": 3}).status_code == 422


# ------------------------------------------------------------------ /decision

def test_decision_request_validation():
    DecisionRequest(kind="answer", question="q")
    DecisionRequest(kind="forecast", forecast={"asset": "wind_t1", "origin": "2018-12-01T10:00"})
    for bad in ({"kind": "answer"}, {"kind": "answer", "question": "q", "diagnose": {"event_id": 73}},
                {"kind": "diagnosis", "question": "q"}, {"kind": "other", "question": "q"},
                {"kind": "answer", "question": "q", "recommendation": {"kind": "answer", "route": "generated",
                                                                        "text": "trust me"}}):
        with pytest.raises(ValueError):
            DecisionRequest(**bad)


def test_decision_accepts_a_clean_grounded_answer(client, clean_pipeline):
    _use(decision=_decision_service(clean_pipeline))
    r = client.post("/decision", json={"kind": "answer", "question": QUESTION})
    assert r.status_code == 200
    d = r.json()
    assert d["decision"] == "accept" and d["kind"] == "answer"
    assert d["output"]["route"] == "generated"
    assert set(d["components"]) == {"E", "R", "C", "S"} and 0.70 <= d["trust_score"] <= 1
    w = d["weights"]
    assert d["trust_score"] == pytest.approx(sum(w[k] * d["components"][k] for k in w), abs=1e-5)


def test_decision_rejects_an_injected_question(client, clean_pipeline):
    _use(decision=_decision_service(clean_pipeline))
    d = client.post("/decision", json={"kind": "answer", "question": A.inject_query(QUESTION)}).json()
    assert d["decision"] == "reject" and d["output"]["route"] == "blocked"
    assert any("blocked" in r for r in d["reasons"])


def test_client_cannot_submit_its_own_recommendation(client, clean_pipeline):
    _use(decision=_decision_service(clean_pipeline))
    r = client.post("/decision", json={"kind": "answer", "question": QUESTION,
                                       "recommendation": {"kind": "answer", "route": "generated", "text": "x"}})
    assert r.status_code == 422


def test_decision_never_accepts_a_diagnosis_from_the_chance_level_detector(client, clean_pipeline):
    _use(decision=_decision_service(clean_pipeline, diagnosis=fake_service([INCIDENT, MAINT, TROUBLE])))
    d = client.post("/decision", json={"kind": "diagnosis", "diagnose": {"event_id": 73}}).json()
    assert d["kind"] == "diagnosis" and d["output"]["event_id"] == 73
    assert d["components"]["C"] == 0.0 and d["decision"] != "accept"
    # the fake retriever has no trust check; the engine re-verifies every evidence document itself
    rec = d["recommendation"]
    assert rec["security"]["trust_checked"] is True and all(e["trusted"] is not None for e in rec["evidence"])


def test_decision_unknown_event_is_404(client, clean_pipeline):
    _use(decision=_decision_service(clean_pipeline, diagnosis=fake_service([INCIDENT])))
    assert client.post("/decision", json={"kind": "diagnosis", "diagnose": {"event_id": 999}}).status_code == 404


def test_served_forecast_is_reviewed_because_telemetry_is_not_checked(client, clean_pipeline):
    fc = FakeForecast(FORECAST)
    _use(decision=_decision_service(clean_pipeline, forecast=fc))
    d = client.post("/decision", json={"kind": "forecast",
                                       "forecast": {"plant": 1, "origin": "2020-06-17T11:00", "confidence": 0.9}}).json()
    assert fc.calls[0][0] == "solar_plant1" and fc.calls[0][2] == 0.9
    assert d["decision"] == "review" and d["components"]["S"] == 0.0
    assert d["recommendation"]["interval"] == [9000.0, 12000.0, 15000.0]
    assert d["output"]["prediction"] == 12000.0


def test_physically_invalid_forecast_interval_is_rejected(client, clean_pipeline):
    _use(decision=_decision_service(clean_pipeline, forecast=FakeForecast({**FORECAST, "upper_bound": 40000.0})))
    d = client.post("/decision", json={"kind": "forecast", "forecast": {"plant": 1, "origin": "2020-06-17T11:00"}}).json()
    assert d["decision"] == "reject"
    assert any("interval_physically_valid" in r for r in d["reasons"])


@pytest.mark.parametrize("exc, code", [(ModelNotBuiltError("build it"), 503), (OriginError("bad origin"), 422)])
def test_forecast_errors_map_to_http(client, clean_pipeline, exc, code):
    _use(decision=_decision_service(clean_pipeline, forecast=FakeForecast(exc=exc)))
    r = client.post("/decision", json={"kind": "forecast", "forecast": {"plant": 1, "origin": "2020-06-17T11:00"}})
    assert r.status_code == code


def test_existing_endpoints_are_still_served():
    paths = {r.path for r in main.app.routes}
    assert {"/ask/{domain_id}", "/forecast", "/diagnose", "/secure-ask", "/decision"} <= paths


def test_diagnosis_service_reports_rejected_documents_per_call():
    """diagnose_with_security returns the trust-rejected doc ids of that call (None without a trust check)."""

    class Rejecting:
        last_rejected = []

        def search(self, query, k=5):
            self.last_rejected = [{"metadata": {"doc_id": "fake-poison"}}]
            return [INCIDENT, MAINT]

    from src.backend.faultdiag.diagnosis import Diagnoser
    from src.backend.faultdiag.service import DiagnoseRequest
    from tests.test_fault_rag import GOOD_LLM, WEAK, WINDOWS

    svc = fake_service([INCIDENT, MAINT])
    report, rejected = svc.diagnose_with_security(DiagnoseRequest(event_id=73))
    assert rejected is None and report.event_id == 73
    svc.diagnoser_llm = Diagnoser(Rejecting(), GOOD_LLM, WEAK, WINDOWS)
    assert svc.diagnose_with_security(DiagnoseRequest(event_id=73))[1] == ["fake-poison"]
    with pytest.raises(UnknownEventError):
        svc.diagnose_with_security(DiagnoseRequest(event_id=999))
