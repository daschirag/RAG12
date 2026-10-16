"""I1: orchestrator (POST /assistant) - intent routing, ablation configs A-F, trust only in F. Fakes only."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.backend import main
from src.backend.faultdiag.service import UnknownEventError
from src.backend.orchestrator import ABLATION, AssistantRequest, Orchestrator, route_intent
from src.backend.trust.service import DecisionResponse

FORECAST = {"prediction": 12000.0, "lower_bound": 9000.0, "upper_bound": 15000.0, "confidence": 0.9,
            "asset": "solar_plant1", "target_time": "2020-06-17T12:00:00", "observed_kw": 11500.0}
REPORT = {"event_id": 73, "diagnosis": "Anomaly on turbine 0.", "severity": "undetermined",
          "likely_cause": {"text": "hydraulic group", "chunk_ids": []},
          "recommended_action": {"text": "inspect the hydraulic system", "chunk_ids": []},
          "confidence": {"level": "low"}}


def _decision(kind, output, decision="review"):
    rec = {"kind": kind, "route": "computed", "text": ""}
    return DecisionResponse(kind=kind, output=output, decision=decision, trust_score=0.55,
                            components={"E": 1.0, "R": 0.5, "C": 0.6, "S": 0.0},
                            weights={"E": 0.35, "R": 0.15, "C": 0.30, "S": 0.20}, constraints=[],
                            reasons=["S = 0.000 below floor 0.5"], recommendation=rec)


class Recorder:
    def __init__(self):
        self.calls = []


class FakeDecisions(Recorder):
    def decide(self, req):
        self.calls.append(req)
        if req.kind == "forecast":
            return _decision("forecast", FORECAST)
        if req.kind == "diagnosis":
            if req.diagnose.event_id == 999:
                raise UnknownEventError("unknown")
            return _decision("diagnosis", REPORT)
        return _decision("answer", {"answer": "Check the batteries. [x::0]", "route": "generated"}, "accept")


class FakeForecast(Recorder):
    def forecast(self, asset, origin, confidence):
        self.calls.append((asset, origin, confidence))
        return dict(FORECAST)


class FakeDiagnosis(Recorder):
    def diagnose(self, req):
        self.calls.append(req)
        return SimpleNamespace(model_dump=lambda mode=None: dict(REPORT))


def make():
    calls = {"none": [], "basic": [], "hybrid": []}

    def qa(name):
        def answer(q):
            calls[name].append(q)
            return SimpleNamespace(text=f"{name} answer", system=name, route="generated", context=[])
        return answer

    orch = Orchestrator(qa={k: qa(k) for k in calls}, forecast_service=FakeForecast(),
                        diagnosis_service=FakeDiagnosis(), decision_service=FakeDecisions())
    return orch, calls


@pytest.fixture
def client():
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_intent_routing_rules():
    r = lambda **kw: route_intent(AssistantRequest(**{"message": "", **kw}))[0]  # noqa: E731
    assert r(message="What does the pitch battery fault mean?") == "question"
    assert r(message="Forecast solar plant 1 for the next hour") == "forecast"
    assert r(message="x", origin="2020-06-17T11:00") == "forecast"
    assert r(message="Diagnose fault event 73") == "diagnosis"
    assert r(message="", event_id=73) == "diagnosis"
    assert r(message="How many events are in Wind Farm A?") == "question"
    assert r(message="Forecast", intent="question") == "question"
    assert route_intent(AssistantRequest(message="please diagnose event 12"))[1] == 12


def test_ablation_table_adds_one_capability_per_config():
    assert ABLATION["A"].retrieval is None and ABLATION["B"].retrieval == "basic"
    assert ABLATION["C"].retrieval == "hybrid" and not ABLATION["C"].forecasting
    assert ABLATION["D"].forecasting and not ABLATION["D"].diagnosis
    assert ABLATION["E"].diagnosis and not ABLATION["E"].trust
    f = ABLATION["F"]
    assert f.retrieval == "secure" and f.forecasting and f.diagnosis and f.security and f.trust
    assert [c for c, fl in ABLATION.items() if fl.trust] == ["F"]


@pytest.mark.parametrize("config, path", [("A", "none"), ("B", "basic"), ("C", "hybrid"), ("D", "hybrid"),
                                          ("E", "hybrid")])
def test_questions_use_the_config_question_path_without_trust(config, path):
    orch, calls = make()
    out = orch.handle(AssistantRequest(message="What is a pitch fault?", config=config))
    assert calls[path] == ["What is a pitch fault?"] and out.trust is None
    assert not orch.decision_service.calls


def test_full_config_returns_trust_and_decision():
    orch, calls = make()
    out = orch.handle(AssistantRequest(message="What is a pitch fault?"))
    assert out.config == "F" and out.handled_by == "secure_pipeline"
    assert out.trust.decision == "accept" and out.trust.trust_score == 0.55
    assert not any(calls.values())                                   # no unsecured path was used


def test_forecast_interval_with_and_without_trust():
    orch, _ = make()
    f = orch.handle(AssistantRequest(message="", origin="2020-06-17T11:00", config="F"))
    assert f.forecast == {k: FORECAST[k] for k in ("prediction", "lower_bound", "upper_bound", "confidence")}
    assert f.trust.decision == "review" and orch.decision_service.calls[0].kind == "forecast"
    d = orch.handle(AssistantRequest(message="", origin="2020-06-17T11:00", asset="wind_t1", config="D"))
    assert d.trust is None and d.forecast["lower_bound"] == 9000.0
    assert orch.forecast_service.calls[0][0] == "wind_t1"


def test_disabled_module_falls_back_to_the_question_path():
    orch, calls = make()
    out = orch.handle(AssistantRequest(message="", origin="2020-06-17T11:00", config="C"))
    assert out.intent == "forecast" and out.handled_by == "hybrid_rag" and out.forecast is None
    assert "solar_plant1" in calls["hybrid"][0]
    out = orch.handle(AssistantRequest(message="Diagnose fault event 73", config="D"))
    assert out.intent == "diagnosis" and out.handled_by == "hybrid_rag"
    assert not orch.diagnosis_service.calls


def test_diagnosis_in_e_and_f():
    orch, _ = make()
    e = orch.handle(AssistantRequest(message="Diagnose fault event 73", config="E"))
    assert e.handled_by == "fault_diagnosis" and e.trust is None and orch.diagnosis_service.calls[0].event_id == 73
    f = orch.handle(AssistantRequest(message="", event_id=73))
    assert f.trust.decision == "review" and "hydraulic" in f.text


def test_assistant_endpoint_and_errors(client):
    orch, _ = make()
    main.app.dependency_overrides[main.get_orchestrator] = lambda: orch
    ok = client.post("/assistant", json={"message": "", "origin": "2020-06-17T11:00"})
    assert ok.status_code == 200 and ok.json()["trust"]["decision"] == "review"
    assert ok.json()["forecast"]["upper_bound"] == 15000.0
    assert client.post("/assistant", json={"message": "Forecast plant 1"}).status_code == 422   # no origin
    assert client.post("/assistant", json={"message": "", "event_id": 999}).status_code == 404
    assert client.post("/assistant", json={"message": ""}).status_code == 422
    assert client.post("/assistant", json={"message": "q", "config": "G"}).status_code == 422
    bad_conf = {"message": "", "origin": "2020-06-17T11:00", "confidence": 0.5}
    assert client.post("/assistant", json=bad_conf).status_code == 422
