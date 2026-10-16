"""B7: POST /diagnose (TestClient) with an injected service; one real-model test without the LLM."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.backend import main
from src.backend.faultdiag.diagnosis import Diagnoser, DiagnosisReport
from src.backend.faultdiag.service import DiagnoseRequest, DiagnosisService
from tests.test_fault_rag import (
    COMPONENTS,
    GOOD_LLM,
    INCIDENT,
    MAINT,
    OTHER_EVENT,
    SIGNAL,
    TROUBLE,
    WEAK,
    WINDOWS,
    FakeRetriever,
)


def fake_service(hits, llm=GOOD_LLM) -> DiagnosisService:
    retriever = FakeRetriever(hits)
    return DiagnosisService(diagnoser_llm=Diagnoser(retriever, llm, WEAK, WINDOWS),
                            diagnoser_fallback=Diagnoser(retriever, None, WEAK, WINDOWS),
                            signals={73: SIGNAL}, components=COMPONENTS)


@pytest.fixture
def client_factory():
    def make(service):
        main.app.dependency_overrides[main.get_diagnosis_service] = lambda: service
        return TestClient(main.app)

    yield make
    main.app.dependency_overrides.pop(main.get_diagnosis_service, None)


def test_request_validation():
    assert DiagnoseRequest(event_id=73).mode == "logged"
    DiagnoseRequest(asset_id=0, window_start="2023-06-10T11:40", window_end="2023-06-17T11:40")
    for bad in ({}, {"event_id": 73, "asset_id": 0}, {"asset_id": 0, "window_start": "2023-06-10T11:40"},
                {"asset_id": 0, "window_start": "2023-06-17T00:00", "window_end": "2023-06-10T00:00"},
                {"asset_id": 0, "window_start": "2023-06-10", "window_end": "2023-06-17", "mode": "unlogged"},
                {"farm": "B", "event_id": 73}):
        with pytest.raises(ValidationError):
            DiagnoseRequest(**bad)


def test_diagnose_event_returns_validated_report(client_factory):
    c = client_factory(fake_service([INCIDENT, MAINT, TROUBLE]))
    resp = c.post("/diagnose", json={"event_id": 73})
    assert resp.status_code == 200
    report = DiagnosisReport.model_validate(resp.json())                 # round-trips through the schema
    assert report.event_id == 73 and report.generation == "llm"
    assert report.confidence.level == "moderate" and report.confidence.reliability.value == pytest.approx(0.742)
    assert "not distinguishable from chance" in report.confidence.explanation
    assert set(report.likely_cause.chunk_ids) <= {e.chunk_id for e in report.evidence}
    assert report.severity == "medium"


def test_weak_evidence_is_low_confidence_and_undetermined_severity(client_factory):
    c = client_factory(fake_service([OTHER_EVENT, TROUBLE]))
    body = c.post("/diagnose", json={"event_id": 73}).json()
    assert body["confidence"]["level"] == "low" and body["confidence"]["evidence_support"] == "weak"
    assert body["severity"] == "undetermined"


def test_unlogged_mode_hides_own_documents(client_factory):
    c = client_factory(fake_service([INCIDENT, MAINT, OTHER_EVENT, TROUBLE]))
    body = c.post("/diagnose", json={"event_id": 73, "mode": "unlogged"}).json()
    assert all(e["event_id"] != 73 for e in body["evidence"])
    assert body["confidence"]["level"] == "low"


def test_window_request_without_detector_score(client_factory):
    c = client_factory(fake_service([INCIDENT, TROUBLE]))
    body = c.post("/diagnose", json={"asset_id": 0, "window_start": "2023-06-10T11:40",
                                     "window_end": "2023-06-17T11:40", "use_llm": False}).json()
    assert body["confidence"]["anomaly_score"] is None and "n/a" in body["confidence"]["explanation"]
    assert body["generation"] == "fallback" and body["event_id"] is None


def test_llm_failure_still_returns_a_report(client_factory):
    def broken(prompt):
        raise TimeoutError("ollama stalled")

    c = client_factory(fake_service([INCIDENT, TROUBLE], llm=broken))
    resp = c.post("/diagnose", json={"event_id": 73})
    assert resp.status_code == 200 and resp.json()["generation"] == "fallback"


def test_errors(client_factory, tmp_path):
    c = client_factory(fake_service([INCIDENT]))
    assert c.post("/diagnose", json={"event_id": 999}).status_code == 404
    assert c.post("/diagnose", json={"farm": "C", "event_id": 73}).status_code == 422
    assert c.post("/diagnose", json={}).status_code == 422
    c2 = client_factory(DiagnosisService(b4_dir=tmp_path))                  # no B4 results
    resp = c2.post("/diagnose", json={"event_id": 73})
    assert resp.status_code == 503 and "fault_diagnosis" in resp.json()["detail"]


def test_openapi_describes_confidence_and_cause(client_factory):
    spec = client_factory(fake_service([INCIDENT])).get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    assert "NOT a probability" in str(schemas["Confidence"])
    assert "retrieved evidence" in str(schemas["DiagnosisReport"])


def test_existing_endpoints_unaffected(client_factory):
    c = client_factory(fake_service([INCIDENT]))
    assert c.get("/health").json() == {"status": "ok"}
    assert "knowledge_base" in c.get("/domains").json()


@pytest.mark.integration
def test_real_service_without_llm():
    """Production retrieval (MiniLM + cross-encoder + expansion) on the real KB; deterministic fallback."""
    service = DiagnosisService()
    report = service.diagnose(DiagnoseRequest(event_id=73, use_llm=False))
    assert report.generation == "fallback"
    assert any(e.doc_id == "fault-a-73" for e in report.evidence)
    assert '"Hydraulic group"' in report.likely_cause.text                  # cause found via sibling expansion
    assert report.confidence.level == "moderate" and report.severity == "medium"
