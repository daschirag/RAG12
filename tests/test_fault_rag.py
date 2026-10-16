"""B5: fault signal -> retrieval -> LLM -> validated DiagnosisReport (fake retriever/LLM; one real-Ollama test)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from src.backend.faultdiag.diagnosis import (
    Confidence,
    CitedText,
    Diagnoser,
    DiagnosisReport,
    EvidenceItem,
    FaultSignal,
    Reliability,
    build_prompt,
    load_reliability,
)

WEAK = Reliability(detector="logistic_regression", value=0.742, ci95_low=0.5, ci95_high=0.95, n_events=22,
                   protocol="leave-one-asset-out")
STRONG_DETECTOR = WEAK.model_copy(update={"ci95_low": 0.7})
T = pd.Timestamp
WINDOWS = {("A", 73): (T("2023-06-10 11:40"), T("2023-06-17 11:40")),
           ("A", 26): (T("2023-10-12 10:20"), T("2023-10-19 10:20"))}
COMPONENTS = {"fault-a-73": "hydraulic_system", "maint-a-73": "hydraulic_system",
              "trouble-wind-hydraulic_system": "hydraulic_system", "fault-a-26": "hydraulic_system"}
SIGNAL = FaultSignal("A", 0, T("2023-06-10 11:40"), T("2023-06-17 11:40"), anomaly_score=0.64, event_id=73)


def chunk(doc_id, doc_type, score, synthetic=False, farm="A", asset=0, event=None, text="text"):
    meta = {"doc_id": doc_id, "doc_type": doc_type, "title": doc_id, "synthetic": synthetic}
    if farm:
        meta.update(farm=farm, asset_id=asset)
    if event is not None:
        meta["event_id"] = event
    return {"chunk_id": f"{doc_id}::0", "text": text, "metadata": meta, "score": score}


INCIDENT = chunk("fault-a-73", "fault_description", 5.0, event=73,
                 text='Fault event A-73 ... Logged root-cause description (original wording): "Hydraulic group"')
MAINT = chunk("maint-a-73", "maintenance_record", 3.0, synthetic=True, event=73)
TROUBLE = chunk("trouble-wind-hydraulic_system", "troubleshooting_procedure", 2.0, synthetic=True, farm=None)
OTHER_EVENT = chunk("fault-a-26", "fault_description", 1.5, event=26)       # same turbine, 4 months later


class FakeRetriever:
    def __init__(self, hits):
        self.hits, self.queries = hits, []

    def search(self, query, k=5):
        self.queries.append(query)
        return self.hits[:k]


def llm_returning(payload):
    return lambda prompt: json.dumps(payload)


GOOD_LLM = llm_returning({"diagnosis": "Hydraulic group fault on turbine 0.",
                          "likely_cause": "The logbook records a hydraulic group fault.",
                          "likely_cause_chunk_ids": ["fault-a-73::0"],
                          "recommended_action": "Check pump running time and accumulator pre-charge.",
                          "action_chunk_ids": ["trouble-wind-hydraulic_system::0"]})


def diagnoser(hits, llm=GOOD_LLM, reliability=WEAK):
    return Diagnoser(FakeRetriever(hits), llm, reliability, WINDOWS)


# ----------------------------------------------------------------- confidence

def test_incident_evidence_gives_moderate_but_never_high_with_weak_detector():
    r = diagnoser([INCIDENT, MAINT, TROUBLE]).diagnose(SIGNAL, COMPONENTS)
    assert r.confidence.evidence_support == "strong"
    assert r.confidence.level == "moderate"                   # CI lower bound 0.5 -> never "high"
    assert "not distinguishable from chance" in r.confidence.explanation
    assert "0.742" in r.confidence.explanation and "0.500" in r.confidence.explanation


def test_weak_evidence_is_flagged_low_confidence():
    r = diagnoser([OTHER_EVENT, TROUBLE]).diagnose(SIGNAL, COMPONENTS)     # nothing about this incident
    assert r.confidence.evidence_support == "weak" and r.confidence.level == "low"
    assert r.severity == "undetermined" and "would be medium" in r.severity_basis   # no asserted severity
    r2 = diagnoser([OTHER_EVENT, TROUBLE], reliability=STRONG_DETECTOR).diagnose(SIGNAL, COMPONENTS)
    assert r2.confidence.level == "low"                      # even a good detector cannot lift weak evidence


def test_low_rerank_score_is_not_strong_evidence():
    weak_match = {**INCIDENT, "score": -2.0}
    r = diagnoser([weak_match, TROUBLE]).diagnose(SIGNAL, COMPONENTS)
    assert r.confidence.evidence_support == "weak" and r.confidence.level == "low"


def test_high_requires_reliable_detector_and_score():
    assert diagnoser([INCIDENT, TROUBLE], reliability=STRONG_DETECTOR).diagnose(
        SIGNAL, COMPONENTS).confidence.level == "high"
    low_score = FaultSignal("A", 0, SIGNAL.window_start, SIGNAL.window_end, anomaly_score=0.2, event_id=73)
    assert diagnoser([INCIDENT, TROUBLE], reliability=STRONG_DETECTOR).diagnose(
        low_score, COMPONENTS).confidence.level == "moderate"


def test_schema_rejects_confident_report_on_weak_evidence():
    r = diagnoser([OTHER_EVENT, TROUBLE]).diagnose(SIGNAL, COMPONENTS)
    data = r.model_dump()
    data["confidence"]["level"] = "moderate"
    with pytest.raises(ValidationError, match="low confidence"):
        DiagnosisReport.model_validate(data)


# ------------------------------------------------------------- evidence & LLM

def test_llm_report_cites_real_chunks_and_severity_is_deterministic():
    r = diagnoser([INCIDENT, MAINT, TROUBLE]).diagnose(SIGNAL, COMPONENTS)
    assert r.generation == "llm"
    assert r.likely_cause.chunk_ids == ["fault-a-73::0"] and r.likely_cause.source == "retrieved_evidence"
    assert r.likely_cause_component == "hydraulic_system"
    assert r.severity == "medium" and "hydraulic_system" in r.severity_basis     # taxonomy default
    assert {e.chunk_id for e in r.evidence} == {"fault-a-73::0", "maint-a-73::0", "trouble-wind-hydraulic_system::0"}


@pytest.mark.parametrize("bad", [
    "not json",
    json.dumps({"diagnosis": "x", "likely_cause": "y"}),                                        # missing keys
    json.dumps({"diagnosis": "Fault.", "likely_cause": "Invented cause.", "likely_cause_chunk_ids": ["made-up::0"],
                "recommended_action": "Do something.", "action_chunk_ids": []}),               # unknown chunk id
    json.dumps({"diagnosis": "Fault.", "likely_cause": "Uncited cause.", "likely_cause_chunk_ids": [],
                "recommended_action": "Do something.", "action_chunk_ids": []}),               # uncited cause
])
def test_invalid_llm_output_falls_back_to_evidence_only(bad):
    r = diagnoser([INCIDENT, MAINT, TROUBLE], llm=lambda p: bad).diagnose(SIGNAL, COMPONENTS)
    assert r.generation == "fallback" and any("rejected" in n for n in r.notes)
    assert r.likely_cause.chunk_ids == ["fault-a-73::0"]
    assert '"Hydraulic group"' in r.likely_cause.text                                   # logged description quoted


HEADER = {"chunk_id": "fault-a-73::0", "text": "Fault event A-73: Wind Farm A, turbine asset 0. Event window: ...",
          "score": 5.0, "metadata": {"doc_id": "fault-a-73", "doc_type": "fault_description", "title": "t",
                                     "synthetic": False, "farm": "A", "asset_id": 0, "event_id": 73,
                                     "chunk_index": 0}}
CAUSE = {"chunk_id": "fault-a-73::1", "text": 'Logged root-cause description (original wording): "Hydraulic group"',
         "score": 0.0, "metadata": {**HEADER["metadata"], "chunk_index": 1}}


def test_fallback_finds_the_cause_in_a_sibling_chunk():
    """B6 bug: the retrieved chunk was the event header; the cause was in the record's other chunk."""
    from src.backend.retrieval.hybrid_retriever import SiblingExpander

    d = Diagnoser(FakeRetriever([HEADER]), None, WEAK, WINDOWS, expander=SiblingExpander([HEADER, CAUSE]))
    r = d.diagnose(SIGNAL, COMPONENTS)
    assert [e.chunk_id for e in r.evidence] == ["fault-a-73::0", "fault-a-73::1"]
    assert r.evidence[1].expanded_from == "fault-a-73::0"
    assert '"Hydraulic group"' in r.likely_cause.text and r.likely_cause.chunk_ids == ["fault-a-73::1"]


def test_fallback_never_says_not_established_when_the_incident_record_is_retrieved():
    r = diagnoser([HEADER], llm=None).diagnose(SIGNAL, COMPONENTS)          # no expansion, cause chunk missing
    assert r.confidence.evidence_support == "strong"
    assert "Not established" not in r.likely_cause.text
    assert "was retrieved, but the retrieved text does not state" in r.likely_cause.text


def test_expansion_respects_unlogged_exclusion():
    from src.backend.retrieval.hybrid_retriever import SiblingExpander

    d = Diagnoser(FakeRetriever([HEADER, TROUBLE]), None, WEAK, WINDOWS, expander=SiblingExpander([HEADER, CAUSE]))
    r = d.diagnose(SIGNAL, COMPONENTS, exclude_own_event=True)
    assert all(e.event_id != 73 for e in r.evidence)


def test_bare_document_ids_resolve_only_to_retrieved_evidence():
    llm = llm_returning({"diagnosis": "Hydraulic group fault.", "likely_cause": "Logged hydraulic group fault.",
                         "likely_cause_chunk_ids": ["fault-a-73"], "recommended_action": "Check the pump.",
                         "action_chunk_ids": ["trouble-wind-hydraulic_system"]})
    r = diagnoser([INCIDENT, MAINT, TROUBLE], llm=llm).diagnose(SIGNAL, COMPONENTS)
    assert r.generation == "llm"
    assert r.likely_cause.chunk_ids == ["fault-a-73::0"]
    assert r.recommended_action.chunk_ids == ["trouble-wind-hydraulic_system::0"]
    # A bare id of a document that was NOT retrieved is still rejected.
    bad = llm_returning({"diagnosis": "Fault.", "likely_cause": "Cause.", "likely_cause_chunk_ids": ["fault-a-26"],
                         "recommended_action": "Act.", "action_chunk_ids": []})
    assert diagnoser([INCIDENT, TROUBLE], llm=bad).diagnose(SIGNAL, COMPONENTS).generation == "fallback"


def test_llm_backend_error_falls_back():
    def broken(prompt):
        raise ConnectionError("ollama down")

    r = diagnoser([INCIDENT, TROUBLE], llm=broken).diagnose(SIGNAL, COMPONENTS)
    assert r.generation == "fallback" and any("unavailable" in n for n in r.notes)


def test_unlogged_mode_excludes_the_events_own_documents():
    r = diagnoser([INCIDENT, MAINT, OTHER_EVENT, TROUBLE], llm=None).diagnose(SIGNAL, COMPONENTS,
                                                                              exclude_own_event=True)
    assert all(e.event_id != 73 for e in r.evidence)
    assert r.confidence.evidence_support == "weak" and r.confidence.level == "low"
    assert r.likely_cause.text.startswith("Not established")


def test_no_evidence_report():
    r = diagnoser([], llm=GOOD_LLM).diagnose(SIGNAL, COMPONENTS)
    assert r.confidence.evidence_support == "none" and r.confidence.level == "low"
    assert r.likely_cause.chunk_ids == [] and r.severity == "undetermined"


def test_schema_rejects_citations_outside_evidence():
    ev = [EvidenceItem(chunk_id="a::0", doc_id="a", doc_type="fault_description", title="t", synthetic=False,
                       rerank_score=1.0, excerpt="x")]
    conf = Confidence(level="low", anomaly_score=0.5, reliability=WEAK, evidence_support="weak", explanation="e")
    with pytest.raises(ValidationError, match="not in the evidence"):
        DiagnosisReport(farm="A", asset_id=0, diagnosis="Hydraulic fault.", confidence=conf,
                        likely_cause=CitedText(text="cause", chunk_ids=["b::0"]), evidence=ev,
                        recommended_action=CitedText(text="act"), severity="low", severity_basis="b",
                        generation="fallback")


def test_prompt_marks_synthetic_and_restricts_to_evidence():
    ev = diagnoser([INCIDENT, MAINT]).retrieve(SIGNAL)
    p = build_prompt(SIGNAL, ev)
    assert "Use ONLY the evidence" in p and "SYNTHETIC" in p and "[fault-a-73::0]" in p


def test_reliability_is_read_from_b4_results():
    r = load_reliability()
    assert r.detector == "logistic_regression" and r.n_events == 22
    assert r.value == pytest.approx(0.742, abs=1e-3) and r.ci95_low == pytest.approx(0.5, abs=1e-3)


def test_ollama_llm_retries_once_after_timeout_then_raises(monkeypatch):
    import httpx
    import ollama

    from src.backend.faultdiag.diagnosis import ollama_llm

    class FakeClient:
        outcomes: list = []

        def __init__(self, timeout):
            assert timeout == 120.0

        def chat(self, **kwargs):
            outcome = FakeClient.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return {"message": {"content": outcome}}

    monkeypatch.setattr(ollama, "Client", FakeClient)
    FakeClient.outcomes = [httpx.ReadTimeout("stalled"), "ok"]
    assert ollama_llm("m")("prompt") == "ok"
    FakeClient.outcomes = [httpx.ReadTimeout("stalled"), httpx.ReadTimeout("stalled again")]
    with pytest.raises(httpx.TimeoutException):
        ollama_llm("m")("prompt")


@pytest.mark.ollama
def test_real_llm_produces_a_valid_report():
    import requests

    from src.backend.faultdiag.diagnosis import ollama_json_llm

    try:
        requests.get("http://127.0.0.1:11434/api/tags", timeout=3).raise_for_status()
    except requests.RequestException:
        pytest.skip("Ollama is not reachable")
    r = diagnoser([INCIDENT, MAINT, TROUBLE], llm=ollama_json_llm()).diagnose(SIGNAL, COMPONENTS)
    assert isinstance(r, DiagnosisReport)
    assert set(r.likely_cause.chunk_ids) <= {e.chunk_id for e in r.evidence}
