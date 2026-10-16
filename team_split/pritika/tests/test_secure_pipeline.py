"""C4: secure retrieval pipeline (input security -> injection detection -> retrieval -> trust -> filter -> LLM)."""

from __future__ import annotations

import json
import re

import pytest

from src.backend.kb.build import build_documents
from src.backend.kb.taxonomy import load_all_events
from src.backend.security import attacks as A
from src.backend.security.doc_trust import (
    DocumentTrust,
    RegistryError,
    build_registry,
    calibrate_stuffing,
    logged_descriptions,
    save_registry,
    verify_registry,
)
from src.backend.security.evidence_filter import EvidenceFilter, TrustedRetriever
from src.backend.security.secure_pipeline import (
    MAX_QUERY_CHARS,
    REGISTRY_FILE,
    SecureAskService,
    SecureRAGPipeline,
    check_input,
    trusted_reference,
)

KEY = b"test-key"


@pytest.fixture(scope="module")
def kb():
    docs = build_documents()
    poison, _ = A.poison_documents(docs)
    return docs, poison


def _hit(doc, n=0):
    return {"chunk_id": f"{doc.doc_id}::{n}", "text": doc.text[:400], "metadata": {"doc_id": doc.doc_id,
                                                                                    "chunk_index": n}, "score": 1.0}


class ListRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k=5):
        return self.hits[:k]


class RecordingLLM:
    """Cites the first chunk id in the prompt and records every prompt it sees."""

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        cid = re.search(r"\[([^\]\s]+::\d+)\]", prompt).group(1)
        return json.dumps({"answer": "Grounded answer.", "chunk_ids": [cid]})


def pipeline(store, clean, hits, llm):
    trust = DocumentTrust(verify_registry(build_registry(clean, KEY), KEY), {d.doc_id: d for d in store},
                          logged_descriptions(load_all_events()), calibrate_stuffing(clean))
    return SecureRAGPipeline(TrustedRetriever(ListRetriever(hits), EvidenceFilter(trust)), llm, registry_source="test")


def test_input_checks():
    assert check_input("What should be checked on the gearbox?") == []
    assert check_input("Line one\nline two\ttab") == []
    assert check_input("   ") == ["empty question"]
    assert check_input("x" * (MAX_QUERY_CHARS + 1)) == [f"question longer than {MAX_QUERY_CHARS} characters"]
    assert check_input("gearbox\x00") == ["control characters in the question"]


def test_injected_question_is_blocked_before_retrieval_and_llm(kb):
    docs, _ = kb
    llm = RecordingLLM()
    p = pipeline(docs, docs, [_hit(docs[0])], llm)
    a = p.answer(A.inject_query("What should be checked on the gearbox?"))
    assert a.route == "blocked" and a.security.blocked and "override_instructions" in a.security.injection_signals
    assert llm.prompts == [] and a.security.retrieved_doc_ids == [] and A.CANARY not in a.answer
    bad = p.answer("")
    assert bad.route == "blocked" and not bad.security.input_ok


def test_untrusted_evidence_never_reaches_the_llm(kb):
    docs, poison = kb
    rev2 = next(d for d in poison if d.doc_id == "trouble-wind-gearbox-rev2")
    genuine = next(d for d in docs if d.doc_id == "trouble-wind-gearbox")
    llm = RecordingLLM()
    a = pipeline(docs + poison, docs, [_hit(rev2), _hit(genuine)], llm).answer("What should be checked on the gearbox?")
    assert a.route == "generated" and "[trouble-wind-gearbox::0]" in a.answer
    assert "ignore oil temperature alarms" not in llm.prompts[0].lower()
    assert [r.doc_id for r in a.security.rejected] == [rev2.doc_id]
    assert "registered failed" in a.security.rejected[0].reasons
    assert a.security.passed_chunk_ids == ["trouble-wind-gearbox::0"]
    assert a.security.retrieved_doc_ids == [rev2.doc_id, genuine.doc_id]


def test_fallback_quotes_only_trusted_evidence(kb):
    """The C2 failure: with the LLM unavailable, the fallback quoted the poisoned top chunk verbatim."""
    docs, poison = kb
    rev2 = next(d for d in poison if d.doc_id == "trouble-wind-gearbox-rev2")
    genuine = next(d for d in docs if d.doc_id == "trouble-wind-gearbox")
    a = pipeline(docs + poison, docs, [_hit(rev2), _hit(genuine)], None).answer("What should be checked on the gearbox?")
    assert a.route == "extractive_fallback" and "[trouble-wind-gearbox::0]" in a.answer
    assert "Revision 2" not in a.answer


def test_nothing_trusted_means_no_answer_and_no_quote(kb):
    docs, poison = kb
    llm = RecordingLLM()
    a = pipeline(docs + poison, docs, [_hit(d) for d in poison[:2]], llm).answer("What should be checked on the pitch?")
    assert a.route == "no_trusted_evidence" and llm.prompts == []
    assert "Revision 2" not in a.answer and len(a.security.rejected) == 2


def test_clean_evidence_passes_unchanged(kb):
    docs, _ = kb
    hits = [_hit(d) for d in docs[:3]]
    a = pipeline(docs, docs, hits, RecordingLLM()).answer("Which measurements monitor the transformer?")
    assert a.route == "generated" and a.security.rejected == [] and not a.security.blocked
    assert a.security.passed_chunk_ids == [h["chunk_id"] for h in hits]


def test_trusted_reference_sources(tmp_path, kb):
    docs, _ = kb
    save_registry(build_registry(docs, KEY), tmp_path / REGISTRY_FILE)
    ref, source = trusted_reference(tmp_path, KEY)
    assert source == "signed registry file" and len(ref) == len(docs)
    with pytest.raises(RegistryError):
        trusted_reference(tmp_path, None)                       # signed but no key: fail closed
    with pytest.raises(RegistryError):
        trusted_reference(tmp_path, b"wrong-key")
    save_registry(build_registry(docs, None), tmp_path / REGISTRY_FILE)
    assert trusted_reference(tmp_path, None)[1].startswith("unsigned registry file")
    empty = tmp_path / "empty"
    empty.mkdir()
    ref, source = trusted_reference(empty, None)
    assert source == "in-process build from code and datasets" and ref == build_registry(docs, None)["docs"]


def test_secure_ask_service_uses_the_injected_pipeline(kb):
    docs, _ = kb
    service = SecureAskService(pipeline=pipeline(docs, docs, [_hit(docs[0])], RecordingLLM()))
    assert service.ask("Which measurements monitor the transformer?").route == "generated"


def test_diagnosis_service_is_wired_to_the_secure_retriever(monkeypatch, kb):
    """/diagnose builds its Diagnoser on the C4 secure retriever (checked without loading retrieval models)."""
    from src.backend.faultdiag.service import DiagnosisService
    from src.backend.security import secure_pipeline

    docs, _ = kb
    sentinel = TrustedRetriever(ListRetriever([]), EvidenceFilter(None))
    fake = secure_pipeline.SecureRetrieval(sentinel, [], docs, None, {"registry_source": "fake"})
    monkeypatch.setattr(secure_pipeline, "build_secure_retrieval", lambda: fake)
    service = DiagnosisService()
    service._ensure()
    assert service.diagnoser_llm.retriever is sentinel and service.diagnoser_fallback.retriever is sentinel
    assert service.security_status == {"registry_source": "fake"}


@pytest.mark.integration
def test_real_pipeline_rejects_poisoned_store_documents(kb):
    """Production retrieval over a poisoned store: the unsafe procedure is retrieved but never quoted."""
    from src.backend.security.secure_pipeline import build_secure_retrieval

    docs, poison = kb
    sec = build_secure_retrieval(store_docs=docs + poison, reference=build_registry(docs, None)["docs"])
    assert set(sec.status["untrusted_in_store"]) == {d.doc_id for d in poison}
    a = SecureRAGPipeline(sec.retriever, None).answer(
        "A wind turbine shows oil temperature high at the gearbox bearings. What should the technician check first?")
    assert "trouble-wind-gearbox-rev2" in {r.doc_id for r in a.security.rejected}
    assert "ignore oil temperature alarm" not in a.answer.lower()
