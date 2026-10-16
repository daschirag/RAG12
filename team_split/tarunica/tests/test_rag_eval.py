"""B6: claim-level evaluation, numeric checks, judge parsing/calibration, systems (fake LLMs, no Ollama)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.backend.evaluation.rag_eval import (
    calibrate_judge,
    calibration_set,
    extract_count,
    judge_claim,
    numbers_in,
    parse_score,
    parse_verdict,
    split_claims,
    strip_citations,
    unsupported_numbers,
)
from src.backend.evaluation.rag_systems import count_answer, greenrag_answer
from src.backend.kb.build import build_documents


@pytest.fixture(scope="module")
def docs():
    return build_documents()


# ------------------------------------------------------------ claims/numbers

def test_split_claims_removes_citations_and_short_fragments():
    text = ("The hydraulic group failed [fault-a-73::0]. It may be due to environmental factors or operational "
            "issues. Yes.\n- Check the pump running time and accumulator pre-charge.")
    assert split_claims(text) == ["The hydraulic group failed .", "It may be due to environmental factors or "
                                  "operational issues.", "Check the pump running time and accumulator pre-charge."]
    assert strip_citations("x [a::0, b::1] y") == "x y"


def test_numbers_and_unsupported_numbers():
    assert numbers_in("asset 21, 4.9 days, 2023-06-10") == ["21", "4.9", "2023", "06", "10"]
    ref = "Wind Farm A turbine 21: window 4.90 days starting 2023-06-10."
    assert unsupported_numbers("Turbine 21 was down for 4.9 days [fault-a-22::0].", ref) == []
    assert unsupported_numbers("Turbine 21 was down for 12 days.", ref) == ["12"]


def test_extract_count():
    assert extract_count("There were 3 failures [history-x::0].") == 3
    assert extract_count("Two gearbox failures were recorded.") == 2
    assert extract_count("I do not know.") is None


# --------------------------------------------------------------------- judge

def test_parse_judge_outputs():
    assert parse_verdict('{"verdict": "supported"}') is True
    assert parse_verdict('{"verdict": "not_supported"}') is False
    assert parse_verdict("NOT SUPPORTED") is False and parse_verdict("garbage") is None
    assert parse_score('{"score": 5}') == 1.0 and parse_score('{"score": 1}') == 0.0
    assert parse_score("I'd say 4") == 0.75 and parse_score("none") is None


def keyword_judge(prompt: str) -> str:
    """Fake judge: supported iff every content word of the claim appears in the context."""
    context = prompt.split('"""')[1].lower()
    claim = prompt.split('Claim: "')[1].split('"\n')[0].lower()
    words = [w for w in claim.replace(".", " ").split() if len(w) > 3]
    return json.dumps({"verdict": "supported" if all(w in context for w in words) else "not_supported"})


def test_valid_citations_do_not_hide_unsupported_prose():
    """The B5 failure: citations are valid, but one sentence adds facts no evidence contains."""
    context = "Logged root-cause description: Hydraulic group. Check pump running time and accumulator pre-charge."
    answer = ("The logged cause is a hydraulic group fault [fault-a-73::0]. "
              "It may be due to environmental factors or operational issues [fault-a-73::0].")
    claims = split_claims(answer)
    assert len(claims) == 2                                    # the citations do not merge or hide sentences
    assert judge_claim(keyword_judge, claims[1], context) is False   # the invented sentence is caught


def test_v2_prompt_includes_question_and_retrieved_documents():
    from src.backend.evaluation.rag_eval import judge_claim_v2

    seen = []

    def capture(prompt):
        seen.append(prompt)
        return '{"verdict": "supported"}'

    assert judge_claim_v2(capture, "No troubleshooting procedure was retrieved.", "text", "Q about turbine 0?",
                          "fault-a-73: title") is True
    p = seen[0]
    assert "Q about turbine 0?" in p and "fault-a-73: title" in p and "text" in p
    assert "Added facts or speculation are NOT supported" in p          # content standard unchanged


def test_process_calibration_items_have_true_and_false_versions():
    from src.backend.evaluation.rag_eval import process_calibration_items

    records = [{"question": "Wind Farm A turbine 0 shows an anomaly starting 2023-06-10.",
                "context": [{"doc_id": "fault-a-73", "text": "t"}, {"doc_id": "trouble-wind-gearbox", "text": "u"}]},
               {"question": "Wind Farm A turbine 10 shows an anomaly.",
                "context": [{"doc_id": "fault-a-10", "text": "v"}]}]
    items = process_calibration_items(records)
    by = {(i["claim"], i["question"][:24]): i["supported"] for i in items}
    assert by[("The record fault-a-73 was retrieved.", records[0]["question"][:24])] is True
    assert by[("The record fault-a-10 was retrieved.", records[0]["question"][:24])] is False
    assert by[("No troubleshooting procedure was retrieved.", records[0]["question"][:24])] is False
    assert by[("No troubleshooting procedure was retrieved.", records[1]["question"][:24])] is True
    assert by[("There is an anomaly signal for turbine 100.", records[0]["question"][:24])] is False
    assert {i["kind"] for i in items} == {"process_true", "process_false", "input_true", "input_false"}


def test_rag_answer_expansion_keeps_exclusions():
    from src.backend.evaluation.rag_systems import rag_answer
    from src.backend.retrieval.hybrid_retriever import SiblingExpander

    chunks = [{"chunk_id": f"{d}::{i}", "text": f"{d}{i}", "metadata": {"doc_id": d, "chunk_index": i, "event_id": e}}
              for d, e in (("fault-a-73", 73), ("fault-a-26", 26)) for i in range(2)]
    hits = [chunks[0], chunks[2]]
    ans = rag_answer("hybrid_rag", lambda p: "answer", FakeRetriever(hits), "q",
                     exclude=lambda h: h["metadata"]["event_id"] == 73, expander=SiblingExpander(chunks))
    assert [c["chunk_id"] for c in ans.context] == ["fault-a-26::0", "fault-a-26::1"]


def test_template_claims_are_verified_deterministically():
    from src.backend.evaluation.rag_eval import verify_template_claim

    comps = {"fault-a-73": "hydraulic_system", "trouble-wind-gearbox": "gearbox"}
    labels = {"hydraulic_system": "Hydraulic system", "gearbox": "Gearbox and gear-oil system"}
    q = "Wind Farm A turbine 0 shows an anomaly starting 2023-06-10. What is the likely cause?"
    header = [{"doc_id": "fault-a-73", "text": "Fault event A-73 header"}]
    with_cause = header + [{"doc_id": "fault-a-73", "text": 'original wording): "Hydraulic group"'}]
    v = lambda claim, ctx: verify_template_claim(claim, q, ctx, comps, labels)
    assert v("Anomaly signal for Wind Farm A turbine 0; evidence points to the hydraulic system.", header) is True
    assert v("Anomaly signal for Wind Farm A turbine 10; evidence points to the hydraulic system.", header) is False
    assert v("Anomaly signal for Wind Farm A turbine 0; evidence points to the gearbox and gear-oil system.",
             header) is False
    assert v('Likely cause: Logged description for this incident: "Hydraulic group"', with_cause) is True
    assert v('Likely cause: Logged description for this incident: "Pitch failure"', with_cause) is False
    rec = ("Likely cause: The record of this incident (fault-a-73) was retrieved, but the retrieved text does not "
           "state the logged cause.")
    assert v(rec, header) is True and v(rec, with_cause) is False
    assert v("Recommended action: Inspect the turbine; no troubleshooting procedure was retrieved.", header) is True
    assert v("Recommended action: Inspect the turbine; no troubleshooting procedure was retrieved.",
             header + [{"doc_id": "trouble-wind-gearbox", "text": "t"}]) is False
    assert v("Recommended action: Follow the checks in trouble-wind-gearbox (synthetic procedure).", header) is False
    assert v("The gearbox bearings are damaged.", header) is None          # not a template: left to the judge


def test_calibration_set_has_known_labels(docs):
    items = calibration_set(docs, n_each=10, seed=0)
    kinds = {i.kind for i in items}
    assert kinds == {"supported", "other_component", "changed_number"}
    for i in items:
        assert (i.claim in i.context) == (i.kind == "supported")      # supported claims are copied from the context
        assert i.supported == (i.kind == "supported")
    assert calibration_set(docs, n_each=10, seed=0) == items          # deterministic


def test_calibrate_judge_metrics(docs):
    items = calibration_set(docs, n_each=10, seed=0)
    oracle = lambda p: json.dumps({"verdict": "supported" if p.split('Claim: "')[1].split('"\n')[0] in
                                   p.split('"""')[1] else "not_supported"})
    res = calibrate_judge(oracle, items)
    assert res["accuracy"] == 1.0 and res["balanced_accuracy"] == 1.0
    always_yes = calibrate_judge(lambda p: '{"verdict": "supported"}', items)
    assert always_yes["true_positive_rate"] == 1.0 and always_yes["true_negative_rate"] == 0.0
    assert always_yes["balanced_accuracy"] == 0.5


# --------------------------------------------------------------------- systems

def test_count_answer_matches_taxonomy():
    text = count_answer("How many hydraulic system failures were recorded in Wind Farm A?")
    assert text.startswith("6 anomaly event(s)")
    assert count_answer("How many widgets broke?") is None


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, q, k=5):
        return self.hits[:k]


HITS = [{"chunk_id": "history-gearbox-a::0", "text": "Historical gearbox events in Wind Farm A: 3 anomaly event(s).",
         "metadata": {"doc_id": "history-gearbox-a"}},
        {"chunk_id": "trouble-wind-gearbox::0", "text": "Check oil level and filters.",
         "metadata": {"doc_id": "trouble-wind-gearbox"}}]


def test_greenrag_routes_count_questions_to_the_computed_answer():
    a = greenrag_answer(lambda p: "unused", FakeRetriever(HITS),
                        "How many gearbox and gear-oil system failures were recorded in Wind Farm A?")
    assert a.route == "numeric" and a.text.startswith("3 anomaly event(s)")


def test_greenrag_citation_validation_and_fallback():
    good = lambda p: json.dumps({"answer": "Check the oil level and filters.", "chunk_ids": ["trouble-wind-gearbox"]})
    a = greenrag_answer(good, FakeRetriever(HITS), "What should be checked on a gearbox?")
    assert a.route == "generated" and a.text.endswith("[trouble-wind-gearbox::0]")
    uncited = lambda p: json.dumps({"answer": "Replace the gearbox.", "chunk_ids": []})
    b = greenrag_answer(uncited, FakeRetriever(HITS), "What should be checked on a gearbox?")
    assert b.route == "extractive_fallback" and "[history-gearbox-a::0]" in b.text
    broken = greenrag_answer(lambda p: "not json", FakeRetriever(HITS), "What should be checked on a gearbox?")
    assert broken.route == "extractive_fallback" and broken.notes


def test_per_answer_metrics(docs):
    from experiments.rag_evaluation import per_answer_table

    items = [{"item_id": "kb::q1", "dataset": "kb", "question": "q", "reference": "ref", "relevant_docs": ["d1"],
              "true_component": None, "count_truth": 3, "event_id": None, "mode": None}]
    answers = [{"item_id": "kb::q1", "system": "greenrag", "text": "There were 3 events.", "route": "generated",
                "context": [{"chunk_id": "d1::0", "doc_id": "d1", "text": "t"},
                            {"chunk_id": "d2::0", "doc_id": "d2", "text": "t"}], "seconds": 1.0}]
    judged = [{"item_id": "kb::q1", "system": "greenrag", "answer_relevance": 0.75, "unsupported_numbers": [],
               "claims": [{"claim": "a", "supported_by_context": True, "supported_by_reference": True},
                          {"claim": "b", "supported_by_context": False, "supported_by_reference": False}]}]
    row = per_answer_table(items, answers, judged, docs).iloc[0]
    assert row.faithfulness == 0.5 and row.claim_unsupported_rate == 0.5 and row.hallucinated == 1.0
    assert row.context_relevance == 0.5 and row.count_exact_match == 1.0 and np.isclose(row.answer_relevance, 0.75)
