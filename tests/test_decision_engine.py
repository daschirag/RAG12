"""C5: trust-aware decision engine (the LLM recommends; only this pipeline decides)."""

from __future__ import annotations

import pytest

from experiments.decision_engine_eval import weight_grid
from src.backend.kb.build import build_documents
from src.backend.kb.taxonomy import load_all_events
from src.backend.security import attacks as A
from src.backend.security.doc_trust import DocumentTrust, build_registry, calibrate_stuffing, logged_descriptions
from src.backend.trust.decision import (
    DecisionConfig,
    DecisionEngine,
    EvidenceUse,
    Recommendation,
    SecurityFacts,
    TrustWeights,
    check_constraints,
    unsafe_directives,
)
from src.backend.trust.signals import load_signals

GRID = weight_grid()
SIG = load_signals()


def engine(weights=None, trust=None):
    cfg = DecisionConfig() if weights is None else DecisionConfig(weights=TrustWeights(E=weights[0], R=weights[1],
                                                                                        C=weights[2], S=weights[3]))
    return DecisionEngine(cfg, SIG, trust=trust)


def ev(cid="fault-a-73::0", score=8.0, trusted=True):
    return EvidenceUse(chunk_id=cid, doc_id=cid.split("::")[0], rerank_score=score, trusted=trusted)


def best_diagnosis(**update) -> Recommendation:
    """A diagnosis that is as good as it can be on every axis except the detector it rests on."""
    base = dict(kind="diagnosis", route="llm",
                text="Hydraulic group fault. Likely cause: hydraulic group [fault-a-73::0]. Recommended action: "
                     "inspect the hydraulic system accumulator and pump.",
                action_text="Inspect the hydraulic system accumulator and pump.",
                evidence=[ev()], cited_chunk_ids=["fault-a-73::0"], component="hydraulic_system", severity="medium",
                evidence_support="strong", confidence_level="moderate",
                security=SecurityFacts(trust_checked=True, registry_source="signed registry file"))
    return Recommendation(**{**base, **update})


def best_answer(**update) -> Recommendation:
    base = dict(kind="answer", route="generated",
                text="Check the battery voltages and the last battery test result. [trouble-wind-pitch_system::0]",
                action_text="Check the battery voltages and the last battery test result.",
                evidence=[ev("trouble-wind-pitch_system::0")], cited_chunk_ids=["trouble-wind-pitch_system::0"],
                security=SecurityFacts(has_user_question=True, injection_checked=True, trust_checked=True,
                                       registry_source="signed registry file"))
    return Recommendation(**{**base, **update})


# ------------------------------------------------------------------ configuration and inputs

def test_weights_must_be_nonnegative_and_sum_to_one():
    TrustWeights(E=0.25, R=0.25, C=0.25, S=0.25)
    with pytest.raises(ValueError):
        TrustWeights(E=0.5, R=0.5, C=0.5, S=0.0)
    with pytest.raises(ValueError):
        TrustWeights(E=1.1, R=-0.1, C=0.0, S=0.0)
    assert len(GRID) == 969 and all(abs(sum(w) - 1) < 1e-9 and min(w) >= 0.05 - 1e-9 for w in GRID)


def test_signals_are_the_measured_values():
    from src.backend.faultdiag.diagnosis import load_reliability

    rel = load_reliability()
    assert (SIG.detector_auc, SIG.detector_ci_low, SIG.detector_ci_high) == (rel.value, rel.ci95_low, rel.ci95_high)
    assert SIG.detector_ci_low == pytest.approx(0.5)            # B4: the CI reaches chance
    inj = SIG.injection_check
    assert (inj.tpr.n, inj.fpr.n) == (15, 15)
    assert inj.tpr.value == pytest.approx(11 / 15) and inj.fpr.value == pytest.approx(1 / 15)
    assert inj.strength == pytest.approx(1 - (4 / 15) / (14 / 15))           # the FPR lowers the strength
    assert SIG.trust_check_full.fpr.n == 189 and SIG.trust_check_full.strength == pytest.approx(1.0)
    assert SIG.trust_check_content_only.strength < SIG.trust_check_full.strength
    assert 0 < SIG.fallback_relevance_ratio.value < 1


# ------------------------------------------------------------------ required test 2

def test_diagnosis_on_a_detector_not_better_than_chance_is_never_accepted():
    rec = best_diagnosis()
    e = engine()
    comp = e.components(rec)
    assert comp["C"] == 0.0 and comp["E"] == 1.0 and comp["S"] == 1.0 and comp["R"] > 0.99
    for w in GRID:
        d = engine(w).decide(rec)
        assert d.decision != "accept", w
    perfect = {"E": 1.0, "R": 1.0, "C": e.components(rec)["C"], "S": 1.0}
    assert all(engine(w).decide(rec, components=perfect).decision != "accept" for w in GRID)


def test_chance_level_cap_cannot_be_configured_away():
    """No valid configuration (any weights, thresholds or floors) lets a C = 0 diagnosis be accepted."""
    with pytest.raises(ValueError):
        DecisionConfig(floors={"E": 0.0, "R": 0.0, "C": 0.0, "S": 0.5})
    with pytest.raises(ValueError):
        DecisionConfig(accept_at=0.3, review_at=0.4)
    rec = best_diagnosis()
    perfect = {"E": 1.0, "R": 1.0, "C": 0.0, "S": 1.0}
    for w in GRID[::7]:
        for acc in (0.05, 0.5, 0.95):
            cfg = DecisionConfig(weights=TrustWeights(E=w[0], R=w[1], C=w[2], S=w[3]), accept_at=acc, review_at=0.0,
                                 floors={"E": 0.0, "R": 0.0, "C": 1e-9, "S": 1e-9})
            assert DecisionEngine(cfg, SIG).decide(rec, components=perfect).decision != "accept"


def test_a_detector_above_chance_could_be_accepted():
    """Control for the test above: the cap comes from C, not from anything else in the diagnosis."""
    import dataclasses

    better = dataclasses.replace(SIG, detector_ci_low=0.9)
    d = DecisionEngine(DecisionConfig(), better).decide(best_diagnosis())
    assert d.decision == "accept", d.reasons


# ------------------------------------------------------------------ required test 3

def test_blocked_request_is_rejected_whatever_the_llm_recommended():
    rec = best_answer(route="blocked", security=SecurityFacts(blocked=True, has_user_question=True))
    for w in GRID:
        assert engine(w).decide(rec).decision == "reject"
    # even if a model had produced a perfect-looking answer for the blocked request
    faked = best_answer(security=SecurityFacts(blocked=True, has_user_question=True, injection_checked=True,
                                               trust_checked=True, registry_source="signed registry file"))
    assert all(engine(w).decide(faked).decision == "reject" for w in GRID)


def test_evidence_failing_document_trust_is_never_accepted():
    rec = best_answer(evidence=[ev("trouble-wind-pitch_system::0", trusted=False)])
    for w in GRID:
        assert engine(w).decide(rec).decision == "reject"


def test_engine_re_verifies_evidence_itself():
    """Upstream claims the evidence is trusted; the engine's own registry check finds the poisoned document."""
    docs = build_documents()
    poison, _ = A.poison_documents(docs)
    trust = DocumentTrust(build_registry(docs, None)["docs"], {d.doc_id: d for d in docs + poison},
                          logged_descriptions(load_all_events()), calibrate_stuffing(docs))
    rec = best_answer(evidence=[ev("trouble-wind-pitch_system-rev2::0", trusted=True)],
                      cited_chunk_ids=["trouble-wind-pitch_system-rev2::0"],
                      text="Check the batteries. [trouble-wind-pitch_system-rev2::0]")
    for w in GRID[::50]:
        d = engine(w, trust=trust).evaluate(rec)
        assert d.decision == "reject" and "failed trust verification" in " ".join(d.reasons)
    genuine = engine(trust=trust).evaluate(best_answer())
    assert genuine.decision == "accept", genuine.reasons


# ------------------------------------------------------------------ constraints and risk rules

def test_unsafe_directives_catch_the_c2_procedures_but_not_the_genuine_kb():
    for text in (A.PITCH_INSERT, A.GEAR_INSERT, A.STUFF_INSERT, A.M0_INSERT):
        assert unsafe_directives(text), text
    flagged = [d.doc_id for d in build_documents() if unsafe_directives(d.text)]
    assert flagged == []                        # includes "never bridge the safety chain to restart a turbine"


def test_high_severity_without_strong_evidence_is_rejected():
    rec = best_diagnosis(severity="high", evidence_support="weak", confidence_level="low")
    assert not next(c for c in check_constraints(rec) if c.name == "high_severity_needs_strong_evidence").passed
    assert engine().decide(rec).decision == "reject"


def test_action_must_reference_the_component():
    rec = best_diagnosis(action_text="Inspect the turbine; no troubleshooting procedure was retrieved.")
    names = {c.name: c.passed for c in check_constraints(rec)}
    assert names["action_references_component"] is False
    assert engine().decide(rec).decision in ("review", "reject")


def test_accept_path_and_tampering_flag():
    assert engine().decide(best_answer()).decision == "accept"
    tampered = best_answer(security=SecurityFacts(has_user_question=True, injection_checked=True, trust_checked=True,
                                                  registry_source="signed registry file",
                                                  rejected_doc_ids=["trouble-wind-gearbox-rev2"]))
    d = engine().decide(tampered)
    assert d.decision == "review" and any("tampering" in r for r in d.reasons)


def test_missing_security_checks_cap_the_decision():
    rec = best_answer(security=SecurityFacts(has_user_question=True))           # no injection or trust check ran
    d = engine().decide(rec)
    assert d.components["S"] == 0.0 and d.decision != "accept"


def test_fallback_answer_scores_lower_evidence_quality():
    fb = best_answer(route="extractive_fallback")
    assert engine().components(fb)["E"] == pytest.approx(SIG.fallback_relevance_ratio.value)


def test_forecast_confidence_uses_the_measured_coverage():
    rec = Recommendation(kind="forecast", route="computed", text="12000 kW [9000, 15000]", forecast_asset="solar_plant1",
                         interval=(9000.0, 12000.0, 15000.0), capacity_kw=33000.0,
                         security=SecurityFacts(telemetry_checked=True))
    nominal, low, _ = SIG.forecast_coverage_low["solar_plant1"]
    comp = engine().components(rec)
    assert comp["C"] == pytest.approx(1 - (nominal - low) / (1 - nominal))
    assert comp["S"] == pytest.approx(SIG.telemetry_check["solar_plant1"].strength)
    unchecked = rec.model_copy(update={"security": SecurityFacts()})
    assert engine().decide(unchecked).decision != "accept"
    bad = rec.model_copy(update={"interval": (16000.0, 12000.0, 15000.0)})
    assert engine().decide(bad).decision == "reject"


# ------------------------------------------------------------------ required test 3, through the real C4 pipeline

class _Hits:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, k=5):
        return self.hits[:k]


def _perfect_llm(prompt):
    """Whatever it is asked, returns a well-formed answer citing the first chunk in the prompt."""
    import json
    import re

    cid = re.search(r"\[([^\]\s]+::\d+)\]", prompt).group(1)
    return json.dumps({"answer": "Check the pitch battery voltages and the last battery test result.",
                       "chunk_ids": [cid]})


def _secure_pipeline(store_docs, hit_docs):
    from src.backend.security.evidence_filter import EvidenceFilter, TrustedRetriever
    from src.backend.security.secure_pipeline import SecureRAGPipeline

    docs = build_documents()
    trust = DocumentTrust(build_registry(docs, None)["docs"], {d.doc_id: d for d in store_docs},
                          logged_descriptions(load_all_events()), calibrate_stuffing(docs))
    hits = [{"chunk_id": f"{d.doc_id}::0", "text": d.text[:400], "score": 8.0,
             "metadata": {"doc_id": d.doc_id, "chunk_index": 0}} for d in hit_docs]
    return SecureRAGPipeline(TrustedRetriever(_Hits(hits), EvidenceFilter(trust)), _perfect_llm,
                             registry_source="in-process build"), trust


def test_injected_question_blocked_by_c4_is_rejected_by_the_engine():
    from src.backend.trust.decision import from_secure_answer

    docs = build_documents()
    pitch = next(d for d in docs if d.doc_id == "trouble-wind-pitch_system")
    pipe, trust = _secure_pipeline(docs, [pitch])
    question = A.inject_query("What should be checked when the pitch system reports a battery fault?")
    ans = pipe.answer(question)
    assert ans.route == "blocked"
    rec = from_secure_answer(ans)
    for w in GRID:
        assert engine(w, trust=trust).evaluate(rec).decision == "reject"
    # the same question without the injection, same LLM: accepted, so the reject comes from the block
    clean = from_secure_answer(pipe.answer("What should be checked when the pitch system reports a battery fault?"))
    assert clean.route == "generated" and engine(trust=trust).evaluate(clean).decision == "accept"


def test_poisoned_store_never_yields_accept_through_the_real_filter():
    from src.backend.trust.decision import from_secure_answer

    docs = build_documents()
    poison, _ = A.poison_documents(docs)
    pipe, trust = _secure_pipeline(docs + poison, poison)          # only poisoned documents are retrieved
    rec = from_secure_answer(pipe.answer("What should be checked when the pitch system reports a battery fault?"))
    assert rec.route == "no_trusted_evidence"
    for w in GRID:
        assert engine(w, trust=trust).evaluate(rec).decision != "accept"
