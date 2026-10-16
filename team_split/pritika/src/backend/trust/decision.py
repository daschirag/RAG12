"""C5 - Trust-aware decision engine.

    LLM recommendation -> evidence verification -> security verification -> operational constraint check
    -> risk assessment -> trust score -> final decision (accept / review / reject)

The LLM (or the evidence-only fallback) only *recommends*. Nothing here executes an action: the output is a
``Decision`` that a person or a downstream system acts on, and only "accept" may be acted on without review.

Trust score ``T = wE*E + wR*R + wC*C + wS*S`` (weights configurable, non-negative, summing to 1), each in [0, 1]:

- **E evidence quality** = citation validity x route factor x evidence support.
  - Citation validity: the share of cited chunks that are in the trusted evidence given to the model; with no
    citation it is 0 (numeric answers, which are computed from data, count as 1).
  - Route factor: 1 for LLM-generated or computed answers. The evidence-only fallback gets B6's measured
    answer relevance relative to generated answers (0.845): a quote is faithful but answers less well.
  - Evidence support (diagnoses, B5): strong 1, weak 0.5, none 0.
  - E is 0 if any evidence used failed trust verification.
- **R retrieval relevance** = the mean sigmoid of the cross-encoder logit over the evidence the recommendation
  uses (its citations, else all evidence given). The ms-marco cross-encoder is trained as a binary relevance
  classifier, so the sigmoid is its relevance probability.
- **C confidence of the underlying model**:
  - Diagnoses: the B4 detector's measured reliability, taken at the lower end of its 95% CI and rescaled so
    that chance (AUC 0.5) gives 0. For B4, ci_low = 0.500, so C = 0: the detector is not distinguishable from
    chance.
  - Knowledge answers: the lower end of GreenRAG's B6 faithfulness CI.
  - Forecasts: the A5 coverage CI shortfall against the nominal level, as a fraction of the miss budget.
- **S security confidence**: the weakest check that ran on this request, at its C3 measured strength
  ``1 - (1 - TPR) / (1 - FPR)`` (one minus the negative likelihood ratio: how much a "pass" lowers the odds of an
  attack; detection rate and false-positive rate both from C3).
  - Question present: the injection detector (held-out phrasings).
  - Evidence: document trust (full registry check, or content checks only if the registry is unsigned).
  - Forecast input: the telemetry detector of that asset.
  - A check that should have run but did not gives 0. A blocked request gives 0.

Decision rules (``DecisionConfig``, documented in docs/decision_engine.md):

- **reject**: the request was blocked, untrusted evidence was used, a hard constraint failed, or T < ``review_at``.
- **accept**: all of the following hold:
  - T >= ``accept_at``;
  - every component is at least its floor (E, C and S >= 0.5; R >= 0.3);
  - no soft constraint failed;
  - no KB tampering was seen in this request (trust verification rejected retrieved documents).
- **review**: everything else.

Because of the floors, a recommendation whose underlying model is not shown to beat chance (C < 0.5), or
whose security checks are weak, can at best reach review, whatever the weights.
"""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..kb.taxonomy import COMPONENT_BY_KEY
from .signals import Signals, load_signals

LLM_ROUTES = ("generated", "llm")
COMPUTED_ROUTES = ("numeric",)
FALLBACK_ROUTES = ("extractive_fallback", "fallback")
TRUSTED_REGISTRIES = ("signed registry file", "in-process build from code and datasets", "in-process build",
                      "supplied reference")


# ------------------------------------------------------------------ inputs

class EvidenceUse(BaseModel):
    chunk_id: str
    doc_id: str
    rerank_score: float
    trusted: bool | None = Field(None, description="Result of document trust verification; None = not checked.")


class SecurityFacts(BaseModel):
    blocked: bool = False
    has_user_question: bool = False
    injection_checked: bool = False
    trust_checked: bool = False
    registry_source: str = ""
    rejected_doc_ids: list[str] = Field(default_factory=list,
                                        description="Retrieved documents removed by trust verification.")
    telemetry_checked: bool = False


class Recommendation(BaseModel):
    """What the LLM or the fallback produced. It is a proposal, never an executed action."""

    kind: Literal["answer", "diagnosis", "forecast"]
    route: str
    text: str
    action_text: str = ""
    evidence: list[EvidenceUse] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    component: str | None = None
    severity: str | None = None
    evidence_support: str | None = None
    confidence_level: str | None = None
    forecast_asset: str | None = None
    interval: tuple[float, float, float] | None = Field(None, description="(lower, prediction, upper) for forecasts")
    capacity_kw: float | None = None
    security: SecurityFacts = Field(default_factory=SecurityFacts)


# ------------------------------------------------------------------ configuration

class TrustWeights(BaseModel):
    E: float = 0.35
    R: float = 0.15
    C: float = 0.30
    S: float = 0.20

    @model_validator(mode="after")
    def _valid(self) -> "TrustWeights":
        w = (self.E, self.R, self.C, self.S)
        if min(w) < 0 or abs(sum(w) - 1.0) > 1e-6:
            raise ValueError(f"weights must be non-negative and sum to 1, got {w}")
        return self


class DecisionConfig(BaseModel):
    weights: TrustWeights = Field(default_factory=TrustWeights)
    accept_at: float = 0.70
    review_at: float = 0.40
    floors: dict[str, float] = Field(default_factory=lambda: {"E": 0.5, "R": 0.3, "C": 0.5, "S": 0.5})

    @model_validator(mode="after")
    def _valid(self) -> "DecisionConfig":
        if not 0 <= self.review_at < self.accept_at <= 1:
            raise ValueError(f"need 0 <= review_at < accept_at <= 1, got {self.review_at}, {self.accept_at}")
        if set(self.floors) != {"E", "R", "C", "S"}:
            raise ValueError(f"floors must cover E, R, C and S, got {sorted(self.floors)}")
        if self.floors["C"] <= 0 or self.floors["S"] <= 0:
            # C = 0 means the underlying model is not shown to beat chance, S = 0 means no security check ran or
            # a check failed: neither may ever be accepted, so these floors cannot be configured away.
            raise ValueError("floors for C and S must be > 0")
        return self


# ------------------------------------------------------------------ outputs

class ConstraintResult(BaseModel):
    name: str
    passed: bool
    hard: bool
    detail: str = ""


class Decision(BaseModel):
    decision: Literal["accept", "review", "reject"]
    trust_score: float
    components: dict[str, float]
    weights: dict[str, float]
    constraints: list[ConstraintResult]
    reasons: list[str]
    recommendation: Recommendation


# ------------------------------------------------------------------ components

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _used_evidence(rec: Recommendation) -> list[EvidenceUse]:
    cited = [e for e in rec.evidence if e.chunk_id in set(rec.cited_chunk_ids)]
    return cited or rec.evidence


def evidence_quality(rec: Recommendation, sig: Signals) -> float:
    if rec.kind == "forecast":
        return 1.0 if rec.interval is not None else 0.0      # a forecast is computed, not retrieved
    if rec.security.blocked or rec.route in ("blocked", "no_trusted_evidence"):
        return 0.0
    if any(e.trusted is False for e in _used_evidence(rec)):
        return 0.0
    if rec.route in COMPUTED_ROUTES:
        validity = 1.0
    elif rec.cited_chunk_ids:
        ids = {e.chunk_id for e in rec.evidence}
        validity = sum(c in ids for c in rec.cited_chunk_ids) / len(rec.cited_chunk_ids)
    else:
        validity = 0.0
    route = 1.0 if rec.route in LLM_ROUTES + COMPUTED_ROUTES else \
        sig.fallback_relevance_ratio.value if rec.route in FALLBACK_ROUTES else 0.0
    support = {"strong": 1.0, "weak": 0.5, "none": 0.0}.get(rec.evidence_support, 1.0) \
        if rec.kind == "diagnosis" else 1.0
    return validity * route * support


def retrieval_relevance(rec: Recommendation) -> float:
    if rec.kind == "forecast" or rec.route in COMPUTED_ROUTES:
        return 1.0                                            # not a retrieval-based output
    used = _used_evidence(rec)
    return sum(_sigmoid(e.rerank_score) for e in used) / len(used) if used else 0.0


def model_confidence(rec: Recommendation, sig: Signals) -> float:
    if rec.kind == "diagnosis":
        return min(1.0, max(0.0, (sig.detector_ci_low - 0.5) / 0.5))
    if rec.kind == "forecast":
        nominal, low, _ = sig.forecast_coverage_low[rec.forecast_asset]
        return min(1.0, max(0.0, 1.0 - max(0.0, nominal - low) / (1.0 - nominal)))
    return sig.answer_faithfulness_low.value


def security_confidence(rec: Recommendation, sig: Signals) -> float:
    s = rec.security
    if s.blocked or any(e.trusted is False for e in _used_evidence(rec)):
        return 0.0
    checks = []
    if rec.kind == "forecast":
        check = sig.telemetry_check.get(rec.forecast_asset)
        checks.append(check.strength if s.telemetry_checked and check is not None else 0.0)
    else:
        if s.has_user_question:
            checks.append(sig.injection_check.strength if s.injection_checked else 0.0)
        if rec.evidence:
            if not s.trust_checked:
                checks.append(0.0)
            else:
                checks.append(sig.trust_check_full.strength if s.registry_source in TRUSTED_REGISTRIES
                              else sig.trust_check_content_only.strength)
    return min(checks) if checks else 0.0


# ------------------------------------------------------------------ operational constraints

_NEGATION = re.compile(r"\b(never|not|don't|do not|must not|should not|avoid)\b[\w\s,]{0,25}$", re.IGNORECASE)
UNSAFE_PATTERNS = (
    ("bypass_safety", r"\b(bridg\w*|bypass\w*|jumper\w*|override\w*)\b[^.]{0,40}\b(safety|protection|interlock|relay|chain)\b"),
    ("disable_protection", r"\bdisabl\w*\b[^.]{0,40}\b(test|protection|interlock|alarm|safety|brake)s?\b"),
    ("ignore_alarm", r"\bignor\w*\b[^.]{0,40}\b(alarms?|warnings?|temperature)\b"),
    ("defer_maintenance", r"\bdefer\w*\b[^.]{0,30}\bmaintenance\b"),
    ("remote_reset", r"\breset\w*\b[^.]{0,20}\balarms?\b[^.]{0,20}\bremote\w*|\bremote\w*\b[^.]{0,20}\breset\w*\b"),
    ("skip_inspection", r"\b(inspection|sampling|replacement)\b[^.]{0,20}\b(not required|not needed|unnecessary)\b|"
                        r"\bno (maintenance|inspection) (is )?(required|needed)\b"),
)
_UNSAFE = [(n, re.compile(p, re.IGNORECASE)) for n, p in UNSAFE_PATTERNS]


def unsafe_directives(text: str) -> list[str]:
    """Unsafe operating directives in the text; a match preceded by a negation ("never bridge ...") is safe."""
    found = []
    for name, rx in _UNSAFE:
        for m in rx.finditer(text or ""):
            if not _NEGATION.search(text[max(0, m.start() - 40):m.start()]):
                found.append(name)
                break
    return found


def _mentions_component(text: str, key: str) -> bool:
    comp = COMPONENT_BY_KEY.get(key)
    if comp is None:
        return False
    low = text.lower()
    words = [w for w in re.findall(r"[a-z]{4,}", comp.label.lower()) if w not in ("system", "with")]
    return any(re.search(p, low) for p in comp.event_patterns) or any(w in low for w in words)


def check_constraints(rec: Recommendation) -> list[ConstraintResult]:
    out = []
    unsafe = unsafe_directives(rec.text)
    out.append(ConstraintResult(name="no_unsafe_directive", passed=not unsafe, hard=True,
                                detail=", ".join(unsafe) or "no safety-bypass, alarm-ignoring or deferral directive"))
    if rec.kind in ("answer", "diagnosis") and rec.route not in ("blocked", "no_trusted_evidence"):
        ids = {e.chunk_id for e in rec.evidence}
        ok = rec.route in COMPUTED_ROUTES or (bool(rec.cited_chunk_ids) and set(rec.cited_chunk_ids) <= ids)
        out.append(ConstraintResult(name="claims_cite_trusted_evidence", passed=ok, hard=True,
                                    detail=f"cited {rec.cited_chunk_ids or 'nothing'}"))
    if rec.kind == "diagnosis":
        high_ok = rec.severity != "high" or (rec.evidence_support == "strong"
                                             and rec.confidence_level in ("moderate", "high"))
        out.append(ConstraintResult(
            name="high_severity_needs_strong_evidence", passed=high_ok, hard=True,
            detail=f"severity {rec.severity}, evidence support {rec.evidence_support}, confidence {rec.confidence_level}"))
        if rec.component:
            out.append(ConstraintResult(
                name="action_references_component", passed=_mentions_component(rec.action_text, rec.component),
                hard=False, detail=f"likely-cause component {rec.component}; action: {rec.action_text[:120]}"))
            others = [k for k in COMPONENT_BY_KEY if k != rec.component and _mentions_component(rec.action_text, k)
                      and not _mentions_component(rec.action_text, rec.component)]
            out.append(ConstraintResult(name="action_consistent_with_cause", passed=not others, hard=False,
                                        detail=("action targets " + ", ".join(others)) if others else "consistent"))
    if rec.kind == "forecast" and rec.interval is not None:
        lo, p, hi = rec.interval
        cap = rec.capacity_kw if rec.capacity_kw is not None else float("inf")
        out.append(ConstraintResult(name="interval_physically_valid", passed=0 <= lo <= p <= hi <= cap, hard=True,
                                    detail=f"[{lo:.1f}, {p:.1f}, {hi:.1f}] kW, capacity {cap}"))
    return out


# ------------------------------------------------------------------ engine

class DecisionEngine:
    """``trust`` (a C3 ``DocumentTrust``) enables the engine's own security verification: every evidence document
    is re-checked against the trusted registry, whatever the upstream pipeline reported (defence in depth)."""

    def __init__(self, config: DecisionConfig | None = None, signals: Signals | None = None, trust=None,
                 trust_source: str = "in-process build") -> None:
        self.config = config or DecisionConfig()
        self.signals = signals or load_signals()
        self.trust, self.trust_source = trust, trust_source

    def verify_security(self, rec: Recommendation) -> Recommendation:
        """Security verification stage: document trust on every evidence item (if the engine has a registry)."""
        if self.trust is None or not rec.evidence:
            return rec
        evidence = []
        for e in rec.evidence:
            doc = self.trust.store.get(e.doc_id)
            ok = doc is not None and self.trust.check_document(doc).trusted
            evidence.append(e.model_copy(update={"trusted": ok and e.trusted is not False}))
        security = rec.security.model_copy(update={"trust_checked": True, "registry_source": self.trust_source})
        return rec.model_copy(update={"evidence": evidence, "security": security})

    def evaluate(self, rec: Recommendation) -> Decision:
        """Full pipeline: evidence + security verification, constraints, risk assessment, trust score, decision."""
        return self.decide(self.verify_security(rec))

    def components(self, rec: Recommendation) -> dict[str, float]:
        sig = self.signals
        return {"E": evidence_quality(rec, sig), "R": retrieval_relevance(rec), "C": model_confidence(rec, sig),
                "S": security_confidence(rec, sig)}

    def decide(self, rec: Recommendation, components: dict[str, float] | None = None,
               constraints: list[ConstraintResult] | None = None) -> Decision:
        comp = components if components is not None else self.components(rec)
        cons = constraints if constraints is not None else check_constraints(rec)
        decision, t, reasons = decide_core(comp, gate_facts(rec, cons), self.config)
        return Decision(decision=decision, trust_score=t, components={k: round(v, 6) for k, v in comp.items()},
                        weights=self.config.weights.model_dump(), constraints=cons, reasons=reasons,
                        recommendation=rec)


def gate_facts(rec: Recommendation, cons: list[ConstraintResult]) -> dict:
    """Weight-independent facts the risk assessment needs (also used by the sensitivity analysis)."""
    hard = []
    if rec.security.blocked:
        hard.append("request blocked by input security / injection detection")
    if any(e.trusted is False for e in _used_evidence(rec)):
        hard.append("recommendation uses evidence that failed trust verification")
    hard += [f"hard constraint failed: {c.name} ({c.detail})" for c in cons if c.hard and not c.passed]
    soft = [f"soft constraint failed: {c.name} ({c.detail})" for c in cons if not c.hard and not c.passed]
    if rec.security.rejected_doc_ids:
        soft.append("KB tampering seen in this request: retrieved documents failed trust verification ("
                    + ", ".join(sorted(set(rec.security.rejected_doc_ids))) + ")")
    return {"hard": hard, "soft": soft}


def decide_core(comp: dict[str, float], facts: dict, cfg: DecisionConfig) -> tuple[str, float, list[str]]:
    """Risk assessment: (decision, trust score, reasons) from components, gate facts and the configuration."""
    w = cfg.weights
    t = round(w.E * comp["E"] + w.R * comp["R"] + w.C * comp["C"] + w.S * comp["S"], 6)
    if facts["hard"]:
        return "reject", t, list(facts["hard"])
    if t < cfg.review_at:
        return "reject", t, [f"trust score {t:.3f} below review threshold {cfg.review_at}"]
    gaps = [f"{k} = {comp[k]:.3f} below floor {cfg.floors[k]}" for k in cfg.floors if comp[k] < cfg.floors[k]]
    gaps += facts["soft"]
    if t < cfg.accept_at:
        gaps.insert(0, f"trust score {t:.3f} below accept threshold {cfg.accept_at}")
    return ("review", t, gaps) if gaps else ("accept", t, [f"trust score {t:.3f}; all checks passed"])


# ------------------------------------------------------------------ adapters

def from_secure_answer(ans) -> Recommendation:
    """A C4 ``SecureAnswer`` (question answering) as a recommendation."""
    rejected = {r.doc_id for r in ans.security.rejected}
    return Recommendation(
        kind="answer", route=ans.route, text=ans.answer, action_text=ans.answer,
        evidence=[EvidenceUse(chunk_id=e.chunk_id, doc_id=e.doc_id, rerank_score=e.rerank_score, trusted=True)
                  for e in ans.evidence],
        cited_chunk_ids=ans.cited_chunk_ids,
        security=SecurityFacts(blocked=ans.security.blocked, has_user_question=True,
                               injection_checked=ans.security.input_ok, trust_checked=not ans.security.blocked,
                               registry_source=ans.security.registry_source, rejected_doc_ids=sorted(rejected)))


def from_diagnosis(report, security: SecurityFacts) -> Recommendation:
    """A B5/B7 ``DiagnosisReport`` as a recommendation. ``security`` says which checks the retrieval ran."""
    rejected = set(security.rejected_doc_ids)
    cited = list(dict.fromkeys(report.likely_cause.chunk_ids + report.recommended_action.chunk_ids))
    return Recommendation(
        kind="diagnosis", route=report.generation,
        text=f"{report.diagnosis} Likely cause: {report.likely_cause.text} "
             f"Recommended action: {report.recommended_action.text}",
        action_text=report.recommended_action.text,
        evidence=[EvidenceUse(chunk_id=e.chunk_id, doc_id=e.doc_id, rerank_score=e.rerank_score,
                              trusted=(e.doc_id not in rejected) if security.trust_checked else None)
                  for e in report.evidence],
        cited_chunk_ids=cited, component=report.likely_cause_component, severity=report.severity,
        evidence_support=report.confidence.evidence_support, confidence_level=report.confidence.level,
        security=security)
