"""B5 - Fault diagnosis connected to RAG: anomaly signal -> retrieval -> LLM -> validated report.

Pipeline: a fault signal (turbine, window, anomaly score from the B4 detector)
-> a retrieval query -> GreenRAG production retrieval (hybrid BM25 + vector
with cross-encoder rerank over the knowledge base) -> an LLM writes the
diagnosis from the retrieved evidence only -> a pydantic-validated report.

Honesty rules (from B4; fixed before running):
- ``confidence`` is NOT a probability that the diagnosis is correct. It reports
  the detector's anomaly score together with the detector's measured
  reliability (event AUC and its 95% CI from B4) and how well the retrieved
  evidence supports this specific incident.
- ``likely_cause`` comes only from retrieved evidence and cites real chunk ids;
  B4's component attribution (chance level) is never used.
- Severity is deterministic: the default severity of the evidence-derived
  component (taxonomy assumption from B1), "undetermined" if no component.
- If the LLM fails or returns invalid output (bad JSON, uncited or unknown chunk
  ids), a deterministic evidence-only fallback is used and marked as such.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from ..kb.taxonomy import COMPONENT_BY_KEY

PROJECT_ROOT = Path(__file__).resolve().parents[3]
B4_DIR = PROJECT_ROOT / "results" / "faultdiag"
DETECTOR = "logistic_regression"          # best mean leave-one-asset-out event AUC in B4 (selected on CV results)
EVIDENCE_K = 5
POOL_K = 20
INCIDENT_TOLERANCE = pd.Timedelta("7D")   # evidence counts as "this incident" if its window is within +/- 7 days
MIN_RERANK_SCORE = 0.0                    # cross-encoder logit; <= 0 means the reranker judged it not relevant

CONFIDENCE_DESCRIPTION = (
    "Measured reliability of the anomaly score plus evidence support. NOT a probability that the diagnosis is "
    "correct. level is 'high' only with strong incident evidence AND a detector whose event-AUC 95% CI lower bound "
    "exceeds 0.5 AND an anomaly score >= 0.5; 'moderate' with strong incident evidence otherwise; 'low' when the "
    "evidence is weak or absent."
)


# ------------------------------------------------------------------------ schema

class Reliability(BaseModel):
    detector: str
    metric: Literal["event_auc"] = "event_auc"
    value: float = Field(..., ge=0, le=1)
    ci95_low: float = Field(..., ge=0, le=1)
    ci95_high: float = Field(..., ge=0, le=1)
    n_events: int
    protocol: str


class Confidence(BaseModel):
    level: Literal["low", "moderate", "high"] = Field(..., description=CONFIDENCE_DESCRIPTION)
    anomaly_score: float | None = Field(None, ge=0, le=1, description="Detector event score (not calibrated).")
    reliability: Reliability
    evidence_support: Literal["strong", "weak", "none"]
    explanation: str


class EvidenceItem(BaseModel):
    chunk_id: str
    doc_id: str
    doc_type: str
    title: str
    synthetic: bool
    farm: str | None = None
    asset_id: int | None = None
    event_id: int | None = None
    rerank_score: float
    excerpt: str
    expanded_from: str | None = Field(None, description="Chunk id of the retrieved hit this sibling chunk was added for.")


class CitedText(BaseModel):
    text: str = Field(..., min_length=3)
    chunk_ids: list[str] = Field(default_factory=list)
    source: Literal["retrieved_evidence"] = "retrieved_evidence"


class DiagnosisReport(BaseModel):
    farm: str
    asset_id: int
    event_id: int | None = None
    diagnosis: str = Field(..., min_length=3)
    confidence: Confidence
    likely_cause: CitedText = Field(..., description="Derived from retrieved evidence only (cites chunk ids).")
    likely_cause_component: str | None = None
    evidence: list[EvidenceItem]
    recommended_action: CitedText
    severity: Literal["high", "medium", "low", "undetermined"]
    severity_basis: str
    generation: Literal["llm", "fallback"]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "DiagnosisReport":
        ids = {e.chunk_id for e in self.evidence}
        for part in (self.likely_cause, self.recommended_action):
            unknown = set(part.chunk_ids) - ids
            if unknown:
                raise ValueError(f"cited chunk ids not in the evidence: {sorted(unknown)}")
        if self.evidence and not self.likely_cause.chunk_ids:
            raise ValueError("likely_cause must cite at least one evidence chunk")
        if self.confidence.evidence_support in ("weak", "none") and self.confidence.level != "low":
            raise ValueError("weak or missing evidence must give low confidence")
        if self.confidence.evidence_support == "none" and self.likely_cause.chunk_ids:
            raise ValueError("no evidence: likely_cause cannot cite chunks")
        return self


# ---------------------------------------------------------------- signal inputs

@dataclass(frozen=True)
class FaultSignal:
    farm: str
    asset_id: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    anomaly_score: float | None
    event_id: int | None = None
    detector: str = DETECTOR


def load_reliability(b4_dir: Path = B4_DIR, detector: str = DETECTOR) -> Reliability:
    """Measured event AUC and 95% CI of the detector, recomputed from the stored B4 predictions."""
    from .models import bootstrap_event_metric

    preds = pd.read_csv(b4_dir / "b4_event_predictions.csv")
    metrics = pd.read_csv(b4_dir / "b4_event_metrics.csv")
    ev0 = preds[(preds.model == detector) & (preds.seed == 0)]
    lo, hi = bootstrap_event_metric(ev0, "auc")
    auc = float(metrics[metrics.model == detector]["auc"].mean())
    return Reliability(detector=detector, value=round(auc, 3), ci95_low=round(lo, 3), ci95_high=round(hi, 3),
                       n_events=int(len(ev0)), protocol="leave-one-asset-out, Wind Farm A, 22 events (B4)")


def load_signals(b4_dir: Path = B4_DIR, detector: str = DETECTOR, farm: str = "A") -> list[FaultSignal]:
    """One signal per Farm A event: out-of-fold detector score (seed 0) and the event window."""
    from ..kb.taxonomy import load_events

    preds = pd.read_csv(b4_dir / "b4_event_predictions.csv")
    score = preds[(preds.model == detector) & (preds.seed == 0)].set_index("event_id")["event_prob"]
    events = load_events(farm)
    return [FaultSignal(farm=farm, asset_id=int(e.asset_id), window_start=e.event_start, window_end=e.event_end,
                        anomaly_score=float(score.loc[e.event_id]), event_id=int(e.event_id))
            for e in events.itertuples()]


def event_windows(farm_events: pd.DataFrame) -> dict[tuple[str, int], tuple[pd.Timestamp, pd.Timestamp]]:
    return {(r.farm, int(r.event_id)): (r.event_start, r.event_end) for r in farm_events.itertuples()}


# -------------------------------------------------------------------- diagnoser

def build_query(signal: FaultSignal) -> str:
    return (f"Wind Farm {signal.farm} turbine {signal.asset_id}: anomaly around {signal.window_start:%Y-%m-%d} "
            f"({signal.window_start:%B %Y}). What is the likely cause, what maintenance was done and what should "
            "be checked?")


def build_prompt(signal: FaultSignal, evidence: list[EvidenceItem]) -> str:
    blocks = "\n\n".join(f"[{e.chunk_id}] ({e.doc_type}{', SYNTHETIC' if e.synthetic else ''})\n{e.excerpt}"
                         for e in evidence)
    return (
        "You are a wind-turbine maintenance assistant. Use ONLY the evidence below. Do not use outside knowledge "
        "and do not invent values. Documents marked SYNTHETIC are generated guidance, not real records.\n\n"
        f"Turbine: Wind Farm {signal.farm}, asset {signal.asset_id}; anomaly window starting "
        f"{signal.window_start:%Y-%m-%d}.\n\nEvidence:\n{blocks}\n\n"
        "Return ONLY a JSON object with these keys:\n"
        '{"diagnosis": "<one sentence>", "likely_cause": "<one or two sentences>", '
        '"likely_cause_chunk_ids": ["<ids from the evidence>"], "recommended_action": "<one or two sentences>", '
        '"action_chunk_ids": ["<ids from the evidence>"]}\n'
        "Cite only chunk ids that appear in square brackets above. If the evidence does not describe this "
        "turbine's fault, say that the cause is not established by the evidence."
    )


LLM_TIMEOUT_S = 120.0     # one Ollama call; a stalled call once blocked an experiment for ~11 hours


def ollama_llm(model: str | None = None, json_mode: bool = False,
               timeout_s: float | None = None) -> Callable[[str], str]:
    """Local LLM (Ollama), temperature 0 / seed 0, with a per-call timeout.

    Retries once after a timeout or a runner crash; a second failure raises, and callers record it.
    The timeout (default 120 s) and the Ollama host come from the settings (GREENRAG_LLM_TIMEOUT_S, OLLAMA_HOST).
    """
    import os

    import httpx
    import ollama

    from ..serving.generator import DEFAULT_OLLAMA_MODEL, _is_runner_crash

    name = model or (os.getenv("OLLAMA_MODEL") or "").strip() or DEFAULT_OLLAMA_MODEL
    from ..settings import get_settings

    settings = get_settings()
    # the ollama client reads OLLAMA_HOST itself (same variable as settings.ollama_host)
    client = ollama.Client(timeout=timeout_s if timeout_s is not None else settings.llm_timeout_s)

    def call(prompt: str) -> str:
        kwargs = {"model": name, "messages": [{"role": "user", "content": prompt}],
                  "options": {"temperature": 0, "seed": 0}}
        if json_mode:
            kwargs["format"] = "json"
        try:
            response = client.chat(**kwargs)
        except httpx.TimeoutException:
            response = client.chat(**kwargs)
        except ollama.ResponseError as exc:
            if not _is_runner_crash(exc):
                raise
            response = client.chat(**kwargs)
        return response["message"]["content"]

    return call


def ollama_json_llm(model: str | None = None) -> Callable[[str], str]:
    """Local LLM (Ollama) in JSON mode (see ``ollama_llm``)."""
    return ollama_llm(model, json_mode=True)


class Diagnoser:
    def __init__(self, retriever, llm: Callable[[str], str] | None, reliability: Reliability,
                 windows: dict[tuple[str, int], tuple[pd.Timestamp, pd.Timestamp]],
                 k: int = EVIDENCE_K, pool: int = POOL_K, expander=None) -> None:
        self.retriever, self.llm, self.reliability = retriever, llm, reliability
        self.windows, self.k, self.pool = windows, k, pool
        self.expander = expander            # optional SiblingExpander (retrieval-time sibling chunks)

    # -- evidence --------------------------------------------------------------
    def retrieve(self, signal: FaultSignal, exclude_own_event: bool = False) -> list[EvidenceItem]:
        kept = []
        for h in self.retriever.search(build_query(signal), self.pool):
            m = h["metadata"]
            if exclude_own_event and m.get("farm") == signal.farm and m.get("event_id") == signal.event_id:
                continue
            kept.append(h)
            if len(kept) == self.k:
                break
        # Siblings come from the kept hits' own documents, so excluded documents stay excluded.
        hits = self.expander.expand(kept) if self.expander is not None else kept
        return [EvidenceItem(chunk_id=h["chunk_id"], doc_id=h["metadata"]["doc_id"], doc_type=h["metadata"]["doc_type"],
                             title=h["metadata"].get("title", ""), synthetic=bool(h["metadata"].get("synthetic", False)),
                             farm=h["metadata"].get("farm"), asset_id=h["metadata"].get("asset_id"),
                             event_id=h["metadata"].get("event_id"), rerank_score=float(h.get("score", 0.0)),
                             excerpt=h["text"][:900], expanded_from=h["metadata"].get("expanded_from"))
                for h in hits]

    def is_incident_evidence(self, e: EvidenceItem, signal: FaultSignal) -> bool:
        """Real-data evidence about this turbine whose event window is within the tolerance of the signal."""
        if e.synthetic or e.farm != signal.farm or e.asset_id != signal.asset_id or e.event_id is None:
            return False
        win = self.windows.get((e.farm, int(e.event_id)))
        if win is None:
            return False
        return win[0] - INCIDENT_TOLERANCE <= signal.window_end and signal.window_start <= win[1] + INCIDENT_TOLERANCE

    def evidence_support(self, evidence: list[EvidenceItem], signal: FaultSignal) -> str:
        if not evidence:
            return "none"
        strong = any(self.is_incident_evidence(e, signal) and e.rerank_score > MIN_RERANK_SCORE for e in evidence)
        return "strong" if strong else "weak"

    def confidence(self, signal: FaultSignal, support: str) -> Confidence:
        detector_ok = self.reliability.ci95_low > 0.5
        score_ok = signal.anomaly_score is not None and signal.anomaly_score >= 0.5
        if support != "strong":
            level = "low"
        elif detector_ok and score_ok:
            level = "high"
        else:
            level = "moderate"
        r = self.reliability
        explanation = (f"Anomaly score {signal.anomaly_score if signal.anomaly_score is not None else 'n/a'} from "
                       f"{r.detector}; measured event AUC {r.value:.3f} (95% CI {r.ci95_low:.3f}-{r.ci95_high:.3f}, "
                       f"{r.n_events} events, {r.protocol})"
                       + ("; the CI reaches 0.5, so the detector is not distinguishable from chance"
                          if not detector_ok else "")
                       + f". Evidence support: {support}"
                       + (" (a retrieved real record about this incident)." if support == "strong" else
                          " (no retrieved record about this incident)." if support == "weak" else " (none retrieved)."))
        return Confidence(level=level, anomaly_score=signal.anomaly_score, reliability=r,
                          evidence_support=support, explanation=explanation)

    @staticmethod
    def evidence_component(evidence: list[EvidenceItem], incident: list[EvidenceItem],
                           components: dict[str, str | None]) -> tuple[str | None, str | None]:
        """(component, chunk_id): from the best incident evidence if any, else the best evidence with a component."""
        for pool in (incident, evidence):
            for e in pool:
                comp = components.get(e.doc_id)
                if comp:
                    return comp, e.chunk_id
        return None, None

    # -- report ----------------------------------------------------------------
    def diagnose(self, signal: FaultSignal, components: dict[str, str | None],
                 exclude_own_event: bool = False) -> DiagnosisReport:
        evidence = self.retrieve(signal, exclude_own_event)
        support = self.evidence_support(evidence, signal)
        conf = self.confidence(signal, support)
        incident = [e for e in evidence if self.is_incident_evidence(e, signal)]
        component, comp_chunk = self.evidence_component(evidence, incident, components)
        comp = COMPONENT_BY_KEY.get(component) if component else None
        # Severity is only asserted when the evidence describes this incident; otherwise a component-based
        # severity would present a guess as certain.
        if comp and support == "strong":
            severity = comp.default_severity
            severity_basis = (f"default severity of component '{component}' (project assumption, B1 taxonomy), "
                              f"derived from incident evidence {comp_chunk}")
        elif comp:
            severity = "undetermined"
            severity_basis = (f"evidence does not describe this incident; the closest evidence ({comp_chunk}) concerns "
                              f"'{component}', whose default severity would be {comp.default_severity}")
        else:
            severity = "undetermined"
            severity_basis = "no wind-turbine component could be derived from the evidence"
        notes = []
        if exclude_own_event:
            notes.append("unlogged mode: this event's own fault and maintenance documents were excluded")
        fields, generation = None, "fallback"
        if self.llm is not None and evidence:
            try:
                fields = self._parse_llm(self.llm(build_prompt(signal, evidence)), evidence)
                generation = "llm"
            except (ValueError, KeyError, TypeError) as exc:
                notes.append(f"LLM output rejected ({type(exc).__name__}: {str(exc)[:120]}); fallback used")
            except Exception as exc:  # noqa: BLE001 - LLM backend errors must not block a report
                notes.append(f"LLM unavailable ({type(exc).__name__}); fallback used")
        if fields is None:
            fields = self._fallback(signal, evidence, incident, component, comp_chunk)
        return DiagnosisReport(
            farm=signal.farm, asset_id=signal.asset_id, event_id=signal.event_id, diagnosis=fields["diagnosis"],
            confidence=conf, likely_cause=CitedText(text=fields["likely_cause"],
                                                    chunk_ids=fields["likely_cause_chunk_ids"]),
            likely_cause_component=component, evidence=evidence,
            recommended_action=CitedText(text=fields["recommended_action"], chunk_ids=fields["action_chunk_ids"]),
            severity=severity, severity_basis=severity_basis, generation=generation, notes=notes)

    @staticmethod
    def _parse_llm(raw: str, evidence: list[EvidenceItem]) -> dict:
        data = json.loads(raw)
        ids = {e.chunk_id for e in evidence}
        out = {}
        for key in ("diagnosis", "likely_cause", "recommended_action"):
            value = data[key]
            if not isinstance(value, str) or len(value.strip()) < 3:
                raise ValueError(f"missing or empty {key}")
            out[key] = value.strip()
        by_doc: dict[str, list[str]] = {}
        for e in evidence:
            by_doc.setdefault(e.chunk_id.split("::")[0], []).append(e.chunk_id)
        for key in ("likely_cause_chunk_ids", "action_chunk_ids"):
            cited = data.get(key) or []
            if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
                raise ValueError(f"{key} must be a list of chunk ids")
            resolved = []
            for c in cited:
                if c in ids:
                    resolved.append(c)
                elif c in by_doc:
                    # The model often drops the "::k" chunk suffix; a bare id is accepted only when it names a
                    # retrieved document, and it resolves to that document's evidence chunks (never outside).
                    resolved.extend(by_doc[c])
                else:
                    raise ValueError(f"{key} cites ids not in the evidence: {c!r}")
            out[key] = list(dict.fromkeys(resolved))
        if not out["likely_cause_chunk_ids"]:
            raise ValueError("likely_cause cites no evidence")
        return out

    @staticmethod
    def _fallback(signal, evidence, incident, component, comp_chunk) -> dict:
        if not evidence:
            return {"diagnosis": f"Anomaly signal for Wind Farm {signal.farm} turbine {signal.asset_id}; no evidence "
                                 "was retrieved.",
                    "likely_cause": "Not established: no evidence retrieved.", "likely_cause_chunk_ids": [],
                    "recommended_action": "Inspect the turbine; no documented procedure was retrieved.",
                    "action_chunk_ids": []}
        if incident:
            # Search every evidence chunk of the incident document(s), not only the first retrieved chunk:
            # the logged description can sit in a sibling chunk of the record.
            incident_docs = {e.doc_id for e in incident}
            found = None
            for e in evidence:
                if e.doc_id in incident_docs:
                    m = re.search(r'original wording\): "(.*?)"', e.excerpt, flags=re.S)
                    if m:
                        found = (e, m.group(1))
                        break
            if found:
                cause_src, cause = found[0], f"Logged description for this incident: \"{found[1]}\""
            else:
                cause_src = incident[0]
                cause = (f"The record of this incident ({cause_src.doc_id}) was retrieved, but the retrieved text "
                         "does not state the logged cause.")
        else:
            cause_src = next((e for e in evidence if e.chunk_id == comp_chunk), evidence[0])
            cause = (f"Not established for this incident; the closest evidence concerns "
                     f"{COMPONENT_BY_KEY[component].label.lower() if component in COMPONENT_BY_KEY else 'an unrelated topic'}"
                     f" ({cause_src.doc_id}).")
        action_src = next((e for e in evidence if e.doc_type == "troubleshooting_procedure"), None)
        action = (f"Follow the checks in {action_src.doc_id} (synthetic procedure)." if action_src else
                  "Inspect the turbine; no troubleshooting procedure was retrieved.")
        return {"diagnosis": f"Anomaly signal for Wind Farm {signal.farm} turbine {signal.asset_id}"
                             + (f"; evidence points to the {COMPONENT_BY_KEY[component].label.lower()}."
                                if component in COMPONENT_BY_KEY else "."),
                "likely_cause": cause, "likely_cause_chunk_ids": [cause_src.chunk_id],
                "recommended_action": action, "action_chunk_ids": [action_src.chunk_id] if action_src else []}


def production_retriever(docs, chunks=None, embed=None):
    """GreenRAG production retrieval (B3): hybrid BM25 + MiniLM, RRF, cross-encoder rerank over semantic chunks.

    Pass ``chunks``/``embed`` to reuse already-built semantic chunks (avoids embedding the KB twice).
    """
    from ..kb.chunking import semantic_chunks
    from ..kb.retrieval_eval import CosineIndex, minilm_embed
    from ..retrieval.hybrid_retriever import BM25Index, HybridRetriever, RerankRetriever, cross_encoder_scorer

    embed = embed or minilm_embed()
    chunks = chunks if chunks is not None else semantic_chunks(docs, embed)
    hybrid = HybridRetriever([BM25Index(chunks), CosineIndex(chunks, embed)], k_rrf=60, candidates=50)
    return RerankRetriever(hybrid, cross_encoder_scorer(), candidates=POOL_K)
