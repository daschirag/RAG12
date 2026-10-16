"""I1 - Orchestrator behind POST /assistant: forecast + diagnosis + security + trust, with ablation configs A-F.

Intent routing (rule-based, fixed): ``forecast`` when an origin is given or the message asks for a forecast,
``diagnosis`` when an event id is given or the message names a fault event ("event 73"), else ``question``.

Ablation configs (docs/PLAN.md I1). Each adds one capability to the previous one:

| config | question path                     | forecast | diagnosis | security (C4) | trust (C5) |
|--------|-----------------------------------|----------|-----------|---------------|------------|
| A      | LLM only (no retrieval)           | -        | -         | -             | -          |
| B      | basic RAG (vector, fixed chunks)  | -        | -         | -             | -          |
| C      | hybrid RAG (BM25 + vector, RRF)   | -        | -         | -             | -          |
| D      | hybrid RAG                        | yes      | -         | -             | -          |
| E      | hybrid RAG                        | yes      | yes       | -             | -          |
| F      | C4 secure GreenRAG pipeline       | yes      | yes       | yes           | yes        |

A config without the module for an intent answers the message through its question path (that is what the
ablation removes). Only F produces a trust score and an accept / review / reject decision; A-E return
``trust = null``. The systems for A-C are the B6 systems (src/backend/evaluation/rag_systems.py), unchanged.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .faultdiag.service import DiagnoseRequest
from .forecasting.schemas import AssetId, ForecastRequest
from .trust.service import DecisionRequest

Config = Literal["A", "B", "C", "D", "E", "F"]
Intent = Literal["question", "forecast", "diagnosis"]


@dataclass(frozen=True)
class Flags:
    retrieval: str | None            # None | "basic" | "hybrid" | "secure"
    forecasting: bool
    diagnosis: bool
    security: bool
    trust: bool


ABLATION: dict[str, Flags] = {
    "A": Flags(None, False, False, False, False),
    "B": Flags("basic", False, False, False, False),
    "C": Flags("hybrid", False, False, False, False),
    "D": Flags("hybrid", True, False, False, False),
    "E": Flags("hybrid", True, True, False, False),
    "F": Flags("secure", True, True, True, True),
}

_FORECAST = re.compile(r"\b(forecast|predict(ion)?|next hour|power (output )?in an hour)\b", re.IGNORECASE)
_EVENT = re.compile(r"\b(?:fault )?event\s*#?\s*(\d+)\b", re.IGNORECASE)
_DIAGNOSE = re.compile(r"\bdiagnos\w*\b", re.IGNORECASE)


class AssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"message": "What should be checked when the pitch system reports a battery fault?", "config": "F"},
        {"message": "", "asset": "solar_plant1", "origin": "2020-06-16T11:00", "confidence": 0.9},
        {"message": "Diagnose fault event 73", "config": "F"}]})

    message: str = Field(..., description="The operator's message.")
    config: Config = Field("F", description="Ablation config A-F (F = full GreenRAG-Sec).")
    intent: Intent | None = Field(None, description="Override the rule-based intent routing.")
    asset: AssetId = Field("solar_plant1", description="Forecast asset.")
    origin: datetime | None = Field(None, description="Forecast origin (needed for a forecast).")
    confidence: float = Field(0.9, description="Forecast interval nominal coverage (0.8, 0.9 or 0.95).")
    event_id: int | None = Field(None, description="Wind Farm A event id for a diagnosis.")
    mode: Literal["logged", "unlogged"] = "logged"

    @model_validator(mode="after")
    def _message(self) -> "AssistantRequest":
        if not self.message.strip() and self.origin is None and self.event_id is None:
            raise ValueError("give a message, a forecast origin or an event_id")
        return self


class Trust(BaseModel):
    decision: Literal["accept", "review", "reject"]
    trust_score: float
    components: dict[str, float]
    weights: dict[str, float]
    reasons: list[str]


class AssistantResponse(BaseModel):
    config: str
    intent: str = Field(description="Intent found by the router.")
    handled_by: str = Field(description="Module that produced the output (question path when the intent's module "
                                        "is disabled in this config).")
    text: str = Field(description="Human-readable answer.")
    output: dict[str, Any] = Field(default_factory=dict, description="Full output of the module.")
    forecast: dict[str, Any] | None = Field(None, description="prediction, lower_bound, upper_bound, confidence.")
    trust: Trust | None = Field(None, description="Only in config F (C5 decision engine).")
    flags: dict[str, Any]


def route_intent(req: AssistantRequest) -> tuple[str, int | None]:
    """(intent, event id parsed from the message if any)."""
    m = _EVENT.search(req.message)
    event_id = req.event_id if req.event_id is not None else (int(m.group(1)) if m else None)
    if req.intent is not None:
        return req.intent, event_id
    if req.origin is not None or _FORECAST.search(req.message):
        return "forecast", event_id
    if req.event_id is not None or (m and (_DIAGNOSE.search(req.message) or "fault" in req.message.lower())):
        return "diagnosis", event_id
    return "question", event_id


class MissingInputError(ValueError):
    pass


@dataclass
class Orchestrator:
    """Lazy holder of every module; tests inject fakes (``qa`` maps a retrieval name to answer(question))."""

    qa: dict[str, Callable[[str], Any]] = field(default_factory=dict)   # "none" | "basic" | "hybrid"
    forecast_service: Any = None
    diagnosis_service: Any = None
    decision_service: Any = None
    llm_factory: Callable[[bool], Callable[[str], str]] | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    # -- question paths A-E (B6 systems) ---------------------------------------
    def _qa(self, name: str) -> Callable[[str], Any]:
        with self._lock:
            if name not in self.qa:
                self.qa[name] = self._build_qa(name)
            return self.qa[name]

    def _build_qa(self, name: str) -> Callable[[str], Any]:
        from .evaluation import rag_systems as R
        from .faultdiag.diagnosis import ollama_llm

        llm = self.llm_factory(False) if self.llm_factory else ollama_llm()
        if name == "none":
            return lambda q: R.llm_only(llm, q)
        from .kb.build import KB_DIR, build_documents, load_documents
        from .kb.chunking import fixed_size_chunks, semantic_chunks
        from .kb.retrieval_eval import CosineIndex, minilm_embed
        from .retrieval.hybrid_retriever import BM25Index, HybridRetriever

        path = KB_DIR / "documents.jsonl"
        docs = load_documents(path) if path.is_file() else build_documents()
        embed = minilm_embed()
        if name == "basic":
            index = CosineIndex(fixed_size_chunks(docs), embed)
            return lambda q: R.rag_answer("basic_rag", llm, index, q)
        sem = semantic_chunks(docs, embed)
        hybrid = HybridRetriever([BM25Index(sem), CosineIndex(sem, embed)], k_rrf=60, candidates=50)
        return lambda q: R.rag_answer("hybrid_rag", llm, hybrid, q)

    def _question(self, flags: Flags, message: str) -> tuple[str, str, dict, Trust | None]:
        if flags.retrieval == "secure":
            d = self.decision_service.decide(DecisionRequest(kind="answer", question=message))
            return "secure_pipeline", d.output["answer"], d.output, _trust(d)
        a = self._qa(flags.retrieval or "none")(message)
        system = {None: "llm_only", "basic": "basic_rag", "hybrid": "hybrid_rag"}[flags.retrieval]
        return system, a.text, {"system": a.system, "answer": a.text, "route": a.route, "context": a.context}, None

    # -- main ------------------------------------------------------------------
    def handle(self, req: AssistantRequest) -> AssistantResponse:
        flags = ABLATION[req.config]
        intent, event_id = route_intent(req)
        fc, trust = None, None
        if intent == "forecast" and flags.forecasting:
            if req.origin is None:
                raise MissingInputError("a forecast needs 'origin' (and 'asset')")
            freq = ForecastRequest(asset=req.asset, origin=req.origin, confidence=req.confidence)
            if flags.trust:
                d = self.decision_service.decide(DecisionRequest(kind="forecast", forecast=freq))
                out, trust = d.output, _trust(d)
            else:
                import pandas as pd

                out = self.forecast_service.forecast(freq.asset_id, pd.Timestamp(freq.origin), freq.confidence)
                out = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in out.items()}
            fc = {k: out[k] for k in ("prediction", "lower_bound", "upper_bound", "confidence")}
            text = (f"{req.asset}: {out['prediction']:.0f} kW expected at {out.get('target_time', 'origin + 1 h')}, "
                    f"{round(req.confidence * 100)}% interval [{out['lower_bound']:.0f}, {out['upper_bound']:.0f}] kW.")
            return AssistantResponse(config=req.config, intent=intent, handled_by="forecasting", text=text,
                                     output=out, forecast=fc, trust=trust, flags=flags.__dict__)
        if intent == "diagnosis" and flags.diagnosis:
            if event_id is None:
                raise MissingInputError("a diagnosis needs 'event_id' (or 'event <id>' in the message)")
            dreq = DiagnoseRequest(event_id=event_id, mode=req.mode)
            if flags.trust:
                d = self.decision_service.decide(DecisionRequest(kind="diagnosis", diagnose=dreq))
                out, trust = d.output, _trust(d)
            else:
                out = self.diagnosis_service.diagnose(dreq).model_dump(mode="json")
            text = (f"{out['diagnosis']} Likely cause: {out['likely_cause']['text']} "
                    f"Recommended action: {out['recommended_action']['text']}")
            return AssistantResponse(config=req.config, intent=intent, handled_by="fault_diagnosis", text=text,
                                     output=out, trust=trust, flags=flags.__dict__)
        message = req.message.strip()
        if not message:        # structured request whose module is disabled in this config: ask it as text
            message = (f"Diagnose fault event {event_id} of Wind Farm A." if event_id is not None else
                       f"Forecast the AC power of {req.asset} one hour after {req.origin}.")
        handled, text, out, trust = self._question(flags, message)
        return AssistantResponse(config=req.config, intent=intent, handled_by=handled, text=text, output=out,
                                 trust=trust, flags=flags.__dict__)


def _trust(d) -> Trust:
    return Trust(decision=d.decision, trust_score=d.trust_score, components=d.components, weights=d.weights,
                 reasons=d.reasons)
