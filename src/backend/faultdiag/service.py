"""B7 - Service behind POST /diagnose (fault signal -> evidence-grounded DiagnosisReport).

Production configuration (B3 + B6 fixes): hybrid BM25 + MiniLM retrieval with cross-encoder rerank over the
knowledge base's semantic chunks, sibling-chunk expansion of the retrieved evidence, the evidence-only fallback,
and the local LLM (Ollama, JSON mode, 120 s timeout with one retry). Retrieval models load lazily on first use.
Since C4 the retriever indexes the KB store (documents.jsonl) and sits behind document trust verification and
the evidence filter (src/backend/security/secure_pipeline.py): untrusted chunks never reach the LLM or the
fallback.

Scope: Wind Farm A (the only farm with B4 detector scores). A known event_id gets the B4 out-of-fold
logistic-regression score; an arbitrary window has no detector score (anomaly_score is null and the confidence
explanation says so).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .diagnosis import (
    B4_DIR,
    Diagnoser,
    DiagnosisReport,
    FaultSignal,
    event_windows,
    load_reliability,
    load_signals,
)

SUPPORTED_FARMS = ("A",)


class DiagnoseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"event_id": 73, "mode": "logged", "use_llm": True},
        {"asset_id": 0, "window_start": "2023-06-10T11:40", "window_end": "2023-06-17T11:40"}]})

    farm: Literal["A"] = Field("A", description="Wind farm (only A has detector scores; see docs/fault_diagnosis.md).")
    event_id: int | None = Field(None, description="Known event id from event_info.csv (uses the B4 detector score).")
    asset_id: int | None = Field(None, description="Turbine id, for an arbitrary window (no detector score).")
    window_start: datetime | None = None
    window_end: datetime | None = None
    mode: Literal["logged", "unlogged"] = Field(
        "logged", description="'unlogged' hides this event's own fault and maintenance documents (a new anomaly "
                              "with no log entry yet). Only meaningful with event_id.")
    use_llm: bool = Field(True, description="False returns the deterministic evidence-only report.")

    @model_validator(mode="after")
    def _one_target(self) -> "DiagnoseRequest":
        window = (self.asset_id, self.window_start, self.window_end)
        if self.event_id is not None and any(v is not None for v in window):
            raise ValueError("give either event_id or asset_id + window_start + window_end, not both")
        if self.event_id is None:
            if any(v is None for v in window):
                raise ValueError("give event_id, or asset_id with window_start and window_end")
            if self.window_end < self.window_start:
                raise ValueError("window_end must not be before window_start")
            if self.mode == "unlogged":
                raise ValueError("mode 'unlogged' needs an event_id")
        return self


class UnknownEventError(LookupError):
    pass


class ServiceUnavailableError(RuntimeError):
    pass


@dataclass
class DiagnosisService:
    """Holds the Diagnoser; tests inject a retriever/LLM, production builds them lazily."""

    diagnoser_llm: Diagnoser | None = None
    diagnoser_fallback: Diagnoser | None = None
    signals: dict[int, FaultSignal] | None = None
    components: dict[str, str | None] | None = None
    b4_dir: Path = B4_DIR
    security_status: dict | None = None         # C4: registry source and untrusted documents in the KB store
    retrieval_provider: Callable | None = None  # C7: shared SecureRetrieval (the API builds it once per process)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._call_lock = threading.Lock()      # the retriever's last_rejected is per call, single-threaded

    def _ensure(self) -> None:
        with self._lock:
            if self.diagnoser_llm is not None:
                return
            if not (self.b4_dir / "b4_event_predictions.csv").is_file():
                raise ServiceUnavailableError("B4 detector results not found. Run: python -m experiments.fault_diagnosis")
            from ..kb.taxonomy import load_events
            from ..retrieval.hybrid_retriever import SiblingExpander
            from ..security.secure_pipeline import build_secure_retrieval
            from .diagnosis import ollama_json_llm

            # C4: production retrieval over the KB store behind document trust + the evidence filter;
            # siblings are expanded only from trusted hits.
            sec = self.retrieval_provider() if self.retrieval_provider is not None else build_secure_retrieval()
            self.security_status = sec.status
            expander = SiblingExpander(sec.chunks)
            reliability, windows = load_reliability(self.b4_dir), event_windows(load_events("A"))
            self.diagnoser_llm = Diagnoser(sec.retriever, ollama_json_llm(), reliability, windows, expander=expander)
            self.diagnoser_fallback = Diagnoser(sec.retriever, None, reliability, windows, expander=expander)
            self.signals = {s.event_id: s for s in load_signals(self.b4_dir)}
            self.components = {d.doc_id: d.component for d in sec.store_docs}

    def diagnose(self, request: DiagnoseRequest) -> DiagnosisReport:
        return self.diagnose_with_security(request)[0]

    def diagnose_with_security(self, request: DiagnoseRequest) -> tuple[DiagnosisReport, list[str] | None]:
        """(report, doc ids removed by trust verification in this call; None if the retriever has no trust check)."""
        self._ensure()
        if request.event_id is not None:
            signal = self.signals.get(request.event_id)
            if signal is None:
                raise UnknownEventError(f"Unknown event_id {request.event_id} for Wind Farm {request.farm}; "
                                        f"known: {sorted(self.signals)}")
        else:
            signal = FaultSignal(farm=request.farm, asset_id=request.asset_id,
                                 window_start=pd.Timestamp(request.window_start),
                                 window_end=pd.Timestamp(request.window_end), anomaly_score=None)
        diagnoser = self.diagnoser_llm if request.use_llm else self.diagnoser_fallback
        with self._call_lock:
            report = diagnoser.diagnose(signal, self.components, exclude_own_event=(request.mode == "unlogged"))
            last = getattr(diagnoser.retriever, "last_rejected", None)
        rejected = None if last is None else sorted({h["metadata"]["doc_id"] for h in last})
        return report, rejected
