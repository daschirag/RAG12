"""C7 - Service behind POST /decision: run a served pipeline, then the C5 trust-aware decision engine.

The request names *what to produce* (a knowledge answer, a diagnosis or a forecast), never a recommendation:
the recommendation and its security facts are always produced server-side (C4 secure pipeline, secure /diagnose,
/forecast), so a client cannot claim that its own text passed the security checks. The engine re-verifies every
evidence document against the trusted registry (defence in depth) before scoring.

Forecasts: the C3 telemetry detector decides per whole day and is not wired into the served /forecast path
(docs/security_evaluation.md), so a served forecast carries ``telemetry_checked = False``. The engine then gives
S = 0 and the decision is at best "review"; this is the documented behaviour, not a failure.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..faultdiag.service import DiagnoseRequest
from ..forecasting.schemas import ForecastRequest
from .decision import Decision, DecisionEngine, Recommendation, SecurityFacts, from_diagnosis, from_secure_answer


class DecisionRequest(BaseModel):
    # extra="forbid": e.g. a client-supplied "recommendation" is refused (422)
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"kind": "answer", "question": "What should be checked when the pitch system reports a battery fault?"},
        {"kind": "diagnosis", "diagnose": {"event_id": 73}},
        {"kind": "forecast", "forecast": {"asset": "solar_plant1", "origin": "2020-06-16T11:00"}}]})

    kind: Literal["answer", "diagnosis", "forecast"]
    question: str | None = Field(None, description="kind 'answer': the question (runs the C4 secure pipeline).")
    diagnose: DiagnoseRequest | None = Field(None, description="kind 'diagnosis': a POST /diagnose request body.")
    forecast: ForecastRequest | None = Field(None, description="kind 'forecast': a POST /forecast request body.")

    @model_validator(mode="after")
    def _payload_matches_kind(self) -> "DecisionRequest":
        given = {k for k in ("question", "diagnose", "forecast") if getattr(self, k) is not None}
        need = {"answer": "question", "diagnosis": "diagnose", "forecast": "forecast"}[self.kind]
        if given != {need}:
            raise ValueError(f"kind '{self.kind}' needs exactly the '{need}' field, got {sorted(given) or 'none'}")
        return self


class DecisionResponse(Decision):
    kind: str
    output: dict[str, Any] = Field(description="The underlying output (SecureAnswer, DiagnosisReport or forecast).")


@dataclass
class DecisionService:
    """Holds the three served pipelines and the engine; tests inject them, production builds them lazily."""

    ask_service: Any = None                    # SecureAskService
    diagnosis_service: Any = None              # DiagnosisService
    forecast_service: Any = None               # ForecastService
    engine: DecisionEngine | None = None
    trust_provider: Callable[[], tuple[Any, str]] | None = None   # () -> (DocumentTrust, registry source)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _plain_engine: DecisionEngine | None = field(default=None, repr=False)

    def _engine(self, needs_trust: bool = True) -> DecisionEngine:
        """Forecasts carry no evidence documents, so they never build the retrieval models for the registry."""
        with self._lock:
            if self.engine is not None:
                return self.engine
            if not needs_trust:
                if self._plain_engine is None:
                    self._plain_engine = DecisionEngine()
                return self._plain_engine
            trust, source = self.trust_provider() if self.trust_provider is not None else (None, "")
            self.engine = DecisionEngine(trust=trust, trust_source=source or "in-process build")
            return self.engine

    def recommend(self, req: DecisionRequest) -> tuple[Recommendation, dict]:
        if req.kind == "answer":
            ans = self.ask_service.ask(req.question)
            return from_secure_answer(ans), ans.model_dump(mode="json")
        if req.kind == "diagnosis":
            report, rejected = self.diagnosis_service.diagnose_with_security(req.diagnose)
            facts = SecurityFacts(trust_checked=rejected is not None, rejected_doc_ids=rejected or [],
                                  registry_source=str((self.diagnosis_service.security_status or {})
                                                      .get("registry_source", "")))
            return from_diagnosis(report, facts), report.model_dump(mode="json")
        from ..forecasting.models import PLANT_CAPACITY_KW
        from ..forecasting.wind import WIND_CAPACITY_KW

        f = req.forecast
        out = self.forecast_service.forecast(f.asset_id, pd.Timestamp(f.origin), f.confidence)
        rec = Recommendation(kind="forecast", route="computed", text="", forecast_asset=f.asset_id,
                             interval=(float(out["lower_bound"]), float(out["prediction"]), float(out["upper_bound"])),
                             capacity_kw=WIND_CAPACITY_KW if f.asset_id == "wind_t1" else PLANT_CAPACITY_KW,
                             security=SecurityFacts(telemetry_checked=False))
        return rec, {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in out.items()}

    def decide(self, req: DecisionRequest) -> DecisionResponse:
        rec, output = self.recommend(req)
        d = self._engine(needs_trust=req.kind != "forecast").evaluate(rec)
        return DecisionResponse(kind=req.kind, output=output, **d.model_dump())
