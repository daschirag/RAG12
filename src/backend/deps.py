"""Process-wide services (lazy). ``main`` and the routers share these function objects, so a test's
``app.dependency_overrides[main.get_x]`` override applies everywhere."""

from __future__ import annotations

from functools import lru_cache

from .faultdiag.service import DiagnosisService
from .forecasting.service import ForecastService
from .orchestrator import Orchestrator
from .security.secure_pipeline import SecureAskService, build_secure_retrieval
from .settings import get_settings
from .trust.service import DecisionService


@lru_cache(maxsize=1)
def get_forecast_service() -> ForecastService:
    """Process-wide forecasting service (models and features load lazily on first use)."""
    return ForecastService()


@lru_cache(maxsize=1)
def get_secure_retrieval():
    """Process-wide C4 secure retrieval (KB store index + document trust + evidence filter), built on first use and
    shared by /diagnose, /secure-ask and /decision."""
    return build_secure_retrieval()


@lru_cache(maxsize=1)
def get_diagnosis_service() -> DiagnosisService:
    """Process-wide diagnosis service (retrieval models and knowledge base load on first use)."""
    return DiagnosisService(retrieval_provider=get_secure_retrieval)


@lru_cache(maxsize=1)
def get_secure_ask_service() -> SecureAskService:
    """Process-wide C4 secure pipeline (shares the secure retrieval with /diagnose)."""
    return SecureAskService(retrieval_provider=get_secure_retrieval, model=get_settings().ollama_model)


@lru_cache(maxsize=1)
def get_decision_service() -> DecisionService:
    """Process-wide decision service over the served pipelines (all lazy)."""

    def trust():
        sec = get_secure_retrieval()
        return sec.trust, sec.status["registry_source"]

    return DecisionService(ask_service=get_secure_ask_service(), diagnosis_service=get_diagnosis_service(),
                           forecast_service=get_forecast_service(), trust_provider=trust)


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    """Process-wide I1 orchestrator over the served modules (all lazy)."""
    return Orchestrator(forecast_service=get_forecast_service(), diagnosis_service=get_diagnosis_service(),
                        decision_service=get_decision_service())
