"""Operations: GET /ready, POST|GET /warmup, GET /system/info.

/health (liveness) stays in main and is unchanged. /ready reports what a request needs: model artifacts,
result files the engine reads, the knowledge-base store, cached retrieval models, Ollama, and whether the lazy
services have been loaded (cold start is about 100 s, so the console calls POST /warmup when it opens).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .examples import ex
from ..settings import PROJECT_ROOT, REQUIRED_OLLAMA_MODELS, get_settings

router = APIRouter(tags=["System"])
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ASSETS = ("solar_plant1", "solar_plant2", "wind_t1")


# ------------------------------------------------------------------ warm-up

@dataclass
class WarmStep:
    name: str
    state: Literal["pending", "running", "done", "failed"] = "pending"
    seconds: float | None = None
    error: str | None = None


@dataclass
class WarmupManager:
    """Runs the named load steps once, in order, in a background thread. A failed step does not stop the others."""

    steps: dict[str, Callable[[], None]]
    status: dict[str, WarmStep] = field(default_factory=dict)
    started_at: float | None = None
    finished_at: float | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self.status = {n: WarmStep(n) for n in self.steps}

    @property
    def state(self) -> str:
        if self.started_at is None:
            return "cold"
        if self.finished_at is None:
            return "warming"
        return "warm" if all(s.state == "done" for s in self.status.values()) else "partial"

    def start(self, background: bool = True) -> bool:
        """Start once; returns False if already started."""
        with self._lock:
            if self.started_at is not None:
                return False
            self.started_at = time.time()
        if background:
            threading.Thread(target=self._run, name="greenrag-warmup", daemon=True).start()
        else:
            self._run()
        return True

    def _run(self) -> None:
        for name, fn in self.steps.items():
            st = self.status[name]
            st.state = "running"
            t0 = time.perf_counter()
            try:
                fn()
                st.state = "done"
            except Exception as exc:  # noqa: BLE001 - reported in /ready and /warmup
                st.state, st.error = "failed", f"{type(exc).__name__}: {str(exc)[:300]}"
            st.seconds = round(time.perf_counter() - t0, 1)
        self.finished_at = time.time()

    def report(self) -> dict:
        elapsed = None
        if self.started_at is not None:
            elapsed = round((self.finished_at or time.time()) - self.started_at, 1)
        return {"state": self.state, "elapsed_seconds": elapsed,
                "steps": [s.__dict__ for s in self.status.values()]}


def _ollama_preload() -> None:
    s = get_settings()
    r = httpx.post(f"{s.ollama_host}/api/generate", json={"model": s.ollama_model, "prompt": "", "keep_alive": "30m"},
                   timeout=300)
    r.raise_for_status()


def default_steps() -> dict[str, Callable[[], None]]:
    from .. import deps
    from ..trust.signals import load_signals

    def forecasting():
        fs = deps.get_forecast_service()
        for a in ASSETS:
            fs.asset_info(a)

    return {
        "forecast_models": forecasting,
        "secure_retrieval": lambda: deps.get_secure_retrieval(),
        "diagnosis": lambda: deps.get_diagnosis_service()._ensure(),
        "secure_ask": lambda: deps.get_secure_ask_service()._ensure(),
        "decision_engine": lambda: (load_signals(), deps.get_decision_service()._engine(True)),
        "llm_preload": _ollama_preload,
    }


_MANAGER: WarmupManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_warmup() -> WarmupManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = WarmupManager(default_steps())
        return _MANAGER


# ------------------------------------------------------------------ readiness checks

class Check(BaseModel):
    name: str
    ok: bool
    required: bool
    detail: str


def _hf_cached(model: str) -> bool:
    hub = Path(os.environ.get("HF_HOME", "")) / "hub"
    name = model if "/" in model else f"sentence-transformers/{model}"
    return (hub / ("models--" + name.replace("/", "--"))).is_dir()


def ollama_status(timeout: float = 2.0) -> tuple[bool, str, list[str]]:
    s = get_settings()
    try:
        r = httpx.get(f"{s.ollama_host}/api/tags", timeout=timeout)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama not reachable at {s.ollama_host} ({type(exc).__name__}); answers fall back to " \
                      "evidence-only text", []
    need = {s.ollama_model, *REQUIRED_OLLAMA_MODELS}
    missing = sorted(m for m in need if m not in names)
    if missing:
        return False, f"Ollama reachable but missing models: {', '.join(missing)} (ollama pull <model>)", names
    return True, f"Ollama reachable; {s.ollama_model} available", names


def readiness_checks() -> list[Check]:
    root = PROJECT_ROOT
    out = []
    missing = [a for a in ASSETS if not (root / "models" / "forecasting" / f"{a}.joblib").is_file()]
    out.append(Check(name="forecast_models", ok=not missing, required=True,
                     detail="3 forecast artifacts present" if not missing else
                     f"missing {missing}; run python -m experiments.build_forecast_model"))
    f = root / "results" / "faultdiag" / "b4_event_predictions.csv"
    out.append(Check(name="fault_detector_results", ok=f.is_file(), required=True,
                     detail="B4 detector scores present" if f.is_file() else
                     "missing; run python -m experiments.fault_diagnosis"))
    kb = root / "data" / "processed" / "knowledge_base" / "documents.jsonl"
    out.append(Check(name="kb_store", ok=kb.is_file(), required=True,
                     detail="knowledge-base store present" if kb.is_file() else
                     "missing; run python -m src.backend.kb.build"))
    cached = [m for m in (EMBEDDING_MODEL, RERANK_MODEL) if _hf_cached(m)]
    out.append(Check(name="retrieval_models_cached", ok=len(cached) == 2, required=True,
                     detail=f"cached: {cached}" if len(cached) == 2 else
                     f"not in the model cache: {sorted({EMBEDDING_MODEL, RERANK_MODEL} - set(cached))}"))
    # Files the decision engine's measured signals are computed from (loading them takes ~18 s, done by warm-up).
    sig = ["faultdiag/b4_event_predictions.csv", "rag/b6_per_answer.csv", "security/c3_injection.csv",
           "security/c3_trust_docs.csv", "forecasting/a5_coverage_ci.csv", "forecasting/wind/coverage_ci.csv",
           "security/c3_telemetry_days.csv"]
    gone = [s for s in sig if not (root / "results" / s).is_file()]
    out.append(Check(name="trust_signal_sources", ok=not gone, required=True,
                     detail=f"{len(sig)} result files for the engine's measured signals present" if not gone else
                     f"missing results/{gone}; run python -m experiments.run_all"))
    chroma = root / "chroma_db"
    out.append(Check(name="legacy_vector_index", ok=chroma.is_dir(), required=False,
                     detail="chroma_db present (used by /query and /ask)" if chroma.is_dir() else
                     "chroma_db missing; POST /ingest/{domain} builds it"))
    ok, detail, _ = ollama_status()
    out.append(Check(name="ollama", ok=ok, required=False, detail=detail))
    s = get_settings()
    if s.backend_mode == "azure":           # configuration only; no network call, no secret read
        need = {"AZURE_SEARCH_ENDPOINT": s.azure_search_endpoint, "AZURE_SEARCH_INDEX": s.azure_search_index}
        if s.kb_source == "blob" or s.appinsights_enabled:
            need["AZURE_STORAGE_ACCOUNT_URL" if s.kb_source == "blob" else "AZURE_KEY_VAULT_URL"] = (
                s.azure_storage_account_url if s.kb_source == "blob" else s.azure_key_vault_url)
        missing = [k for k, v in need.items() if not v]
        out.append(Check(name="azure_config", ok=not missing, required=True,
                         detail="azure settings present" if not missing else f"azure mode but unset: {missing}"))
    return out


class ReadyResponse(BaseModel):
    status: Literal["ready", "degraded", "cold", "warming", "not_ready"] = Field(
        description="not_ready: a required artifact is missing (HTTP 503). cold: artifacts present, models not loaded "
                    "yet (first request takes ~100 s; call POST /warmup). warming: loading. degraded: loaded but an "
                    "optional dependency (Ollama) is down or a warm-up step failed. ready: everything loaded.")
    mode: str
    checks: list[Check]
    warmup: dict


@router.get("/ready", response_model=ReadyResponse,
            responses={**ex("GET /ready"), 503: {"description": "A required artifact is missing"}})
def ready(warm: WarmupManager = Depends(get_warmup)):
    """Readiness: model artifacts, result files, knowledge-base store, retrieval model cache, Ollama and the
    warm-up state. Returns 503 only when a required artifact is missing."""
    checks = readiness_checks()
    w = warm.report()
    if any(c.required and not c.ok for c in checks):
        status = "not_ready"
    elif w["state"] in ("cold", "warming"):
        status = w["state"]
    elif w["state"] == "partial" or any(not c.ok for c in checks):
        status = "degraded"
    else:
        status = "ready"
    body = ReadyResponse(status=status, mode=get_settings().backend_mode, checks=checks, warmup=w)
    return JSONResponse(body.model_dump(), status_code=503 if status == "not_ready" else 200)


class WarmupResponse(BaseModel):
    started_now: bool
    state: str
    elapsed_seconds: float | None
    steps: list[dict]


@router.post("/warmup", response_model=WarmupResponse, status_code=202, responses=ex("POST /warmup", 202))
def warmup(warm: WarmupManager = Depends(get_warmup)) -> WarmupResponse:
    """Start loading every model in the background (forecast models, retrieval + reranker + KB index, diagnosis,
    secure pipeline, decision engine, and the Ollama generator). Idempotent: later calls report progress."""
    started = warm.start()
    return WarmupResponse(started_now=started, **warm.report())


@router.get("/warmup", response_model=WarmupResponse, responses=ex("GET /warmup"))
def warmup_status(warm: WarmupManager = Depends(get_warmup)) -> WarmupResponse:
    """Progress of the background warm-up (per step: state, seconds, error)."""
    return WarmupResponse(started_now=False, **warm.report())


class SystemInfo(BaseModel):
    mode: str = Field(description="local (default) or azure (plan A, authorised 2026-10-16).")
    generator_model: str
    ollama_host: str
    ollama_models: list[str]
    embedding_model: str
    rerank_model: str
    vector_store: str
    console_origins: list[str]
    llm_timeout_s: float
    python: str
    platform: str
    code_version: str
    packages: dict[str, str]
    cloud: dict[str, str | bool] = Field(default_factory=dict, description="Azure resources in use (host names only, "
                                                                              "never keys).")


@router.get("/system/info", response_model=SystemInfo, responses=ex("GET /system/info"))
def system_info() -> SystemInfo:
    """Mode, models and package versions (no secrets). The header status pill uses /system/status."""
    from ..evaluation.provenance import environment_info

    s = get_settings()
    env = environment_info()
    _, _, names = ollama_status()
    return SystemInfo(mode=s.backend_mode, generator_model=s.ollama_model, ollama_host=s.ollama_host,
                      ollama_models=names, embedding_model=EMBEDDING_MODEL, rerank_model=RERANK_MODEL,
                      vector_store="ChromaDB (legacy domains) + in-memory hybrid BM25/MiniLM index (knowledge base)",
                      console_origins=s.origins, llm_timeout_s=s.llm_timeout_s, python=env["python"],
                      platform=env["platform"], code_version=env["git_commit"], packages=env["packages"],
                      cloud=cloud_summary())


def _host(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).hostname or "" if url else ""


def cloud_summary() -> dict[str, str | bool]:
    """What BACKEND_MODE=azure would use: host names and provider choices only."""
    s = get_settings()
    return {"backend_mode": s.backend_mode, "embedding_provider": s.embeddings, "generator_provider": s.generator,
            "kb_source": s.kb_source, "search_host": _host(s.azure_search_endpoint), "search_index": s.azure_search_index,
            "storage_host": _host(s.azure_storage_account_url), "key_vault_host": _host(s.azure_key_vault_url),
            "appinsights_enabled": s.appinsights_enabled}


def startup_banner() -> list[str]:
    s = get_settings()
    return ["GreenRAG-Sec API",
            f"  mode            : {s.backend_mode}" + ("" if s.backend_mode == "azure" else
                                                         " (FastAPI + ChromaDB + Ollama; no cloud services)"),
            f"  generator       : {s.ollama_model} @ {s.ollama_host} (timeout {s.llm_timeout_s:.0f} s)",
            f"  retrieval       : {EMBEDDING_MODEL} + BM25 hybrid, rerank {RERANK_MODEL}",
            "  console         : /console/   docs: /docs   readiness: /ready (POST /warmup to preload)",
            f"  CORS origins    : {', '.join(s.origins)}"]
