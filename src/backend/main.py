"""GreenRAG-Sec FastAPI application (local mode: FastAPI + ChromaDB + MiniLM/cross-encoder + Ollama).

Routers: system (health, readiness, warm-up, info), Track A forecasting, Track B RAG + fault diagnosis, Track C
security + trust, integration (orchestrator), results (measured numbers read from results/), legacy domain RAG.
The single-page console is served at /console/.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .runtime_env import configure_cache_dirs

# Load .env from the project root (two levels above src/backend).
# Expected keys are documented in .env.example — leave secrets out of git.
# Loaded before configure_cache_dirs so GREENRAG_CACHE_DIR in .env applies.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")
configure_cache_dirs()

from . import api_support  # noqa: E402
from .api import catalog, lab, ops, results  # noqa: E402
from .api.examples import ex  # noqa: E402
from .config import DomainManifest, list_domains, load_manifest  # noqa: E402
from .deps import (  # noqa: E402,F401 - getters re-exported: tests override main.get_*
    get_decision_service,
    get_diagnosis_service,
    get_forecast_service,
    get_orchestrator,
    get_secure_ask_service,
    get_secure_retrieval,
)
from .forecasting.schemas import ForecastRequest, ForecastResult  # noqa: E402
from .forecasting.service import ForecastService, ModelNotBuiltError, OriginError  # noqa: E402
from .faultdiag.diagnosis import DiagnosisReport  # noqa: E402
from .faultdiag.service import (  # noqa: E402
    DiagnoseRequest,
    DiagnosisService,
    ServiceUnavailableError,
    UnknownEventError,
)
from .orchestrator import AssistantRequest, AssistantResponse, MissingInputError, Orchestrator  # noqa: E402
from .retrieval import get_vector_store  # noqa: E402
from .security.secure_pipeline import SecureAnswer, SecureAskService  # noqa: E402
from .serving import get_generator  # noqa: E402
from .serving.router import (  # noqa: E402
    build_deterministic_numeric_answer,
    build_metric_payload,
    classify_query,
    format_ask_sources,
)
from .settings import get_settings  # noqa: E402
from .trust.service import DecisionRequest, DecisionResponse, DecisionService  # noqa: E402

logging.basicConfig(level=getattr(logging, get_settings().log_level.upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_ROOT = _PROJECT_ROOT / "data" / "processed"
CONSOLE_DIR = _PROJECT_ROOT / "src" / "console"
TAG_A, TAG_B, TAG_C = catalog.TAG_A, catalog.TAG_B, lab.TAG_C

DESCRIPTION = """
GreenRAG-Sec: a local, security-aware assistant for renewable-energy operations
(1-hour-ahead forecasting with calibrated intervals, knowledge-grounded fault diagnosis,
a secure retrieval pipeline and a trust-aware decision engine). **Mode: local** (FastAPI + ChromaDB + Ollama);
no cloud service is used.

**What "confidence" means here (read this first):**

- `/forecast` `confidence` (0.80 / 0.90 / 0.95) is the **nominal coverage of the prediction interval**: over many
  forecasts the interval method aims to contain the observed value that often. It is *not* the probability that the
  point forecast is right. Measured coverage is in `GET /results/summary/forecasting`.
- `/diagnose` `confidence.level` (low / moderate; never high) combines the fault detector's **measured reliability**
  (event AUC with its CI) and how strongly retrieved evidence supports the cause. It is *not* a probability that the
  diagnosis is correct.
- `/decision` `trust_score` T = 0.35 E + 0.15 R + 0.30 C + 0.20 S (evidence, retrieval relevance, model confidence,
  security) is a **ranking score for accept / review / reject**, not a probability. Only "accept" may be acted on
  without review; nothing is ever executed by the API.

Errors always have the body `{code, message, hint, detail, request_id}`; every response has an `X-Request-ID`
header. Cold start is about 100 s: call `POST /warmup` and watch `GET /ready`.
"""

TAGS = [
    {"name": "System", "description": "Liveness, readiness, warm-up, mode and versions."},
    {"name": TAG_A, "description": "Minmini K (23BIT0198): 1-hour-ahead AC power for two solar plants and one wind "
                                   "turbine with CQR prediction intervals."},
    {"name": TAG_B, "description": "Tarunica S (23BIT0185): knowledge base, hybrid retrieval with rerank, "
                                   "evidence-grounded Wind Farm A fault diagnosis."},
    {"name": TAG_C, "description": "Pritika Kasinathan (23BIT0232): secure retrieval pipeline, trust-aware decision "
                                   "engine and the security lab (synthetic attacks against the live detectors)."},
    {"name": "Integration", "description": "Orchestrator: routes a message by intent under ablation configs A-F."},
    {"name": "Results", "description": "Measured numbers read from results/ (never typed in), with caveats."},
    {"name": "Legacy domain RAG", "description": "Original per-domain retrieval and ask endpoints (unchanged)."},
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for line in ops.startup_banner():
        logger.info(line)
    try:                                   # GREENRAG_APPINSIGHTS=true: request log -> Application Insights
        from .cloud import appinsights

        if appinsights.install() is not None:
            logger.info("Application Insights request telemetry: on")
    except Exception as exc:  # noqa: BLE001 - telemetry must never stop the API; the message holds no secret
        logger.warning("Application Insights telemetry not started (%s)", type(exc).__name__)
    yield
    from .cloud import appinsights

    appinsights.uninstall()


app = FastAPI(
    title="GreenRAG-Sec",
    description=DESCRIPTION,
    version="1.0.0",
    openapi_tags=TAGS,
    lifespan=lifespan,
)

# CORS: only the console origins from settings (GREENRAG_CONSOLE_ORIGINS); the console at /console/ is same-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)
api_support.install(app)
# Routers' routes are added to the app itself (not wrapped), so every entry of app.routes has a .path.
# Re-registered on the app (not wrapped by include_router): every app.routes entry keeps a .path, and the handlers
# are built against the app so app.dependency_overrides apply (tests rely on both).
for _router in (ops.router, results.router, catalog.router, lab.router):
    for _r in _router.routes:
        app.add_api_route(_r.path, _r.endpoint, response_model=_r.response_model, status_code=_r.status_code,
                          tags=_r.tags, summary=_r.summary, description=_r.description, responses=_r.responses,
                          methods=list(_r.methods), name=_r.name, response_class=_r.response_class,
                          include_in_schema=_r.include_in_schema)
if CONSOLE_DIR.is_dir():
    app.mount("/console", StaticFiles(directory=CONSOLE_DIR, html=True), name="console")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/console/")


def _load_chunks_jsonl(domain_id: str) -> list[dict]:
    """Load ingestion output from ``data/processed/{domain_id}/chunks.jsonl``."""
    path = PROCESSED_ROOT / domain_id / "chunks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"No processed chunks for domain '{domain_id}'. "
            f"Expected file: {path}. "
            f"Run: python -m src.backend.ingestion.pipeline --domain {domain_id}"
        )

    chunks: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc
    return chunks


@app.get("/health", tags=["System"], responses=ex("GET /health"))
def health() -> dict[str, str]:
    """Liveness check for local and cloud probes."""
    return {"status": "ok"}


@app.get("/system/status", tags=["System"], responses=ex("GET /system/status"))
def system_status() -> dict[str, str]:
    """Lightweight console status for the frontend header pill."""
    mode = (os.getenv("BACKEND_MODE") or "local").strip().lower()
    if mode not in {"local", "azure"}:
        mode = "local"

    if mode == "azure":
        vector_store = "Azure AI Search"
        if get_settings().generator == "ollama":          # plan A: the LLM stays local
            generator_model = (os.getenv("OLLAMA_MODEL") or "").strip() or "qwen2.5:7b-instruct"
        else:
            generator_model = (
                (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip() or "azure-deployment"
            )
    else:
        vector_store = "ChromaDB + sentence-transformers"
        generator_model = (
            (os.getenv("OLLAMA_MODEL") or "").strip() or "qwen2.5:7b-instruct"
        )

    return {
        "backend_mode": mode,
        "vector_store": vector_store,
        "generator_model": generator_model,
    }


@app.get("/domains", tags=["Legacy domain RAG"], responses=ex("GET /domains"))
def get_domains() -> list[str]:
    """List available domain_ids from the domain_manifests folder."""
    return list_domains()


@app.get("/domains/{domain_id}", response_model=DomainManifest, tags=["Legacy domain RAG"], responses=ex("GET /domains/{domain_id}"))
def get_domain(domain_id: str) -> DomainManifest:
    """Return the validated manifest for a single domain."""
    try:
        return load_manifest(domain_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain_id}' was not found. "
            f"Available domains: {list_domains()}",
        ) from exc
    except ValueError as exc:
        # Manifest exists but is malformed — surface a clear client error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest/{domain_id}", tags=["Legacy domain RAG"])
def ingest_domain(domain_id: str) -> dict:
    """Embed + store processed JSONL chunks for ``domain_id`` via the active backend."""
    try:
        load_manifest(domain_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain_id}' was not found.",
        ) from exc

    try:
        chunks = _load_chunks_jsonl(domain_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        store = get_vector_store()
        store.embed_and_store(domain_id, chunks)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingest failed for domain '%s'", domain_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "domain_id": domain_id,
        "chunks_stored": len(chunks),
        "message": f"Embedded and stored {len(chunks)} chunk(s) for '{domain_id}'.",
    }


@app.get("/query/{domain_id}", tags=["Legacy domain RAG"], responses=ex("GET /query/{domain_id}"))
def query_domain(
    domain_id: str,
    q: str = Query(..., min_length=1, description="Natural-language search query"),
    top_k: int = Query(5, ge=1, le=50),
) -> dict:
    """Retrieve the top_k chunks most similar to ``q`` for ``domain_id``."""
    try:
        load_manifest(domain_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain_id}' was not found.",
        ) from exc

    try:
        store = get_vector_store()
        results = store.search(domain_id, q, top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed for domain '%s'", domain_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "domain_id": domain_id,
        "query": q,
        "top_k": top_k,
        "results": results,
    }


@app.get("/query/{domain_id}/top", tags=["Legacy domain RAG"], responses=ex("GET /query/{domain_id}/top"))
def query_domain_top(
    domain_id: str,
    metric: str = Query(
        ...,
        description="Numeric metadata field, e.g. avg_wind_speed or avg_ac_power",
    ),
    top_k: int = Query(3, ge=1, le=50),
    order: str = Query("desc", description="Sort order: asc or desc"),
) -> dict:
    """Rank chunks by a numeric metadata metric (highest/lowest lookup)."""
    try:
        load_manifest(domain_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain_id}' was not found.",
        ) from exc

    try:
        store = get_vector_store()
        results = store.search_by_metric(
            domain_id,
            metric_field=metric,
            top_k=top_k,
            order=order.lower(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Metric query failed for domain '%s'", domain_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "domain_id": domain_id,
        "metric": metric,
        "order": order.lower(),
        "top_k": top_k,
        "results": results,
    }


@app.get("/ask/{domain_id}", tags=["Legacy domain RAG"], responses=ex("GET /ask/{domain_id}"))
def ask_domain(
    domain_id: str,
    q: str = Query(..., min_length=1, description="Natural-language question"),
    top_k: int = Query(5, ge=1, le=50),
) -> dict:
    """Retrieve context and answer.

    Numeric superlative questions use ``search_by_metric`` plus a
    deterministic Python fact sentence (optional LLM enrichment).
    Semantic questions still use full LLM generation.
    """
    try:
        manifest = load_manifest(domain_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain_id}' was not found.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    routing = classify_query(q, manifest)
    retrieval_mode = "numeric" if routing is not None else "semantic"

    try:
        store = get_vector_store()
        if routing is not None:
            metric_field, order = routing
            chunks = store.search_by_metric(
                domain_id,
                metric_field=metric_field,
                top_k=top_k,
                order=order,
            )
        else:
            metric_field, order = "", ""
            chunks = store.search(domain_id, q, top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ask retrieval failed for domain '%s'", domain_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not chunks:
        return {
            "domain_id": domain_id,
            "query": q,
            "answer": "No relevant data found for this domain.",
            "sources": [],
            "retrieval_mode": retrieval_mode,
            "metric": None,
        }

    sources = format_ask_sources(chunks)

    if retrieval_mode == "numeric":
        try:
            deterministic = build_deterministic_numeric_answer(
                metric_field, order, chunks[0]
            )
            metric = build_metric_payload(metric_field, chunks[0])
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        answer = deterministic
        try:
            enrichment = get_generator().enrich_numeric_fact(deterministic, chunks)
            if enrichment:
                answer = f"{deterministic} {enrichment}"
        except Exception as exc:  # noqa: BLE001
            # Never block the ground-truth number on LLM failure.
            logger.warning(
                "Numeric enrichment skipped for domain '%s': %s", domain_id, exc
            )

        return {
            "domain_id": domain_id,
            "query": q,
            "answer": answer,
            "sources": sources,
            "retrieval_mode": retrieval_mode,
            "metric": metric,
        }

    try:
        generator = get_generator()
        answer = generator.generate_answer(
            q, chunks, manifest, retrieval_mode=retrieval_mode
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ask generation failed for domain '%s'", domain_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "domain_id": domain_id,
        "query": q,
        "answer": answer,
        "sources": sources,
        "retrieval_mode": retrieval_mode,
        "metric": None,
    }


_HINT_BUILD = "Build the missing artifact with the command in 'message', then retry (GET /ready lists every check)."
_HINT_ORIGIN = "Use an origin on the asset's grid inside its record (GET /forecast/assets)."


@app.post("/forecast", response_model=ForecastResult, tags=[TAG_A],
          responses={**ex("POST /forecast"), 422: {"description": "Invalid request or origin outside the record"},
                     503: {"description": "Forecast model artifact not built"}})
def forecast(
    request: ForecastRequest,
    service: ForecastService = Depends(get_forecast_service),
) -> ForecastResult:
    """1-hour-ahead AC power forecast (solar plant or wind turbine) with a calibrated interval.

    ``confidence`` is the nominal coverage level of the interval, not the probability that the point
    prediction is correct (see docs/forecasting.md). ``in_sample`` is true when the target day was used to fit
    the deployed model (true for the whole record): judge accuracy by the rolling-origin results instead.
    """
    try:
        result = service.forecast(request.asset_id, pd.Timestamp(request.origin), request.confidence)
    except ModelNotBuiltError as exc:
        raise api_support.ApiError(503, str(exc), hint=_HINT_BUILD) from exc
    except OriginError as exc:
        raise api_support.ApiError(422, str(exc), hint=_HINT_ORIGIN) from exc
    return ForecastResult(**result)


@app.post("/diagnose", response_model=DiagnosisReport, tags=[TAG_B],
          responses={**ex("POST /diagnose"), 404: {"description": "Unknown event id"},
                     503: {"description": "B4 detector results missing"}})
def diagnose(
    request: DiagnoseRequest,
    service: DiagnosisService = Depends(get_diagnosis_service),
) -> DiagnosisReport:
    """Evidence-grounded fault diagnosis for a Wind Farm A event or window.

    ``confidence`` is the anomaly score's measured reliability plus evidence support, not a probability that
    the diagnosis is correct; ``likely_cause`` comes only from retrieved evidence (cited chunk ids). If the LLM
    times out or fails, the deterministic evidence-only report is returned (``generation = "fallback"``).
    """
    try:
        return service.diagnose(request)
    except UnknownEventError as exc:
        raise api_support.ApiError(404, str(exc), hint="List valid events with GET /diagnose/events.") from exc
    except ServiceUnavailableError as exc:
        raise api_support.ApiError(503, str(exc), hint=_HINT_BUILD) from exc


class SecureAskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"question": "What should be checked when the pitch system reports a battery fault?"}]})
    question: str = Field(..., description="Question about the knowledge base (wind-turbine faults, SCADA, solar).")


@app.post("/secure-ask", response_model=SecureAnswer, tags=[TAG_C], responses=ex("POST /secure-ask"))
def secure_ask(
    request: SecureAskRequest,
    service: SecureAskService = Depends(get_secure_ask_service),
) -> SecureAnswer:
    """Knowledge-base answer through the C4 pipeline: input checks -> injection detection (block) -> retrieval ->
    document trust -> evidence filter -> LLM. ``security`` reports what each stage did. A blocked question is a
    200 with ``route = "blocked"``, not an error. If the LLM fails, the answer quotes the top trusted evidence
    (``route = "extractive_fallback"``). /ask is unchanged."""
    return service.ask(request.question)


@app.post("/decision", response_model=DecisionResponse, tags=[TAG_C],
          responses={**ex("POST /decision"), 404: {"description": "Unknown event id"},
                     422: {"description": "Invalid request"}, 503: {"description": "Model or results missing"}})
def decision(
    request: DecisionRequest,
    service: DecisionService = Depends(get_decision_service),
) -> DecisionResponse:
    """Run one served pipeline (secure answer, diagnosis or forecast), then the C5 trust-aware decision engine.

    Returns accept / review / reject with the trust score T = wE*E + wR*R + wC*C + wS*S, its components, the
    constraint checks and the reasons. Nothing is executed: only "accept" may be acted on without review.
    The recommendation is always produced server-side; a client cannot submit its own.
    """
    try:
        return service.decide(request)
    except UnknownEventError as exc:
        raise api_support.ApiError(404, str(exc), hint="List valid events with GET /diagnose/events.") from exc
    except (ServiceUnavailableError, ModelNotBuiltError) as exc:
        raise api_support.ApiError(503, str(exc), hint=_HINT_BUILD) from exc
    except OriginError as exc:
        raise api_support.ApiError(422, str(exc), hint=_HINT_ORIGIN) from exc


@app.post("/assistant", response_model=AssistantResponse, tags=["Integration"],
          responses={**ex("POST /assistant"), 404: {"description": "Unknown event id"},
                     422: {"description": "Missing origin / event id or invalid request"}})
def assistant(
    request: AssistantRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> AssistantResponse:
    """I1: route a message by intent (question / forecast / diagnosis) under ablation config A-F.

    Config F (default) runs the secure pipeline and returns the C5 trust score and accept / review / reject;
    A-E are the ablation baselines and return ``trust = null``.
    """
    try:
        return orchestrator.handle(request)
    except (MissingInputError, ValidationError, OriginError) as exc:
        raise api_support.ApiError(422, str(exc), hint="A forecast needs 'origin' (and 'asset'); a diagnosis needs "
                                                       "'event_id'.") from exc
    except UnknownEventError as exc:
        raise api_support.ApiError(404, str(exc), hint="List valid events with GET /diagnose/events.") from exc
    except (ServiceUnavailableError, ModelNotBuiltError) as exc:
        raise api_support.ApiError(503, str(exc), hint=_HINT_BUILD) from exc
