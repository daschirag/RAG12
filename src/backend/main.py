"""GreenRAG FastAPI application.

Exposes health, domain-manifest, retrieval, and grounded ask endpoints.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .runtime_env import configure_cache_dirs

# Load .env from the project root (two levels above src/backend).
# Expected keys are documented in .env.example — leave secrets out of git.
# Loaded before configure_cache_dirs so GREENRAG_CACHE_DIR in .env applies.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")
configure_cache_dirs()

from .config import DomainManifest, list_domains, load_manifest  # noqa: E402
from .retrieval import get_vector_store  # noqa: E402
from .serving import get_generator  # noqa: E402
from .serving.router import (  # noqa: E402
    build_deterministic_numeric_answer,
    build_metric_payload,
    classify_query,
    format_ask_sources,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_ROOT = _PROJECT_ROOT / "data" / "processed"

app = FastAPI(
    title="GreenRAG",
    description="Secure multi-tenant RAG framework for renewable energy assistants.",
    version="0.1.0",
)

# Frontend is a separate static site (any localhost port); allow those origins.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check for local and cloud probes."""
    return {"status": "ok"}


@app.get("/system/status")
def system_status() -> dict[str, str]:
    """Lightweight console status for the frontend header pill."""
    mode = (os.getenv("BACKEND_MODE") or "local").strip().lower()
    if mode not in {"local", "azure"}:
        mode = "local"

    if mode == "azure":
        vector_store = "Azure AI Search"
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


@app.get("/domains")
def get_domains() -> list[str]:
    """List available domain_ids from the domain_manifests folder."""
    return list_domains()


@app.get("/domains/{domain_id}", response_model=DomainManifest)
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


@app.post("/ingest/{domain_id}")
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


@app.get("/query/{domain_id}")
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


@app.get("/query/{domain_id}/top")
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


@app.get("/ask/{domain_id}")
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
