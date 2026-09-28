"""Shared fixtures: TestClient with in-memory fakes for the vector store and LLM."""

from __future__ import annotations

import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.backend import main
from src.backend.retrieval.base import VectorStore
from src.backend.retrieval.metrics import flatten_chunk_metadata, validate_metric_request
from src.backend.serving.generator import Generator

SAMPLE_CHUNKS: list[dict[str, Any]] = [
    {
        "chunk_id": "solar_000000",
        "domain_id": "solar",
        "text": "Plant 1 inverter A produced an average AC power of 257.75 kW "
        "under 0.2037 irradiation with ambient 25.80 C.",
        "metadata": {"equipment_id": "A", "avg_ac_power": 257.75, "avg_irradiation": 0.2037},
    },
    {
        "chunk_id": "solar_000001",
        "domain_id": "solar",
        "text": "Plant 1 inverter B produced an average AC power of 612.10 kW "
        "under 0.5120 irradiation with ambient 31.20 C.",
        "metadata": {"equipment_id": "B", "avg_ac_power": 612.10, "avg_irradiation": 0.5120},
    },
    {
        "chunk_id": "solar_000002",
        "domain_id": "solar",
        "text": "Night-time window: inverter C reported zero AC power and zero irradiation.",
        "metadata": {"equipment_id": "C", "avg_ac_power": 0.0, "avg_irradiation": 0.0},
    },
]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeVectorStore(VectorStore):
    """In-memory store: keyword-overlap search, exact metric ranking."""

    def __init__(self) -> None:
        self.collections: dict[str, list[dict[str, Any]]] = {}

    def embed_and_store(self, domain_id: str, chunks: list[dict[str, Any]]) -> None:
        stored = self.collections.setdefault(domain_id, [])
        by_id = {c["chunk_id"]: i for i, c in enumerate(stored)}
        for chunk in chunks:
            record = {
                "chunk_id": chunk["chunk_id"],
                "domain_id": domain_id,
                "text": chunk["text"],
                "metadata": flatten_chunk_metadata(domain_id, chunk.get("metadata")),
            }
            if chunk["chunk_id"] in by_id:
                stored[by_id[chunk["chunk_id"]]] = record
            else:
                stored.append(record)

    def search(self, domain_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        terms = set(_tokens(query))
        scored = [
            {**c, "score": float(len(terms & set(_tokens(c["text"]))))}
            for c in self.collections.get(domain_id, [])
        ]
        scored.sort(key=lambda c: c["score"], reverse=True)
        return scored[:top_k]

    def search_by_metric(
        self, domain_id: str, metric_field: str, top_k: int, order: str = "desc"
    ) -> list[dict[str, Any]]:
        metric_field, order = validate_metric_request(metric_field, order)
        hits = [
            {**c, "score": float(c["metadata"][metric_field])}
            for c in self.collections.get(domain_id, [])
            if metric_field in c["metadata"]
        ]
        hits.sort(key=lambda c: c["score"], reverse=order == "desc")
        return hits[:top_k]


class FakeGenerator(Generator):
    """Deterministic generator that echoes the ids of the chunks it was given."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_answer(self, query, context_chunks, domain_manifest, retrieval_mode="semantic"):
        self.calls.append({"query": query, "mode": retrieval_mode})
        ids = ",".join(c["chunk_id"] for c in context_chunks)
        return f"FAKE ANSWER [{ids}]"

    def enrich_numeric_fact(self, deterministic_answer, context_chunks):
        return ""


@pytest.fixture
def fake_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def fake_generator() -> FakeGenerator:
    return FakeGenerator()


@pytest.fixture
def processed_root(tmp_path, monkeypatch):
    """Temp data/processed tree containing a small solar chunks.jsonl."""
    import json

    solar_dir = tmp_path / "solar"
    solar_dir.mkdir()
    with (solar_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in SAMPLE_CHUNKS:
            handle.write(json.dumps(chunk) + "\n")
    monkeypatch.setattr(main, "PROCESSED_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def client(monkeypatch, fake_store, fake_generator, processed_root) -> TestClient:
    """TestClient with fakes injected; BACKEND_MODE forced to local."""
    monkeypatch.setenv("BACKEND_MODE", "local")
    monkeypatch.setattr(main, "get_vector_store", lambda: fake_store)
    monkeypatch.setattr(main, "get_generator", lambda: fake_generator)
    return TestClient(main.app)
