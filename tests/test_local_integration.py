"""End-to-end API tests on the real local backend (MiniLM + ChromaDB, temp dir).

The generator is still faked for semantic answers; the real Ollama path is
covered in test_ollama_integration.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.backend import main
from src.backend.retrieval.local_store import LocalVectorStore

pytestmark = pytest.mark.integration


@pytest.fixture
def local_client(monkeypatch, tmp_path, processed_root, fake_generator):
    monkeypatch.setenv("BACKEND_MODE", "local")
    store = LocalVectorStore(persist_dir=tmp_path / "chroma")
    monkeypatch.setattr(main, "get_vector_store", lambda: store)
    monkeypatch.setattr(main, "get_generator", lambda: fake_generator)
    return TestClient(main.app)


def test_local_ingest_query_ask(local_client):
    assert local_client.post("/ingest/solar").json()["chunks_stored"] == 3
    # Re-ingest exercises the update path without duplicating ids.
    assert local_client.post("/ingest/solar").status_code == 200

    hits = local_client.get(
        "/query/solar", params={"q": "night time with no sunlight", "top_k": 3}
    ).json()["results"]
    assert len(hits) == 3
    assert hits[0]["chunk_id"] == "solar_000002"
    assert all(-1.0 <= h["score"] <= 1.0 for h in hits)

    top = local_client.get(
        "/query/solar/top", params={"metric": "avg_ac_power", "top_k": 1}
    ).json()["results"]
    assert top[0]["chunk_id"] == "solar_000001"
    assert top[0]["score"] == pytest.approx(612.10)

    body = local_client.get(
        "/ask/solar", params={"q": "What was the lowest irradiation?"}
    ).json()
    assert body["retrieval_mode"] == "numeric"
    assert body["answer"].startswith("The lowest irradiation recorded was 0.0000 (C,")
