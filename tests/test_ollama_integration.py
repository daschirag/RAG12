"""Real LLM path through /ask (needs `ollama serve` with the configured model)."""

from __future__ import annotations

import os

import pytest
import requests
from fastapi.testclient import TestClient

from src.backend import main
from src.backend.serving.generator import DEFAULT_OLLAMA_MODEL, LocalGenerator

pytestmark = pytest.mark.ollama

OLLAMA_URL = "http://127.0.0.1:11434/api/tags"


def _ollama_models() -> list[str] | None:
    try:
        resp = requests.get(OLLAMA_URL, timeout=3)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    return [m["name"] for m in resp.json().get("models", [])]


@pytest.fixture
def ollama_client(monkeypatch, processed_root, fake_store):
    models = _ollama_models()
    if models is None:
        pytest.skip("Ollama is not reachable at 127.0.0.1:11434 (run `ollama serve`)")
    model = (os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()
    if model not in models:
        pytest.skip(f"Ollama model {model!r} is not pulled (have: {models})")
    monkeypatch.setenv("BACKEND_MODE", "local")
    monkeypatch.setattr(main, "get_vector_store", lambda: fake_store)
    monkeypatch.setattr(main, "get_generator", lambda: LocalGenerator(model=model))
    return TestClient(main.app)


def test_ask_semantic_with_ollama(ollama_client):
    ollama_client.post("/ingest/solar")
    resp = ollama_client.get(
        "/ask/solar", params={"q": "What AC power did inverter B produce?", "top_k": 3}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieval_mode"] == "semantic"
    assert body["answer"].strip()
    assert "612" in body["answer"]
