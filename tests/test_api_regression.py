"""Regression tests for the pre-existing GreenRAG endpoints (fakes injected)."""

from __future__ import annotations

from conftest import SAMPLE_CHUNKS


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_system_status_local(client):
    body = client.get("/system/status").json()
    assert body["backend_mode"] == "local"
    assert body["vector_store"] == "ChromaDB + sentence-transformers"
    assert body["generator_model"]


def test_system_status_azure(client, monkeypatch):
    monkeypatch.setenv("BACKEND_MODE", "azure")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-test")
    body = client.get("/system/status").json()
    assert body == {
        "backend_mode": "azure",
        "vector_store": "Azure AI Search",
        "generator_model": "gpt-test",
    }


def test_system_status_unknown_mode_falls_back_to_local(client, monkeypatch):
    monkeypatch.setenv("BACKEND_MODE", "bogus")
    assert client.get("/system/status").json()["backend_mode"] == "local"


def test_domains_list(client):
    resp = client.get("/domains")
    assert resp.status_code == 200
    domains = resp.json()
    assert {"solar", "wind", "fault_diagnosis"} <= set(domains)
    assert domains == sorted(domains)


def test_domain_manifest_valid(client):
    body = client.get("/domains/solar").json()
    assert body["domain_id"] == "solar"
    assert "{context}" in body["prompt_template"]
    assert "{question}" in body["prompt_template"]


def test_domain_manifest_unknown_404(client):
    assert client.get("/domains/does_not_exist").status_code == 404


def test_ingest_stores_chunks(client, fake_store):
    resp = client.post("/ingest/solar")
    assert resp.status_code == 200
    assert resp.json()["chunks_stored"] == len(SAMPLE_CHUNKS)
    assert len(fake_store.collections["solar"]) == len(SAMPLE_CHUNKS)


def test_ingest_is_idempotent(client, fake_store):
    client.post("/ingest/solar")
    client.post("/ingest/solar")
    assert len(fake_store.collections["solar"]) == len(SAMPLE_CHUNKS)


def test_ingest_unknown_domain_404(client):
    assert client.post("/ingest/does_not_exist").status_code == 404


def test_ingest_missing_chunks_file_404(client):
    # wind has a manifest but the temp processed root only holds solar.
    resp = client.post("/ingest/wind")
    assert resp.status_code == 404
    assert "chunks.jsonl" in resp.json()["detail"]


def test_query_semantic(client):
    client.post("/ingest/solar")
    resp = client.get("/query/solar", params={"q": "night zero irradiation", "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["top_k"] == 2
    assert len(body["results"]) == 2
    assert body["results"][0]["chunk_id"] == "solar_000002"


def test_query_validation(client):
    assert client.get("/query/solar", params={"q": ""}).status_code == 422
    assert client.get("/query/solar", params={"q": "x", "top_k": 0}).status_code == 422
    assert client.get("/query/nope", params={"q": "x"}).status_code == 404


def test_query_top_metric(client):
    client.post("/ingest/solar")
    resp = client.get("/query/solar/top", params={"metric": "avg_ac_power", "top_k": 1})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["chunk_id"] == "solar_000001"
    asc = client.get(
        "/query/solar/top", params={"metric": "avg_ac_power", "top_k": 1, "order": "asc"}
    ).json()
    assert asc["results"][0]["chunk_id"] == "solar_000002"


def test_query_top_bad_metric_400(client):
    resp = client.get("/query/solar/top", params={"metric": "not_a_metric"})
    assert resp.status_code == 400


def test_ask_numeric_is_deterministic(client, fake_generator):
    client.post("/ingest/solar")
    resp = client.get("/ask/solar", params={"q": "Which inverter had the highest AC power?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieval_mode"] == "numeric"
    assert body["answer"].startswith("The highest AC power (kW) recorded was 612.10 (B,")
    assert body["metric"] == {"field": "avg_ac_power", "value": 612.1, "max": 4000.0}
    assert body["sources"][0]["chunk_id"] == "solar_000001"
    # The numeric path must not call full LLM generation.
    assert fake_generator.calls == []


def test_ask_semantic_uses_generator(client, fake_generator):
    client.post("/ingest/solar")
    resp = client.get("/ask/solar", params={"q": "what happens at night", "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieval_mode"] == "semantic"
    assert body["metric"] is None
    assert body["answer"].startswith("FAKE ANSWER [")
    assert fake_generator.calls == [{"query": "what happens at night", "mode": "semantic"}]
    assert {s["chunk_id"] for s in body["sources"]} <= {c["chunk_id"] for c in SAMPLE_CHUNKS}


def test_ask_empty_collection(client):
    body = client.get("/ask/solar", params={"q": "anything"}).json()
    assert body["answer"] == "No relevant data found for this domain."
    assert body["sources"] == []


def test_ask_unknown_domain_404(client):
    assert client.get("/ask/nope", params={"q": "x"}).status_code == 404


def test_ask_generator_failure_returns_500(client, monkeypatch):
    from src.backend import main

    def boom():
        raise RuntimeError("LLM backend unavailable")

    client.post("/ingest/solar")
    monkeypatch.setattr(main, "get_generator", boom)
    resp = client.get("/ask/solar", params={"q": "what happens at night"})
    assert resp.status_code == 500
    assert "LLM backend unavailable" in resp.json()["detail"]
