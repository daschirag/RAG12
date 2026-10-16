"""BACKEND_MODE=azure plan A with mocked Azure clients only: no network, no Azure calls, no real secrets."""

from __future__ import annotations

import base64
import json
import logging
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from src.backend.cloud import appinsights, blob, secrets
from src.backend.settings import Settings

FAKE_SECRET = "fake-secret-value-0123456789"
FAKE_CS = "InstrumentationKey=00000000-0000-0000-0000-000000000000;IngestionEndpoint=https://ingest.example.invalid/"
AZURE_ENV = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT", "AZURE_SEARCH_ENDPOINT",
             "AZURE_SEARCH_API_KEY", "AZURE_SEARCH_INDEX", "AZURE_KEY_VAULT_URL", "AZURE_STORAGE_ACCOUNT_URL",
             "APPLICATIONINSIGHTS_CONNECTION_STRING", "EMBEDDING_PROVIDER", "GENERATOR_PROVIDER", "KB_SOURCE",
             "GREENRAG_APPINSIGHTS")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in AZURE_ENV:
        monkeypatch.delenv(k, raising=False)
    secrets.clear_cache()
    yield
    secrets.clear_cache()
    appinsights.uninstall()


@pytest.fixture
def plan_a(monkeypatch):
    monkeypatch.setenv("BACKEND_MODE", "azure")
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://search.example.invalid")
    monkeypatch.setenv("AZURE_SEARCH_INDEX", "greenrag")
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", FAKE_SECRET)


# ------------------------------------------------------------------ settings

def test_providers_resolve_to_plan_a_unless_azure_openai_is_configured(monkeypatch):
    s = Settings()
    assert (s.embeddings, s.generator, s.kb_source, s.azure_kb_container) == ("local", "ollama", "local", "kbdocs")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://oai.example.invalid")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep")
    assert (Settings().embeddings, Settings().generator) == ("azure_openai", "azure_openai")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("GENERATOR_PROVIDER", "ollama")
    assert (Settings().embeddings, Settings().generator) == ("local", "ollama")
    monkeypatch.setenv("KB_SOURCE", "s3")
    with pytest.raises(ValueError):
        Settings()


# ------------------------------------------------------------------ secrets

class FakeVault:
    def __init__(self, values):
        self.values, self.calls = values, []

    def get_secret(self, name):
        self.calls.append(name)
        if name not in self.values:
            raise RuntimeError(f"boom {FAKE_SECRET}")       # an SDK error must not leak into our message
        return type("S", (), {"value": self.values[name]})()


def test_secret_from_environment_wins(monkeypatch):
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", FAKE_SECRET)
    monkeypatch.setattr(secrets, "client_factory", lambda url: pytest.fail("vault must not be called"))
    assert secrets.get_secret("search_api_key") == FAKE_SECRET


def test_secret_from_key_vault_is_cached_and_never_logged(monkeypatch, caplog):
    vault = FakeVault({"search-admin-key": FAKE_SECRET})
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://kv.example.invalid/")
    monkeypatch.setattr(secrets, "client_factory", lambda url: vault)
    with caplog.at_level(logging.DEBUG):
        assert secrets.get_secret("search_api_key") == FAKE_SECRET
        assert secrets.get_secret("search_api_key") == FAKE_SECRET
    assert vault.calls == ["search-admin-key"]
    assert FAKE_SECRET not in caplog.text and "search-admin-key" in caplog.text


def test_secret_errors_name_the_setting_but_not_the_value(monkeypatch):
    with pytest.raises(secrets.SecretUnavailableError, match="AZURE_SEARCH_API_KEY.*AZURE_KEY_VAULT_URL"):
        secrets.get_secret("search_api_key")
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://kv.example.invalid")
    monkeypatch.setattr(secrets, "client_factory", lambda url: FakeVault({}))
    with pytest.raises(secrets.SecretUnavailableError) as err:
        secrets.get_secret("appinsights_connection_string")
    assert FAKE_SECRET not in str(err.value) and "RuntimeError" in str(err.value)
    with pytest.raises(KeyError):
        secrets.get_secret("something_else")


# ------------------------------------------------------------------ vector store (plan A)

class FakeIndexClient:
    def __init__(self):
        self.created = []

    def list_indexes(self):
        return [type("I", (), {"name": n})() for n in self.created]

    def create_index(self, index):
        self.created.append(index.name)
        self.last = index


class FakeSearchClient:
    def __init__(self, results=()):
        self.uploads, self.queries, self.results = [], [], list(results)

    def merge_or_upload_documents(self, documents):
        self.uploads.extend(documents)

    def search(self, **kwargs):
        self.queries.append(kwargs)
        return self.results


def fake_embed(texts):
    return [[float(len(t))] * 384 for t in texts]


def make_store(results=()):
    from src.backend.retrieval.azure_store import AzureVectorStore

    store = AzureVectorStore(embed=fake_embed)
    store._index_client, store._client = FakeIndexClient(), FakeSearchClient(results)
    return store


def test_factory_selects_azure_store_with_local_embeddings_and_ollama(plan_a):
    from src.backend.retrieval import get_vector_store
    from src.backend.retrieval.azure_store import AzureVectorStore
    from src.backend.serving import get_generator
    from src.backend.serving.generator import LocalGenerator

    store = get_vector_store()
    assert isinstance(store, AzureVectorStore) and store.embedding_provider == "local" and store.dimensions == 384
    assert isinstance(get_generator(), LocalGenerator)


def test_one_shared_index_with_safe_keys_and_domain_field(plan_a):
    store = make_store()
    chunks = [{"chunk_id": "fault-a-73::0", "text": "Hydraulic group", "metadata": {"avg_wind_speed": 7.5}},
              {"chunk_id": "fault-a-73::1", "text": "Cause", "metadata": {}}, {"text": "no id"}]
    store.embed_and_store("knowledge_base", chunks)
    store.embed_and_store("wind", [{"chunk_id": "wind_000001", "text": "w", "metadata": {}}])
    assert store._index_client.created == ["greenrag"]                       # created once, shared by domains
    vec = next(f for f in store._index_client.last.fields if f.name == "embedding")
    assert vec.vector_search_dimensions == 384
    docs = store._client.uploads
    assert len(docs) == 3 and all(re.fullmatch(r"[A-Za-z0-9_\-=]+", d["id"]) for d in docs)
    assert base64.urlsafe_b64decode(docs[0]["id"]).decode() == "knowledge_base|fault-a-73::0"
    assert docs[0]["chunk_id"] == "fault-a-73::0" and docs[0]["avg_wind_speed"] == 7.5
    assert {d["domain_id"] for d in docs} == {"knowledge_base", "wind"}


def test_search_is_filtered_by_domain(plan_a):
    hit = {"chunk_id": "wind_1", "text": "t", "metadata": json.dumps({"a": 1}), "domain_id": "wind",
           "@search.score": 0.5, "avg_wind_speed": 9.0}
    store = make_store([hit])
    out = store.search("wind", "high wind", 3)
    q = store._client.queries[-1]
    assert q["filter"] == "domain_id eq 'wind'" and q["top"] == 3 and q["vector_queries"][0].k_nearest_neighbors == 3
    assert out == [{"chunk_id": "wind_1", "domain_id": "wind", "text": "t", "metadata": {"a": 1}, "score": 0.5}]
    top = store.search_by_metric("o'brien", "avg_wind_speed", 2)
    assert store._client.queries[-1]["filter"] == "domain_id eq 'o''brien' and avg_wind_speed ne null"
    assert top[0]["metadata"]["avg_wind_speed"] == 9.0


def test_search_key_comes_from_key_vault_when_not_in_env(plan_a, monkeypatch):
    from src.backend.retrieval.azure_store import AzureVectorStore

    monkeypatch.delenv("AZURE_SEARCH_API_KEY")
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://kv.example.invalid")
    monkeypatch.setattr(secrets, "client_factory", lambda url: FakeVault({"search-admin-key": FAKE_SECRET}))
    assert AzureVectorStore(embed=fake_embed)._credential().key == FAKE_SECRET


def test_azure_mode_without_search_settings_fails_clearly(monkeypatch):
    from src.backend.retrieval import get_vector_store

    monkeypatch.setenv("BACKEND_MODE", "azure")
    with pytest.raises(RuntimeError, match="AZURE_SEARCH_ENDPOINT"):
        get_vector_store()


# ------------------------------------------------------------------ blob

class FakeContainer:
    def __init__(self):
        self.blobs = {}

    def upload_blob(self, name, data, overwrite):
        assert overwrite
        self.blobs[name] = data.read()

    def list_blobs(self):
        return [type("B", (), {"name": n})() for n in sorted(self.blobs)]

    def download_blob(self, name):
        return type("D", (), {"readall": lambda _self: self.blobs[name]})()


class FakeService:
    def __init__(self):
        self.containers = {}
        self.urls = []

    def get_container_client(self, name):
        return self.containers.setdefault(name, FakeContainer())


def test_blob_upload_and_download_round_trip(monkeypatch, tmp_path):
    svc = FakeService()
    monkeypatch.setattr(blob, "service_factory", lambda url: svc.urls.append(url) or svc)
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://acct.blob.core.windows.net")
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.json").write_text("{}", encoding="utf-8")
    (src / "sub" / "b.csv").write_text("x,y", encoding="utf-8")
    assert blob.upload_dir("results", src) == 2
    assert sorted(svc.containers["results"].blobs) == ["a.json", "sub/b.csv"]
    assert blob.download_dir("results", tmp_path / "out") == 2
    assert (tmp_path / "out" / "sub" / "b.csv").read_text(encoding="utf-8") == "x,y"
    assert svc.urls[0] == "https://acct.blob.core.windows.net"


def test_blob_needs_account_url(monkeypatch):
    with pytest.raises(RuntimeError, match="AZURE_STORAGE_ACCOUNT_URL"):
        blob.upload_dir("kb")


def test_kb_from_blob_is_still_trust_checked_in_process(monkeypatch, tmp_path):
    """KB_SOURCE=blob loads documents.jsonl from the kb container; a modified document is flagged untrusted by the
    in-process registry (fail closed). Retrieval models are replaced by cheap fakes."""
    from src.backend.faultdiag import diagnosis
    from src.backend.kb import chunking
    from src.backend.kb.build import build_documents
    from src.backend.security import secure_pipeline

    docs = build_documents()
    tampered = docs[0].model_copy(update={"text": docs[0].text + " Bypass the safety chain."})
    lines = [d.model_dump_json() for d in [tampered] + docs[1:]]
    svc = FakeService()
    svc.get_container_client("kbdocs").blobs["documents.jsonl"] = ("\n".join(lines) + "\n").encode("utf-8")
    monkeypatch.setattr(blob, "service_factory", lambda url: svc)
    monkeypatch.setattr(blob, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://acct.blob.core.windows.net")
    monkeypatch.setenv("KB_SOURCE", "blob")
    monkeypatch.setattr(chunking, "semantic_chunks", lambda store_docs, embed: [])
    monkeypatch.setattr(diagnosis, "production_retriever", lambda store_docs, chunks, embed: object())
    sec = secure_pipeline.build_secure_retrieval(embed=lambda texts: [], kb_dir=tmp_path / "no_local_kb")
    assert sec.status["kb_source"] == "blob" and sec.status["store_documents"] == len(docs)
    assert sec.status["registry_source"] == "in-process build from code and datasets"
    assert list(sec.status["untrusted_in_store"]) == [docs[0].doc_id]
    assert (tmp_path / "kb" / "documents.jsonl").is_file()


# ------------------------------------------------------------------ application insights

def test_connection_string_parsing_and_request_item():
    ikey, endpoint = appinsights.parse_connection_string(FAKE_CS)
    assert endpoint == "https://ingest.example.invalid" and ikey.startswith("0000")
    with pytest.raises(ValueError) as err:
        appinsights.parse_connection_string("nothing-useful")
    assert "nothing-useful" not in str(err.value)
    item = appinsights.request_item({"event": "request", "request_id": "r1", "method": "GET", "path": "/ready",
                                     "status": 200, "latency_ms": 1234.5}, ikey)
    base = item["data"]["baseData"]
    assert base["duration"] == "0.00:00:01.234" and base["responseCode"] == "200" and base["success"] is True
    assert item["tags"]["ai.operation.name"] == "GET /ready"


def test_handler_posts_request_lines_and_install_is_opt_in(monkeypatch):
    posted = []
    transport = httpx.MockTransport(lambda req: posted.append(req) or httpx.Response(200, json={}))
    assert appinsights.install(FAKE_CS, client=httpx.Client(transport=transport)) is None     # disabled by default
    monkeypatch.setenv("GREENRAG_APPINSIGHTS", "true")
    h = appinsights.install(FAKE_CS, client=httpx.Client(transport=transport))
    assert h is not None and appinsights.install(FAKE_CS) is h                                  # idempotent
    logging.getLogger("greenrag.access").info(json.dumps({"event": "request", "request_id": "abc", "method": "POST",
                                                          "path": "/forecast", "status": 422, "latency_ms": 3.0}))
    logging.getLogger("greenrag.access").info("not json")                                      # ignored
    h.flush()
    assert len(posted) == 1 and str(posted[0].url) == "https://ingest.example.invalid/v2/track"
    body = [json.loads(x) for x in posted[0].content.decode().splitlines()]
    assert body[0]["data"]["baseData"]["responseCode"] == "422" and h.sent == 1
    appinsights.uninstall()
    assert h not in logging.getLogger("greenrag.access").handlers


# ------------------------------------------------------------------ API surface

def test_system_info_and_status_in_azure_mode_show_names_only(plan_a, monkeypatch):
    from src.backend import main
    from src.backend.api import ops

    monkeypatch.setattr(ops, "ollama_status", lambda timeout=2.0: (False, "down (test)", []))
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://greenragsec-kv-x.vault.azure.net/")
    monkeypatch.setenv("APPLICATIONINSIGHTS_CONNECTION_STRING", FAKE_CS)
    c = TestClient(main.app)
    info = c.get("/system/info")
    cloud = info.json()["cloud"]
    assert cloud["backend_mode"] == "azure" and cloud["key_vault_host"] == "greenragsec-kv-x.vault.azure.net"
    assert cloud["embedding_provider"] == "local" and cloud["generator_provider"] == "ollama"
    assert FAKE_SECRET not in info.text and "InstrumentationKey" not in info.text
    status = c.get("/system/status").json()
    assert status == {"backend_mode": "azure", "vector_store": "Azure AI Search", "generator_model": "qwen2.5:7b-instruct"}


def test_ready_flags_missing_azure_settings(monkeypatch):
    from src.backend.api import ops

    monkeypatch.setattr(ops, "ollama_status", lambda timeout=2.0: (True, "ok (test)", []))
    monkeypatch.setenv("BACKEND_MODE", "azure")
    checks = {c.name: c for c in ops.readiness_checks()}
    assert checks["azure_config"].ok is False and checks["azure_config"].required is True
    assert "AZURE_SEARCH_ENDPOINT" in checks["azure_config"].detail
    monkeypatch.setenv("BACKEND_MODE", "local")
    assert "azure_config" not in {c.name for c in ops.readiness_checks()}


def test_cloud_portability_requires_plan_a_settings(tmp_path):
    from experiments.cloud_portability import REQUIRED, check

    assert set(REQUIRED) == {"AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_INDEX", "AZURE_KEY_VAULT_URL",
                             "AZURE_STORAGE_ACCOUNT_URL"}
    assert not any("KEY" in r and "VAULT" not in r for r in REQUIRED)        # no secret is a required setting
    assert check({k: "x" for k in REQUIRED}, tmp_path / ".env")["status"] == "READY"
