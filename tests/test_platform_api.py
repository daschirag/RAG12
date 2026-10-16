"""Platform endpoints: /ready, /warmup, /system/info, error format, request ids, CORS, settings, OpenAPI docs,
the console mount, and /results/summary (values checked against the result files themselves)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.backend import main
from src.backend.api import ops, results
from src.backend.api.ops import Check, WarmupManager
from src.backend.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


@pytest.fixture
def client():
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


@pytest.fixture
def no_ollama(monkeypatch):
    monkeypatch.setattr(ops, "ollama_status", lambda timeout=2.0: (False, "Ollama not reachable (test)", []))


# ------------------------------------------------------------------ settings

def test_settings_from_environment(monkeypatch):
    monkeypatch.setenv("BACKEND_MODE", "bogus")
    monkeypatch.setenv("OLLAMA_MODEL", "")
    monkeypatch.setenv("GREENRAG_CONSOLE_ORIGINS", "http://a:1/, http://b:2")
    monkeypatch.setenv("GREENRAG_LLM_TIMEOUT_S", "30")
    s = Settings()
    assert s.backend_mode == "local" and s.ollama_model == "qwen2.5:7b-instruct"
    assert s.origins == ["http://a:1", "http://b:2"] and s.llm_timeout_s == 30
    monkeypatch.setenv("GREENRAG_LLM_TIMEOUT_S", "0")
    with pytest.raises(ValueError):
        Settings()


# ------------------------------------------------------------------ health / ready / warm-up / info

def test_health_unchanged(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_checks_and_cold_state(client, no_ollama):
    warm = WarmupManager({"a": lambda: None})
    main.app.dependency_overrides[ops.get_warmup] = lambda: warm
    r = client.get("/ready")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "cold" and body["mode"] == "local"
    names = {c["name"] for c in body["checks"]}
    assert {"forecast_models", "kb_store", "retrieval_models_cached", "trust_signal_sources", "ollama"} <= names
    warm.start(background=False)
    assert client.get("/ready").json()["status"] == "degraded"        # warm, but Ollama is down (optional)


def test_ready_is_503_when_a_required_artifact_is_missing(client, monkeypatch):
    monkeypatch.setattr(ops, "readiness_checks", lambda: [Check(name="kb_store", ok=False, required=True, detail="x")])
    main.app.dependency_overrides[ops.get_warmup] = lambda: WarmupManager({})
    r = client.get("/ready")
    assert r.status_code == 503 and r.json()["status"] == "not_ready"


def test_ready_is_ready_when_everything_passes(client, monkeypatch):
    monkeypatch.setattr(ops, "readiness_checks", lambda: [Check(name="x", ok=True, required=True, detail="")])
    warm = WarmupManager({"a": lambda: None})
    warm.start(background=False)
    main.app.dependency_overrides[ops.get_warmup] = lambda: warm
    assert client.get("/ready").json()["status"] == "ready"


def test_warmup_runs_once_in_the_background_and_reports_failures(client):
    calls = []

    def slow():
        time.sleep(0.2)
        calls.append("slow")

    def broken():
        raise RuntimeError("model file missing")

    warm = WarmupManager({"slow": slow, "broken": broken, "after": lambda: calls.append("after")})
    main.app.dependency_overrides[ops.get_warmup] = lambda: warm
    first = client.post("/warmup")
    assert first.status_code == 202 and first.json()["started_now"] is True and first.json()["state"] == "warming"
    assert client.post("/warmup").json()["started_now"] is False            # idempotent
    for _ in range(100):
        if warm.finished_at:
            break
        time.sleep(0.05)
    rep = client.get("/warmup").json()
    assert rep["state"] == "partial" and calls == ["slow", "after"]           # a failure does not stop later steps
    steps = {s["name"]: s for s in rep["steps"]}
    assert steps["broken"]["state"] == "failed" and "model file missing" in steps["broken"]["error"]


def test_system_info_has_versions_and_no_secrets(client, no_ollama, monkeypatch):
    monkeypatch.setenv("GREENRAG_TRUST_KEY", "super-secret-value-123")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-secret-456")
    r = client.get("/system/info")
    body = r.json()
    assert r.status_code == 200 and body["mode"] == "local" and body["packages"]["fastapi"]
    assert "super-secret-value-123" not in r.text and "azure-secret-456" not in r.text
    assert any("local" in line for line in ops.startup_banner())


# ------------------------------------------------------------------ error format, request ids, CORS

def test_error_body_is_uniform(client):
    r = client.get("/no/such/path")
    body = r.json()
    assert r.status_code == 404 and set(body) == {"code", "message", "hint", "detail", "request_id"}
    assert body["code"] == "not_found" and body["request_id"] == r.headers["x-request-id"]
    v = client.post("/forecast", json={"plant": 9, "origin": "nope"})
    assert v.status_code == 422 and v.json()["code"] == "invalid_request"
    assert isinstance(v.json()["detail"], list) and "plant" in v.json()["message"]
    assert all("input" not in e for e in v.json()["detail"])                    # inputs are not echoed back
    x = client.post("/assistant", json={"message": "q", "unknown_field": 1})   # strict models: extra fields refused
    assert x.status_code == 422


def test_unhandled_errors_become_500_with_request_id(monkeypatch):
    class Boom:
        def forecast(self, *a):
            raise ZeroDivisionError("bug")

    main.app.dependency_overrides[main.get_forecast_service] = lambda: Boom()
    try:
        r = TestClient(main.app, raise_server_exceptions=False).post(
            "/forecast", json={"asset": "solar_plant1", "origin": "2020-06-16T11:00"})
    finally:
        main.app.dependency_overrides.clear()
    assert r.status_code == 500 and r.json()["code"] == "internal_error" and "bug" not in r.json()["message"]


def test_request_id_is_echoed_when_safe(client):
    assert client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"
    assert client.get("/health", headers={"X-Request-ID": "bad id <>"}).headers["x-request-id"] != "bad id <>"


def test_cors_allows_only_the_console_origins(client):
    ok = client.get("/health", headers={"Origin": "http://127.0.0.1:5500"})
    assert ok.headers.get("access-control-allow-origin") == "http://127.0.0.1:5500"
    bad = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


# ------------------------------------------------------------------ OpenAPI and console

def test_openapi_has_track_tags_descriptions_and_recorded_examples(client):
    spec = client.get("/openapi.json").json()
    tags = [t["name"] for t in spec["tags"]]
    assert {"System", "Track A - Forecasting", "Track B - RAG and fault diagnosis", "Track C - Security and trust",
            "Integration", "Results"} <= set(tags)
    assert "nominal coverage" in spec["info"]["description"] and "not a probability" in spec["info"]["description"]
    examples = json.loads((ROOT / "src" / "backend" / "api" / "openapi_examples.json").read_text(encoding="utf-8"))
    for path, ops_ in spec["paths"].items():
        for method, op in ops_.items():
            assert op.get("tags"), (method, path)
            assert op.get("description") or op.get("summary"), (method, path)
            if (method, path) != ("post", "/ingest/{domain_id}"):       # ingest mutates the index: not recorded
                ok = [r for r in op["responses"].values() if "example" in r.get("content", {}).get("application/json", {})]
                assert ok and "Recorded from the running system" in ok[0]["description"], (method, path)
    assert examples["captured_at_utc"] and all(v["status"] in (200, 202) for v in examples["responses"].values())


def test_console_is_served_and_calls_existing_routes(client):
    page = client.get("/console/")
    assert page.status_code == 200 and "GreenRAG-Sec Console" in page.text
    assert client.get("/", follow_redirects=False).headers["location"] == "/console/"
    js = "".join(p.read_text(encoding="utf-8") for p in (ROOT / "src" / "console" / "js").rglob("*.js"))
    routes = {r.path for r in main.app.routes}
    for path in ("/assistant", "/decision", "/forecast/series", "/forecast/assets", "/diagnose/events", "/kb/chunks",
                 "/results/summary", "/security/lab/telemetry", "/security/lab/injection", "/security/lab/poisoning",
                 "/security/lab/catalogue", "/system/status", "/system/info", "/ready", "/warmup",
                 "/assistant/suggestions"):
        assert path in js and path in routes, path
    assert "azure" not in js.lower()                    # the console never claims a cloud deployment


# ------------------------------------------------------------------ results summary (checked against the files)

def test_results_summary_matches_the_result_files(client):
    r = client.get("/results/summary")
    assert r.status_code == 200
    body = r.json()
    assert [t["track"] for t in body["tracks"]] == ["forecasting", "rag-diagnosis", "security-trust", "cloud"]
    fc = next(t for t in body["tracks"] if t["track"] == "forecasting")
    wind = pd.read_csv(RES / "forecasting" / "wind" / "point_metrics.csv").groupby("model").mae.mean()
    got = {row["model"]: row["mae"] for row in fc["metrics"]["wind_rolling_origin"]}
    assert got == pytest.approx(wind.to_dict())
    assert f"{wind['persistence']:.1f}" in fc["caveats"][0]
    sec = next(t for t in body["tracks"] if t["track"] == "security-trust")
    de = json.loads((RES / "security" / "c6_decision_engine.json").read_text(encoding="utf-8"))
    held = de["accepted_answer_correctness"]["heldout_injection"]
    assert f"{held['accepted_that_obeyed']['k']} of those {held['accepted_that_obeyed']['n']}" in sec["caveats"][0]
    assert all(src.startswith(("results/", "data/")) for t in body["tracks"] for src in t["sources"])


def test_results_track_variants(client):
    r = client.get("/results/summary/rag-diagnosis", params={"include_markdown": True})
    body = r.json()
    assert r.status_code == 200 and body["markdown"] and "results/rag/b6_summary.md" in body["markdown"]
    b4 = pd.read_csv(RES / "faultdiag" / "b4_event_metrics.csv").groupby("model").auc.mean()
    got = {row["model"]: row["auc"] for row in body["metrics"]["fault_detectors"]}
    assert got == pytest.approx(b4.to_dict())
    assert client.get("/results/summary/nope").status_code == 422
    assert client.get("/results/summary/cloud").json()["metrics"]["status"] in ("PENDING", "READY")


def test_results_missing_file_is_503(client, monkeypatch, tmp_path):
    monkeypatch.setattr(results, "RES", tmp_path)
    r = client.get("/results/summary/security-trust")
    assert r.status_code == 503 and "missing" in r.json()["message"]
