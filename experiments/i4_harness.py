"""I4 - local vs Azure (plan A): same requests through the app in both modes; only measured values are written.

    python -m experiments.i4_harness --part ingest              # Azure: 4 domains -> shared index; local: KB -> Chroma
    python -m experiments.i4_harness --part run --mode local    # one heavy job at a time
    python -m experiments.i4_harness --part run --mode azure    # loads .env.azure (non-secret); secrets from Key Vault
    python -m experiments.i4_harness --report

Reduced item set (user request, 2026-10-16). In-process TestClient, so the full middleware runs; latency is the
client-side wall time per request.
- /query/knowledge_base: all 64 labelled B2 queries, top 5, 3 repetitions (no LLM).
- /forecast: 50 consecutive solar_plant1 origins from 2020-06-16 06:00, 3 repetitions (no LLM).
- /ask/knowledge_base: the first 16 labelled queries, top 5, one run (LLM). The legacy /ask generator calls Ollama
  with default sampling (no temperature or seed set), unlike /secure-ask, /decision and /diagnose (0 / 0).
- /secure-ask and /decision (kind answer): 10 questions, one run: the first 5 labelled queries (clean) and the
  next 5 with the C2 query-injection template appended (attack).
- /diagnose: the first 5 Wind Farm A events by id, LLM on, one run.
Warm-up (POST /warmup until /ready is not warming) is timed separately and excluded from request latency.
The LLM is reloaded every 30 LLM calls; the run stops if free RAM < 500 MB (resumable: done items are skipped).
What differs between modes: the vector store for /query and /ask (Chroma vs AI Search, same MiniLM vectors), the
knowledge-base source for the secure pipeline (local file vs Blob), secrets (Key Vault) and telemetry (App Insights).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cloud"
AZ_BIN = r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin"
DOMAINS = ("solar", "wind", "fault_diagnosis", "knowledge_base")
REPS_CHEAP = 3
BATCH = 30
MIN_FREE_MB = 500


def raw_path(mode: str) -> Path:
    return OUT / f"i4_raw_{mode}.jsonl"


def setup_env(mode: str) -> None:
    """Environment for one mode; must run before the app is imported."""
    if AZ_BIN not in os.environ.get("PATH", ""):
        os.environ["PATH"] = AZ_BIN + os.pathsep + os.environ.get("PATH", "")      # AzureCliCredential needs az
    if mode == "azure":
        from dotenv import dotenv_values

        for k, v in dotenv_values(ROOT / ".env.azure").items():
            if v is not None:
                os.environ[k] = v
    else:
        os.environ["BACKEND_MODE"] = "local"
        os.environ["KB_SOURCE"] = "local"
        os.environ["GREENRAG_APPINSIGHTS"] = "false"


def _chunks(domain: str) -> list[dict]:
    path = ROOT / "data" / "processed" / domain / "chunks.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ------------------------------------------------------------------ ingest

def ingest() -> None:
    setup_env("azure")
    from src.backend.retrieval.azure_store import AzureVectorStore

    store = AzureVectorStore()
    counts = {}
    for d in DOMAINS:
        ch = _chunks(d)
        store.embed_and_store(d, ch)
        counts[d] = len(ch)
    client = store._search_client()
    time.sleep(5)                                        # the index count is eventually consistent
    total = client.get_document_count()
    per = {d: client.search(search_text="*", filter=f"domain_id eq '{d}'", include_total_count=True, top=0).get_count()
           for d in DOMAINS}
    # local Chroma lacked the knowledge_base domain; add it so /query compares the same chunks
    os.environ["BACKEND_MODE"] = "local"
    from src.backend.retrieval.local_store import LocalVectorStore

    LocalVectorStore().embed_and_store("knowledge_base", _chunks("knowledge_base"))
    import chromadb

    local = {c.name: c.count() for c in chromadb.PersistentClient(path=str(ROOT / "chroma_db")).list_collections()}
    rec = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "chunks_uploaded": counts,
           "azure_index": store.index_name, "azure_document_count": total, "azure_per_domain": per,
           "local_chroma_per_domain": local, "embedding": "all-MiniLM-L6-v2 (local, 384-d) for both"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "i4_ingest.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))


# ------------------------------------------------------------------ run

ASK_N, SECURE_CLEAN, SECURE_ATTACK, DIAG_N, FORECAST_N = 16, 5, 5, 5, 50


def _items():
    """Reduced I4 item set (user, 2026-10-16). Fixed, deterministic selection; no sampling."""
    import pandas as pd

    from src.backend.kb.build import build_documents
    from src.backend.kb.queries import build_queries
    from src.backend.kb.taxonomy import load_events
    from urllib.parse import quote

    from src.backend.security.attacks import inject_query

    qs = build_queries(build_documents())                   # 64 labelled queries, fixed order
    events = sorted(int(e) for e in load_events("A").event_id)[:DIAG_N]
    origins = [t.strftime("%Y-%m-%dT%H:%M") for t in pd.date_range("2020-06-16 06:00", periods=FORECAST_N, freq="15min")]
    clean = [(f"clean:{q.query_id}", q.text) for q in qs[:SECURE_CLEAN]]
    attack = [(f"attack:{q.query_id}", inject_query(q.text)) for q in qs[SECURE_CLEAN:SECURE_CLEAN + SECURE_ATTACK]]
    out = []
    for rep in range(REPS_CHEAP):
        out += [("query", q.query_id, rep, ("GET", f"/query/knowledge_base?top_k=5&q={quote(q.text)}", None), False) for q in qs]
    for rep in range(REPS_CHEAP):
        out += [("forecast", o, rep, ("POST", "/forecast", {"asset": "solar_plant1", "origin": o, "confidence": 0.9}),
                 False) for o in origins]
    out += [("ask", q.query_id, 0, ("GET", f"/ask/knowledge_base?top_k=5&q={quote(q.text)}", None), True) for q in qs[:ASK_N]]
    out += [("secure_ask", i, 0, ("POST", "/secure-ask", {"question": text}), True) for i, text in clean + attack]
    out += [("decision", i, 0, ("POST", "/decision", {"kind": "answer", "question": text}), True)
            for i, text in clean + attack]
    out += [("diagnose", str(e), 0, ("POST", "/diagnose", {"event_id": e, "use_llm": True}), True) for e in events]
    return out


def summarise(endpoint: str, body) -> dict:
    """The fields compared between modes (everything else, e.g. request ids, is ignored)."""
    if not isinstance(body, dict):
        return {"body": body}
    if endpoint == "query":
        return {"chunk_ids": [r["chunk_id"] for r in body.get("results", [])]}
    if endpoint == "ask":
        return {"answer": body.get("answer"), "mode": body.get("retrieval_mode"),
                "sources": [s.get("chunk_id") if isinstance(s, dict) else s for s in body.get("sources", [])]}
    if endpoint == "secure_ask":
        return {"answer": body.get("answer"), "route": body.get("route"),
                "evidence": [e["chunk_id"] for e in body.get("evidence", [])]}
    if endpoint == "decision":
        o = body.get("output", {})
        return {"decision": body.get("decision"), "trust_score": body.get("trust_score"), "answer": o.get("answer"),
                "route": o.get("route"), "evidence": [e["chunk_id"] for e in o.get("evidence", [])]}
    if endpoint == "diagnose":
        return {"diagnosis": body.get("diagnosis"), "cause": (body.get("likely_cause") or {}).get("text"),
                "component": body.get("likely_cause_component"), "severity": body.get("severity"),
                "generation": body.get("generation"), "evidence": [e["chunk_id"] for e in body.get("evidence", [])]}
    if endpoint == "forecast":
        return {k: body.get(k) for k in ("prediction", "lower_bound", "upper_bound")}
    return body


def run(mode: str) -> None:
    setup_env(mode)
    from fastapi.testclient import TestClient

    from experiments.attack_simulation import ModelCycler
    from experiments.run_all import free_ram_mb
    from src.backend import main
    from src.backend.settings import get_settings

    path = raw_path(mode)
    done = set()
    if path.is_file():
        done = {(r["endpoint"], r["item"], r["rep"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                if r.get("endpoint")}
    client = TestClient(main.app, raise_server_exceptions=False)
    t0 = time.perf_counter()
    client.post("/warmup")
    status = "warming"
    while status in ("warming", "cold"):
        time.sleep(3)
        status = client.get("/ready").json()["status"]
    warm_s = round(time.perf_counter() - t0, 1)
    info = client.get("/system/info").json()
    cycler = ModelCycler(get_settings().ollama_model, BATCH)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": True, "mode": mode, "ready": status, "warmup_s": warm_s,
                             "cloud": info.get("cloud"), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        for endpoint, item, rep, (method, url, body), llm in _items():
            if (endpoint, item, rep) in done:
                continue
            if free_ram_mb() < MIN_FREE_MB:
                print(f"STOP: free RAM below {MIN_FREE_MB} MB")
                sys.exit(3)
            t = time.perf_counter()
            r = client.request(method, url, json=body)
            ms = (time.perf_counter() - t) * 1000
            try:
                payload = r.json()
            except ValueError:
                payload = r.text[:300]
            rec = {"endpoint": endpoint, "item": item, "rep": rep, "status": r.status_code, "latency_ms": round(ms, 1),
                   "out": summarise(endpoint, payload) if r.status_code == 200 else {"error": payload}}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if llm:
                cycler.tick()
    print(f"{mode}: done (warm-up {warm_s} s, ready={status})")


# ------------------------------------------------------------------ report

def report() -> None:
    import numpy as np
    import pandas as pd

    from src.backend.kb.build import build_documents
    from src.backend.kb.queries import build_queries
    from src.backend.kb.retrieval_eval import metrics_at_k

    recs, meta = {}, {}
    for mode in ("local", "azure"):
        rows = [json.loads(x) for x in raw_path(mode).read_text(encoding="utf-8").splitlines() if x.strip()]
        meta[mode] = [r for r in rows if r.get("meta")]
        recs[mode] = pd.DataFrame([r for r in rows if not r.get("meta")])
    lat = []
    for ep in ["query", "ask", "secure_ask", "decision", "diagnose", "forecast"]:
        row = {"endpoint": ep}
        for mode in ("local", "azure"):
            g = recs[mode][recs[mode].endpoint == ep]
            ok = g[g.status == 200]
            row |= {f"{mode}_requests": len(g), f"{mode}_ok": len(ok),
                    f"{mode}_median_ms": float(np.median(ok.latency_ms)) if len(ok) else None,
                    f"{mode}_p95_ms": float(np.percentile(ok.latency_ms, 95)) if len(ok) else None}
        lat.append(row)
    lat = pd.DataFrame(lat)
    eq = []
    for ep in ["query", "ask", "secure_ask", "decision", "diagnose", "forecast"]:
        a = {r.item: r.out for r in recs["local"][(recs["local"].endpoint == ep) & (recs["local"].rep == 0)].itertuples()}
        b = {r.item: r.out for r in recs["azure"][(recs["azure"].endpoint == ep) & (recs["azure"].rep == 0)].itertuples()}
        both = sorted(set(a) & set(b))
        same = sum(json.dumps(a[i], sort_keys=True) == json.dumps(b[i], sort_keys=True) for i in both)
        row = {"endpoint": ep, "items_compared": len(both), "exact_match": same,
               "exact_match_rate": same / len(both) if both else None}
        if ep in ("ask", "secure_ask", "decision"):
            key = "answer"
            row["answer_text_match"] = sum(a[i].get(key) == b[i].get(key) for i in both)
            ev = "sources" if ep == "ask" else "evidence"
            row["same_sources_or_evidence"] = sum(a[i].get(ev) == b[i].get(ev) for i in both)
        if ep == "decision":
            row["same_decision"] = sum(a[i].get("decision") == b[i].get("decision") for i in both)
        if ep == "diagnose":
            row["same_component"] = sum(a[i].get("component") == b[i].get("component") for i in both)
            row["same_evidence"] = sum(a[i].get("evidence") == b[i].get("evidence") for i in both)
        eq.append(row)
    eq = pd.DataFrame(eq)
    qs = {q.query_id: q for q in build_queries(build_documents())}
    qm = []
    a = {r.item: r.out["chunk_ids"] for r in recs["local"][(recs["local"].endpoint == "query") & (recs["local"].rep == 0) & (recs["local"].status == 200)].itertuples()}
    b = {r.item: r.out["chunk_ids"] for r in recs["azure"][(recs["azure"].endpoint == "query") & (recs["azure"].rep == 0) & (recs["azure"].status == 200)].itertuples()}
    for qid in sorted(set(a) & set(b)):
        da = [c.split("::")[0] for c in a[qid]]
        db = [c.split("::")[0] for c in b[qid]]
        sa, sb = set(a[qid]), set(b[qid])
        qm.append({"query_id": qid, "same_order": a[qid] == b[qid], "same_set": sa == sb,
                   "jaccard": len(sa & sb) / len(sa | sb) if sa | sb else 1.0,
                   "ndcg5_local": metrics_at_k(da, qs[qid].relevant, 5)["ndcg_at_k"],
                   "ndcg5_azure": metrics_at_k(db, qs[qid].relevant, 5)["ndcg_at_k"]})
    qm = pd.DataFrame(qm)
    OUT.mkdir(parents=True, exist_ok=True)
    lat.to_csv(OUT / "i4_latency.csv", index=False, encoding="utf-8")
    eq.to_csv(OUT / "i4_equivalence.csv", index=False, encoding="utf-8")
    qm.to_csv(OUT / "i4_query_metrics.csv", index=False, encoding="utf-8")
    summary = {"meta": meta, "latency": lat.to_dict("records"), "equivalence": eq.to_dict("records"),
               "query": {"queries": len(qm), "same_order": int(qm.same_order.sum()), "same_set": int(qm.same_set.sum()),
                         "mean_jaccard": float(qm.jaccard.mean()), "ndcg5_local": float(qm.ndcg5_local.mean()),
                         "ndcg5_azure": float(qm.ndcg5_azure.mean())}}
    (OUT / "i4_summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: None), encoding="utf-8")
    write_status(summary, lat, eq, meta)
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, indent=1, default=str))


def _n(v) -> str:
    return "-" if v is None or v != v else f"{v:,.0f}"


def _frac(v, n) -> str:
    return "-" if v is None or v != v else f"{int(v)}/{n}"


def write_status(summary, lat, eq, meta) -> None:
    w = {m: (meta[m][-1] if meta[m] else {}) for m in meta}
    L = ["# I4 cloud portability: local vs Azure (plan A)", "",
         f"Measured {w['azure'].get('ts', '?')} with `python -m experiments.i4_harness` (reduced item set, user request).",
         "Both modes ran on this machine through the same app (in-process TestClient). Azure mode used AI Search Free",
         "(index `greenrag`), Blob (`kbdocs`), Key Vault and App Insights in centralindia. MiniLM embeddings and the",
         "Ollama LLM (qwen2.5:7b-instruct) are local in both modes, so latency differences come from the Azure calls.", "",
         f"Warm-up before measuring (one each, excluded from the latencies): local {w['local'].get('warmup_s')} s, "
         f"Azure {w['azure'].get('warmup_s')} s.", "",
         "## Latency per endpoint (ms)", "",
         "/query and /forecast: all requests of 3 repetitions after warm-up. LLM endpoints: one run; median and p95",
         "across the items of that run.", "",
         "| endpoint | n local (ok) | local median | local p95 | n Azure (ok) | Azure median | Azure p95 |",
         "|---|---|---|---|---|---|---|"]
    for r in lat.to_dict("records"):
        L.append(f"| {r['endpoint']} | {r['local_requests']} ({r['local_ok']}) | {_n(r['local_median_ms'])} | "
                 f"{_n(r['local_p95_ms'])} | {r['azure_requests']} ({r['azure_ok']}) | {_n(r['azure_median_ms'])} | "
                 f"{_n(r['azure_p95_ms'])} |")
    q = summary["query"]
    n = q["queries"]
    L += ["", "## /query retrieval equivalence (knowledge_base domain, top 5, first repetition)", "",
          f"- Same top-5 chunks in the same order: {q['same_order']}/{n} queries.",
          f"- Same top-5 set: {q['same_set']}/{n}; mean Jaccard overlap {q['mean_jaccard']:.3f} (n = {n}).",
          f"- nDCG@5 (document level, B2 labels): local {q['ndcg5_local']:.3f}, Azure {q['ndcg5_azure']:.3f} (n = {n}).",
          "", "## Output equivalence (first run; exact match = every compared field identical)", "",
          "| endpoint | items compared | exact match | answer text same | sources/evidence same | other |",
          "|---|---|---|---|---|---|"]
    for r in eq.to_dict("records"):
        k = r["items_compared"]
        other = []
        if r.get("same_decision") is not None and r.get("same_decision") == r.get("same_decision"):
            other.append(f"same decision {_frac(r['same_decision'], k)}")
        if r.get("same_component") is not None and r.get("same_component") == r.get("same_component"):
            other.append(f"same component {_frac(r['same_component'], k)}, same evidence {_frac(r['same_evidence'], k)}")
        L.append(f"| {r['endpoint']} | {k} | {r['exact_match']}/{k} | {_frac(r.get('answer_text_match'), k)} | "
                 f"{_frac(r.get('same_sources_or_evidence'), k)} | {'; '.join(other) or '-'} |")
    L += ["", "Notes:",
          "- /ask: the legacy generator calls Ollama with default sampling (no temperature or seed), so its answer text",
          "  varies between any two runs; the retrieved sources were the same for every item. /secure-ask, /decision",
          "  and /diagnose use temperature 0 and seed 0.",
          "- /secure-ask and /decision include 5 query-injection items; they were blocked and rejected in both modes,",
          "  so those 5 items match without reaching the LLM.",
          f"- /query median rises from {_n(lat.set_index('endpoint').loc['query', 'local_median_ms'])} to "
          f"{_n(lat.set_index('endpoint').loc['query', 'azure_median_ms'])} ms (n = 192 each): each Azure request calls AI",
          "  Search over the internet and, because the legacy endpoints build a new store per request, also lists",
          "  the indexes first. The /ask difference is not separable from the LLM variance in this sample.",
          "", "## Not run", "",
          "- /ask on the other 48 labelled queries, /secure-ask and /decision on the other 54, /diagnose on the other",
          "  17 Farm A events (reduced plan).",
          "- No repetitions for the LLM endpoints (one run each), so their latency includes Ollama variance.",
          "- No KB-poisoning attacks in Azure mode (they need an attacked store); the 5 attack items are query injections.",
          "- No hosted API: the app runs locally in both modes; user-to-cloud network latency is not measured.",
          "- No cost measurement here.", ""]
    (OUT / "i4_status.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("ingest", "run"))
    ap.add_argument("--mode", choices=("local", "azure"))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.part == "ingest":
        ingest()
    elif a.part == "run":
        run(a.mode)
    if a.report:
        report()


if __name__ == "__main__":
    main()
