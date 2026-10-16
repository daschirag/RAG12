# GreenRAG-Sec threat model (C1)

Scope: the architecture that exists in this repository and that C2-C7 will be tested against: **local mode only**
(FastAPI + ChromaDB + sentence-transformers + local Ollama). The Azure code path (`BACKEND_MODE=azure`) remains in
the code but is not deployed, set up or tested (project decision, 2026-09-30); Azure-specific threats are listed
only as out of scope.

## System as implemented

```
dataset/ (CSV: solar, wind, fault SCADA + event_info.csv)
   |  A1/wind preprocessing, B4 feature extraction (python, local files)
   v
data/processed/ (cleaned frames, KB documents.jsonl + chunks.jsonl)   results/ (metrics)   models/forecasting/*.joblib
   |  POST /ingest/{domain}: MiniLM embeddings -> ChromaDB folder ./chroma_db
   v
retrieval: BM25 + MiniLM vector (RRF) -> cross-encoder rerank -> sibling-chunk expansion   (src/backend/retrieval/)
   v
LLM: local Ollama (qwen2.5:7b-instruct, HTTP 127.0.0.1:11434), evidence-only prompts, citation validation
   v
outputs: /ask answers, POST /forecast intervals, POST /diagnose DiagnosisReport (confidence, severity, evidence)
   v
decision engine: not implemented yet (C5)
```

Entry points: the FastAPI app `src/backend/main.py` (uvicorn, binds 127.0.0.1 by default) with `/health`,
`/system/status`, `/domains`, `POST /ingest/{id}`, `/query/{id}`, `/query/{id}/top`, `/ask/{id}`, `POST /forecast`,
`POST /diagnose`; the static console UI `src/frontend/`; and the offline scripts in `experiments/`.

Trust boundaries: (1) data files entering from `dataset/` and `data/processed/`; (2) HTTP requests to the API;
(3) text crossing from retrieval into the LLM prompt; (4) model/package downloads (Hugging Face Hub, Ollama, uv);
(5) the output that a human (or, later, the decision engine) acts on.

Status of each control: **existing** (in the code today, file named), **partial** (exists but only partly covers
the threat), **planned** (phase of docs/PLAN.md that will address it), **out of scope** (acknowledged, not addressed
in this prototype). Severity is a qualitative judgement for this prototype (H/M/L), not a measured risk.

## 1. IoT / SCADA

| ID | Attack | Vulnerability (in this system) | Impact | Detection | Mitigation | Status |
|---|---|---|---|---|---|---|
| SC-1 (H) | False data injection: altered sensor values (irradiation, wind speed, temperatures, power) | Telemetry is read from CSV and trusted; no integrity check or signature (`src/backend/forecasting/preprocess.py`, `src/backend/forecasting/wind.py`) | Wrong forecasts and intervals (A3-A6, wind), missed or false fault alarms (B4), misleading RAG answers | Only gross violations: A1 physical bounds and trailing robust-z outlier flags | Rate-of-change, cross-sensor consistency and a statistical telemetry detector (C3); data-quality flag passed to the decision engine (C5); attack scenarios in C2 | partial (C2, C3, C5) |
| SC-2 (H) | Sensor spoofing / replay: a plausible stream (e.g. a replayed normal day) replaces the real one | Values within physical limits pass all checks; no cross-check of related signals (AC power vs irradiation, power vs power curve) | Detector and forecasts see normal behaviour while a fault develops | None today | Cross-sensor physics checks and replay detection (repeated windows) (C3) | planned (C2, C3) |
| SC-3 (M) | Status-code or logbook manipulation | In Wind Farm A `status_type_id` follows the fault logbook and separates anomaly from normal windows perfectly (B4 finding) | A detector using it would trust a manipulable label; logbook text also feeds the knowledge base | Leakage identified in B4 | `status_type_id` excluded from all B4 features and row filters (`src/backend/faultdiag/data.py`); logbook-derived KB documents to be trust-scored (C3/C4) | existing (exclusion); planned (trust, C3/C4) |
| SC-4 (M) | Missing or delayed data used to hide an event (dropped rows, gaps) | Gaps are expected in the data (3.9% of wind slots, 106 solar slots); forecasts use train-only imputation | An attacker's gap looks like an ordinary data gap | The A1 and wind preprocessing reports count missing slots (`src/backend/forecasting/preprocess.py`, `src/backend/forecasting/wind.py`); no alarm on unusual gaps | Gap-rate monitoring as part of the telemetry detector (C3) | partial (C3) |

## 2. Ingestion

| ID | Attack | Vulnerability (in this system) | Impact | Detection | Mitigation | Status |
|---|---|---|---|---|---|---|
| IN-1 (H) | Knowledge-base poisoning at build time: a malicious document (e.g. a fake procedure telling technicians to bypass the safety chain) | Any document passing the KBDocument schema enters the KB; no source allow-list (`src/backend/kb/build.py`, `src/backend/kb/schema.py`) | Retrieved as evidence and repeated by the LLM as guidance | Provenance fields and a content SHA-256 per document are recorded at build (`results/kb/kb_build.json`) but never re-verified later | Document trust verification against the build manifest and a provenance allow-list in the retrieval path (C3/C4); poisoning scenarios in C2 | partial (C2-C4) |
| IN-2 (H) | Tampering with `data/processed/<domain>/chunks.jsonl` between build and `POST /ingest` | `/ingest` embeds whatever the file contains; chunk metadata carries `doc_sha256` but it is not checked | Altered chunks enter ChromaDB without trace | None at ingest time | Verify chunk document hashes against the build manifest before embedding; reject mismatches (C3) | planned (C3) |
| IN-3 (M) | Indirect prompt injection planted in text that becomes KB content (logbook descriptions, maintenance records) | Logbook descriptions are copied verbatim into fault documents (B1) and are not screened | Instructions reach the LLM prompt through retrieval | None today | Screen KB text at build time and retrieved text at query time (C3/C4) | planned (C2-C4) |
| IN-4 (L) | Malformed or oversized input files | Loaders assume known formats; fault data is large (Farm C 17 GB) | Pipeline failure | Explicit timestamp formats and required-column checks raise clear errors (`src/backend/forecasting/preprocess.py`, `src/backend/kb/taxonomy.py`) | Keep strict parsing; develop on Farm A with sampling | existing |

## 3. Cloud (for the local stack: hosting and supply chain)

| ID | Attack | Vulnerability (in this system) | Impact | Detection | Mitigation | Status |
|---|---|---|---|---|---|---|
| CL-1 (M) | Supply-chain substitution of downloaded models (MiniLM embedder, cross-encoder) | Models are fetched from the Hugging Face Hub by name without a pinned revision or hash (`src/backend/retrieval/local_store.py`, `src/backend/kb/retrieval_eval.py`, `src/backend/retrieval/hybrid_retriever.py`) | A changed model silently alters retrieval | None | Pin model revisions and record hashes of the cached files; cache is kept in `D:\Rag\cache` (`src/backend/runtime_env.py`) | partial (cache location only) |
| CL-2 (M) | Supply-chain substitution of the Ollama model or Python packages | Ollama models are pulled by tag; Python packages are installed by name with uv (`requirements.txt` has no versions) | Changed generator behaviour or malicious code in a dependency | Package versions are recorded in every result file (`src/backend/evaluation/provenance.py`) | Pin package versions and Ollama model digests for releases (I5) | partial |
| CL-3 (M) | Exposure of local services on the network (API on 8000, Ollama on 11434) | uvicorn binds 127.0.0.1 by default (the documented run command in `CLAUDE.md` passes no `--host`); Ollama's API has no authentication | If bound to 0.0.0.0, anyone on the network could use the API or swap models | None | Keep both bound to localhost; authentication before any exposure (see DE/API notes, C7) | partial (localhost default) |
| CL-4 (L) | Azure deployment threats (key theft, public endpoints, quota exhaustion) | Azure mode uses API keys from `.env` (`src/backend/retrieval/azure_store.py`); not deployed | None while local-only | n/a | Out of scope by decision (local mode only); `.env` is gitignored | out of scope |

## 4. Database (local files: processed data, results, model artifacts)

| ID | Attack | Vulnerability (in this system) | Impact | Detection | Mitigation | Status |
|---|---|---|---|---|---|---|
| DB-1 (H) | Malicious model artifact executes code when loaded | `POST /forecast` loads `models/forecasting/*.joblib` with `joblib.load` (pickle) (`src/backend/forecasting/service.py`) | Arbitrary code execution in the API process | None; artifacts are not hashed | Load only from the local folder written by `experiments/build_forecast_model.py` (documented); add a hash check against `results/forecasting/a6_model_card.json` before loading (C3) | partial (C3) |
| DB-2 (M) | Tampering with stored results or processed data (`results/`, `data/processed/`) | Protected only by OS permissions; no signatures | Wrong metrics or inputs go unnoticed | Provenance stamps (versions, base commit) in result files (`src/backend/evaluation/provenance.py`); committed results can be compared with regenerated ones | All results are regenerable from code; compare reruns; signed manifests if deployed | partial |
| DB-3 (L) | Leakage of stored data | Public datasets and synthetic KB text only; no personal data | Low | n/a | Keep secrets out of the repo (`.env` gitignored) | existing |

## 5. Vector DB (ChromaDB, `./chroma_db`)

| ID | Attack | Vulnerability (in this system) | Impact | Detection | Mitigation | Status |
|---|---|---|---|---|---|---|
| VD-1 (H) | Direct poisoning: writing documents into `./chroma_db` outside the ingestion path, or calling `POST /ingest` with tampered chunks | ChromaDB is an unauthenticated local folder; retrieval trusts every stored chunk (`src/backend/retrieval/local_store.py`) | Attacker-chosen text is returned as evidence | None | Verify retrieved chunks (doc id + hash) against the KB build manifest before use (C3/C4) | planned (C3, C4) |
| VD-2 (M) | Ranking manipulation: keyword-stuffed or adversarial text that ranks top for many queries | Hybrid retrieval ranks by lexical and embedding similarity only; no per-source weighting | One poisoned document dominates retrieval | None; B2/B3 metrics measure relevance, not poisoning | Trust-weighted evidence filtering (C4); measure how often poisoned documents are retrieved (C2, C6) | planned (C2, C4, C6) |
| VD-3 (L) | Cross-domain leakage | One collection per domain (`src/backend/retrieval/local_store.py`); `access_policy: tenant_isolated` in the manifests (`src/backend/config/domain_manifest.py`) is declarative only | Low with public data | n/a | Per-domain collections separate data today; enforce policies once authentication exists | partial |

## 6. Retriever

| ID | Attack | Vulnerability (in this system) | Impact | Detection | Mitigation | Status |
|---|---|---|---|---|---|---|
| RT-1 (H) | Poisoned or untrusted evidence passed to the LLM as fact | Production retrieval (hybrid + rerank + sibling expansion) has no trust check; `[SYNTHETIC]` documents are labelled but not down-weighted | Unsafe or wrong recommendations grounded in malicious text | None | Secure retrieval pipeline: trust verification and evidence filtering between retriever and LLM (C3, C4) | planned (C3, C4) |
| RT-2 (M) | Query steering: another turbine's id or dates in a `/diagnose` window request | Retrieval queries are built from request fields; "strong" evidence trusts farm/asset/window metadata | Evidence about another incident presented as strong support | Strong evidence requires same farm, same turbine and an overlapping window (`src/backend/faultdiag/diagnosis.py`); pydantic request validation (`src/backend/faultdiag/service.py`) | Input security checks (C4) | partial (C4) |
| RT-3 (M) | Sibling expansion pulling in untrusted neighbour text | Expansion adds adjacent chunks of retrieved documents (`src/backend/retrieval/hybrid_retriever.py`); a poisoned chunk next to a relevant one rides along | Poisoned text reaches the prompt without being ranked | None | Apply the same trust check to expanded chunks (C4) | planned (C4) |
| RT-4 (L) | Costly queries (very long text) slow the reranker | Rerank costs about 1 s per query on CPU (B3); no length limit on `/ask` | Latency degradation | None | Query length limits (C4) | planned (C4) |

## 7. LLM (local Ollama)

| ID | Attack | Vulnerability (in this system) | Impact | Detection | Mitigation | Status |
|---|---|---|---|---|---|---|
| LM-1 (H) | Direct prompt injection in the user query ("ignore your rules...") | User text goes into the prompt on `/ask` and `/diagnose` window requests; no screening | Evidence-only instructions ignored, invented content | None | Prompt-injection detector and input security before retrieval and generation (C3, C4) | planned (C2-C4) |
| LM-2 (H) | Indirect prompt injection through retrieved documents or maintenance records | Retrieved text is concatenated into the prompt; the model cannot separate instructions from evidence | As LM-1, triggered by content the user never wrote | None | Screen retrieved text (C3) and filter untrusted evidence (C4); citation validation already rejects citations outside the evidence (`src/backend/faultdiag/diagnosis.py`) | planned (C2-C4) |
| LM-3 (H) | Hallucination (not an attacker, but the main integrity risk measured) | The generator embellishes: B6 found 17 invented content claims in 44 GreenRAG fault answers | Plausible but unsupported maintenance advice | B6 claim-level evaluation (offline); in the pipeline, confidence is low when evidence does not cover the incident | Evidence-only prompts, citation validation with evidence-only fallback, severity only with strong evidence (`src/backend/faultdiag/diagnosis.py`, `src/backend/evaluation/rag_eval.py`); the decision engine must never act on LLM text alone (C5) | partial (C5) |
| LM-4 (M) | Backend stall or crash (runner crash, hung call) | Ollama crashed on cold loads (Phase 0) and one call hung for 39,097 s (B6) | Requests block | Timeout and crash detection in the LLM clients | 120 s timeout with one retry, one retry on runner crash, evidence-only fallback in `/diagnose` (`src/backend/faultdiag/diagnosis.py`, `src/backend/serving/generator.py`) | existing |

## 8. Decision engine

| ID | Attack | Vulnerability (in this system) | Impact | Detection | Mitigation | Status |
|---|---|---|---|---|---|---|
| DE-1 (H) | LLM output used directly as an operational decision | No decision engine yet; reports are advisory | An injected or hallucinated recommendation could be acted on | n/a until C5 | Trust-aware decision engine: evidence verification, security verification, operational constraints, risk, trust score, accept / review / reject; the LLM never decides (C5) | planned (C5) |
| DE-2 (M) | Over-confident inputs: weak detector scores or weak evidence treated as certain | The fault detector is not distinguishable from chance (event AUC 0.742, 95% CI [0.500, 0.950], B4) | Decisions based on unreliable signals | `/diagnose` confidence carries the measured reliability, cannot be "high" with this detector, and is low when evidence is weak (schema-enforced, `src/backend/faultdiag/diagnosis.py`) | Feed measured reliability and evidence support into the trust score (C5) | partial (C5) |
| DE-3 (M) | Repudiation: no record of which evidence and scores led to an outcome | `/diagnose` returns evidence and scores but nothing is stored | Decisions cannot be audited | n/a | Decision records with inputs, trust-score components and outcome (C5, C7) | planned (C5, C7) |
| DE-4 (H) | Unauthenticated callers triggering data changes (e.g. `POST /ingest` overwriting a collection) or decisions | No authentication on any endpoint (`src/backend/security/auth.py` is empty); CORS restricts browser origins to localhost but is not access control | Anyone who can reach the port can re-ingest data or request decisions | None | Localhost-only binding today; token authentication for state-changing endpoints (`/ingest`, `/decision`) with C7 | partial (C7) |

## Coverage by later phases

| phase | addresses |
|---|---|
| C2 attack simulator | SC-1, SC-2, IN-1, IN-3, VD-2, LM-1, LM-2 (false data injection, spoofing, KB poisoning, prompt injection in queries, documents, maintenance records and KB entries) |
| C3 detectors | SC-1, SC-2, SC-4, IN-1, IN-2, IN-3, DB-1, VD-1, RT-1, LM-1, LM-2 |
| C4 secure retrieval | IN-1, VD-1, VD-2, RT-1, RT-2, RT-3, RT-4, LM-1, LM-2 |
| C5 decision engine | SC-1 (data-quality input), LM-3, DE-1, DE-2, DE-3 |
| C6 security evaluation | measures detection rate, false positives and attack success for the C2 scenarios |
| C7 API | DE-3, DE-4 |

Summary: 30 threats across 8 layers; 4 existing, 13 partial, 12 planned, 1 out of scope (counts checked against the
tables by `tests/test_threat_model.py`).
