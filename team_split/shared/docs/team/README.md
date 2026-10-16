# GreenRAG-Sec team documentation

GreenRAG-Sec is a local, security-aware assistant for renewable-energy operations. It runs entirely on one
machine (FastAPI + ChromaDB/in-memory hybrid index + MiniLM + a cross-encoder + Ollama); **no cloud service is
used** (decision 2026-09-30). Every number in these documents is copied from a file under `results/` (named next to
it) or from a live API call; nothing is estimated.

| member | track | document |
|---|---|---|
| Minmini K (23BIT0198) | A - forecasting with uncertainty | [minmini_forecasting.md](minmini_forecasting.md) |
| Tarunica S (23BIT0185) | B - knowledge base, hybrid RAG, fault diagnosis | [tarunica_rag_diagnosis.md](tarunica_rag_diagnosis.md) |
| Pritika Kasinathan (23BIT0232) | C - security and trust | [pritika_security_trust.md](pritika_security_trust.md) |

Each document has: scope, files, endpoints, measured results, limitations, likely viva questions with answers, a
live demo script, and a "what is not done" list.

## Shared architecture

```
 Console (/console/)  ──►  FastAPI  ──►  POST /assistant  (intent router, ablation configs A-F)
                               │
       question ──► input checks ─► injection detector ─► hybrid retrieval (BM25 + MiniLM, cross-encoder rerank)
                               ─► document trust (signed registry + content checks) ─► evidence filter ─► Ollama LLM
       forecast ──► XGBoost point + quantile XGBoost + CQR per group (interval)
       diagnosis ─► B4 detector score + secure retrieval over the knowledge base ─► LLM (or evidence-only fallback)
                               │
                               ▼
              Trust-aware decision engine: T = 0.35E + 0.15R + 0.30C + 0.20S  ─►  accept / review / reject
```

The LLM only recommends; only "accept" may be acted on without review and the API never executes anything.
The C3 telemetry detector is evaluated offline and in the Security lab; it is **not** wired into `/forecast`.
A screenshot of every console page (light, dark and 390 px mobile) is in `docs/screenshots/`.

## How to run (5 commands)

```powershell
uv pip install -r requirements.txt               # 1. dependencies (uv only)
ollama serve                                     # 2. local LLM (models: qwen2.5:7b-instruct, llama3.2:3b)
python -m src.backend.kb.build                   # 3. knowledge base (data/processed/knowledge_base/)
python -m experiments.build_forecast_model       # 4. deployed forecast models (models/forecasting/)
uvicorn src.backend.main:app                     # 5. API + console: http://127.0.0.1:8000/console/
```

The console calls `POST /warmup` when it opens and shows progress until `GET /ready` reports `ready`
(the warm-up in this session took 66-73 s, see `GET /warmup`). Tests: `python -m pytest`.
Experiment status: `python -m experiments.run_all` (reuses finished results, never re-runs them).
Configuration: environment variables only (`src/backend/settings.py`: `BACKEND_MODE`, `OLLAMA_MODEL`, `OLLAMA_HOST`,
`GREENRAG_CONSOLE_ORIGINS`, `GREENRAG_LLM_TIMEOUT_S`, `GREENRAG_LOG_LEVEL`, `GREENRAG_CACHE_DIR`).

## API map

Interactive docs with a recorded real example for every endpoint: `http://127.0.0.1:8000/docs`.
Every error has the body `{code, message, hint, detail, request_id}`; every response carries `X-Request-ID`.

| group | endpoint | owner | what it does |
|---|---|---|---|
| System | `GET /health` | shared | liveness (`{"status": "ok"}`) |
| System | `GET /ready` | shared | artifacts, result files, KB store, model cache, Ollama, warm-up state (503 only if a required artifact is missing) |
| System | `POST /warmup`, `GET /warmup` | shared | load every model in the background; progress per step |
| System | `GET /system/status`, `GET /system/info` | shared | mode ("local"), models, package versions (no secrets) |
| Track A | `POST /forecast` | Minmini | 1-h-ahead AC power + CQR interval for `solar_plant1`, `solar_plant2`, `wind_t1` |
| Track A | `GET /forecast/assets`, `GET /forecast/series` | Minmini | assets, fit windows, and a day of forecasts with observed values |
| Track B | `POST /diagnose` | Tarunica | evidence-grounded Wind Farm A diagnosis (logged / unlogged mode) |
| Track B | `GET /diagnose/events`, `GET /kb/chunks` | Tarunica | event list with detector scores; text of cited chunks |
| Track C | `POST /secure-ask` | Pritika | knowledge-base answer through the secure pipeline |
| Track C | `POST /decision` | Pritika | run a pipeline, then the trust-aware decision engine |
| Track C | `GET /security/lab/catalogue`, `POST /security/lab/{telemetry,injection,poisoning}` | Pritika | run the synthetic attacks against the live detectors |
| Integration | `POST /assistant`, `GET /assistant/suggestions` | shared | intent routing under ablation configs A-F (trust only in F) |
| Results | `GET /results/summary`, `GET /results/summary/{track}` | shared | measured numbers read from `results/`, with caveats |
| Legacy | `/domains`, `/ingest/{id}`, `/query/{id}`, `/query/{id}/top`, `/ask/{id}` | shared | original per-domain retrieval (unchanged) |

## Project-wide limitations (all members)

- **Synthetic data.** All attack data is synthetic, and 90 of 189 knowledge-base documents are synthetic
  (marked `[SYNTHETIC]`).
- **No cloud.** Nothing is deployed to any cloud, and no local-vs-cloud number exists
  (`results/cloud/i4_status.md`: PENDING).
- **Ablation A-F not run.** The configs exist in `src/backend/orchestrator.py` and the console, but no
  ablation experiment has been run (`results/run_all_status.md`).
- **No authentication** on any endpoint (threat model).
- **Not committed.** All work since commit 4e586aa is uncommitted (the no-git rule); the team commits it.
