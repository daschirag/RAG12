# GreenRAG-Sec

A secure retrieval-augmented assistant for renewable-energy operations, running fully locally:
FastAPI + ChromaDB/MiniLM retrieval + a cross-encoder reranker + Ollama (qwen2.5:7b-instruct). It combines
1-hour-ahead power forecasting with calibrated intervals, wind-turbine fault diagnosis grounded in a knowledge
base, a secure retrieval pipeline against poisoning and prompt injection, and a trust-aware decision engine
that turns every recommendation into accept / review / reject.

The original project title says "Azure Cloud". Since 2026-09-30 the project is **local mode only**: the
`BACKEND_MODE=azure` code path exists but has never been deployed or tested (see Limitations).

All results come from real runs in this repo (fixed seeds, raw outputs under `results/`). Headline numbers are in
[docs/RESULTS.md](docs/RESULTS.md); each one links to its summary file.

## Quick start

```powershell
uv pip install -r requirements.txt          # never pip
ollama serve                                # models: qwen2.5:7b-instruct, llama3.2:3b, nomic-embed-text
python -m src.backend.kb.build              # knowledge base -> data/processed/knowledge_base/
python -m experiments.build_forecast_model  # deployed forecasters -> models/forecasting/ (gitignored)
uvicorn src.backend.main:app                # API + console: http://127.0.0.1:8000/console/ (no --reload)
```

The console (`src/console/`, plain HTML/CSS/ES modules, no build step) is served by the API itself. It calls
`POST /warmup` on open and shows loading progress until `GET /ready` reports `ready`; cold start is about a
minute or more. Its pages are Assistant, Forecast, Fault diagnosis, Security lab, Results and About; every number
it shows comes from a live API call or `GET /results/summary`. Screenshots (light, dark, 390 px mobile):
`docs/screenshots/` (`python scripts/capture_screenshots.py`, uses the installed Chrome). The older console in
`src/frontend/` (legacy `/ask` only) still works with `python -m http.server 5500`.

Configuration is environment variables only, read by one settings class (`src/backend/settings.py`):
`BACKEND_MODE`, `OLLAMA_MODEL`, `OLLAMA_HOST`, `GREENRAG_CONSOLE_ORIGINS` (CORS allow-list),
`GREENRAG_LLM_TIMEOUT_S`, `GREENRAG_LOG_LEVEL`, `GREENRAG_CACHE_DIR`. The startup log prints a banner with the
mode and models. Team documentation (per member, with demo scripts): [docs/team/](docs/team/README.md).

Tests: `python -m pytest` (the `ollama` marker needs Ollama running; quick subset:
`python -m pytest -m "not integration and not ollama"`).
Experiments: `python -m experiments.run_all` reports what is complete and reuses it. `--run-missing` runs only
the incomplete steps, one at a time.

## API

| endpoint | what it does |
|---|---|
| `GET /health`, `/system/status`, `/domains`, `/domains/{id}` | status and domain manifests |
| `POST /ingest/{id}`, `GET /query/{id}`, `/query/{id}/top`, `/ask/{id}` | original per-domain retrieval and grounded ask (unchanged) |
| `POST /forecast` | 1-h-ahead AC power for `solar_plant1`, `solar_plant2` or `wind_t1`, with a CQR interval at 0.80/0.90/0.95 nominal coverage |
| `POST /diagnose` | evidence-grounded Wind Farm A fault diagnosis (B5 report, cited chunk ids, deterministic severity) |
| `POST /secure-ask` | C4 secure pipeline: input checks, injection detection (block), retrieval, document trust, evidence filter, then the LLM |
| `POST /decision` | runs a served pipeline (answer / diagnosis / forecast), then the C5 engine: accept / review / reject, trust score `T = 0.35E + 0.15R + 0.30C + 0.20S`, components and reasons |
| `POST /assistant`, `GET /assistant/suggestions` | orchestrator: routes by intent under ablation config A-F. Only F (full) returns the trust decision |
| `GET /ready`, `POST /warmup`, `GET /warmup`, `GET /system/info` | readiness (artifacts, model cache, Ollama, warm-up), background model loading, versions |
| `GET /forecast/assets`, `GET /forecast/series` | forecast assets and fit windows; a day of forecasts with intervals and observed values |
| `GET /diagnose/events`, `GET /kb/chunks` | Wind Farm A events with detector scores; text of cited knowledge-base chunks |
| `GET /security/lab/catalogue`, `POST /security/lab/{telemetry,injection,poisoning}` | run the synthetic attacks against the live detectors and the engine |
| `GET /results/summary`, `GET /results/summary/{track}` | measured numbers read from `results/` with caveats |

Interactive docs with grouped tags and a recorded real response for every endpoint: `/docs`. Errors always
have the body `{code, message, hint, detail, request_id}`, and every response carries an `X-Request-ID` header
(one structured log line per request with its latency). If the LLM fails or times out, `/secure-ask`,
`/diagnose`, `/decision` and `/assistant` fall back to evidence-only text; the legacy `/ask` keeps its original
behaviour (HTTP 500).

Only "accept" may be acted on without review. The LLM only recommends; the API never executes an action.

## Repository map

- `src/backend/forecasting/`: preprocessing, features, models (persistence, RF, XGBoost, LSTM, TCN), conformal
  intervals, wind replication, `/forecast` service.
- `src/backend/kb/`: knowledge base (99 real + 90 synthetic documents), chunking, labelled queries.
- `src/backend/retrieval/`: BM25, vector, RRF hybrid, cross-encoder rerank, query classifier.
- `src/backend/faultdiag/`: Farm A detectors, diagnosis report, `/diagnose` service.
- `src/backend/security/`: attacks (synthetic), telemetry / injection / document-trust detectors, evidence filter,
  secure pipeline.
- `src/backend/trust/`: decision engine, measured signals, `/decision` service.
- `src/backend/orchestrator.py`: `/assistant`, ablation configs A-F.
- `src/backend/api/`: system (ready/warm-up/info), results summary, catalog and security-lab routers; recorded OpenAPI examples.
- `src/backend/settings.py`, `api_support.py`: settings class; error format, request ids, request log.
- `src/console/`: the single-page console.
- `experiments/`: one module per experiment, plus `run_all.py`. `results/<area>/`: CSV, markdown and grayscale
  figures.
- Docs: [forecasting](docs/forecasting.md), [knowledge base](docs/knowledge_base.md),
  [fault diagnosis](docs/fault_diagnosis.md), [RAG evaluation](docs/rag_evaluation.md),
  [threat model](docs/threat_model.md), [attack simulation](docs/attack_simulation.md),
  [detectors](docs/detectors.md), [secure retrieval](docs/secure_retrieval.md),
  [decision engine](docs/decision_engine.md), [security evaluation](docs/security_evaluation.md),
  [results](docs/RESULTS.md), [plan](docs/PLAN.md), status in [PROGRESS.md](PROGRESS.md).

## Limitations (honest list)

**Data**
- **Solar data is short.** Each plant covers 34 days, the fixed test block is 5 days and the rolling origin
  covers 12 days. Interval coverage CIs are wide: Plant 1 cqr_mondrian 90% daylight coverage is 0.893 with
  CI [0.853, 0.927].
- **No humidity column.** The plan asked for humidity, but the solar dataset has none, and no external
  source was added.
- **Faults are wind-turbine faults, not solar-inverter faults.** The labelled fault data (Wind Farms A-C)
  holds wind-turbine faults. The plan's inverter fault list has no labelled counterpart: inverter material in
  the KB is synthetic, and only Farm A was modelled.
- **Licences and source URLs are unrecorded.** The Kaggle licence text and URLs were never recorded in the
  repo, and the fault dataset README states no licence (`results/report_numbers.md`).

**Models and evaluation**
- **Wind forecasting loses to persistence on MAE.** Persistence scores 298.1 kW against XGBoost's 329.6.
  The learned models are only slightly better on RMSE, and every quantile method under-covers in the
  high-power regime.
- **The fault detector is not distinguishable from chance.** Event AUC is 0.742 with CI [0.50, 0.95], and
  attribution top-1 is 0/12. As a result, the decision engine never accepts a diagnosis (C = 0), which is by
  design.
- **Half the knowledge base is synthetic.** 90 of 189 KB documents (manuals, troubleshooting, maintenance
  records, standard overviews) are original synthetic text marked `[SYNTHETIC]`. Answers grounded in them are
  only as good as that text.
- **The judge is a local LLM.** qwen2.5:7b-instruct judges faithfulness and hallucination. It is not ground
  truth: it judges its own family's outputs, and its calibration set is small and easy. Template statements are
  verified by code instead.
- **Missed injections get through.** In C6 the injection detector never blocked 4 of the 15 held-out
  phrasings. The decision engine then accepted 51 answers that obeyed the attacker (e.g. "I approve
  skipping the pitch battery test"), because its unsafe-directive check is a keyword safeguard
  (`docs/security_evaluation.md`).
- **Detector rates are optimistic.** The injection detector catches the known C2 template 66/66, but held-out
  phrasings only 11/15. All attack data is synthetic and written by the same project that wrote the defences.

**Deployment**
- **The telemetry detector is not in the served forecast path.** It decides per whole day, so served
  forecasts are at best "review" from `/decision`.
- **No authentication.** No endpoint has authentication or rate limiting (`security/auth.py` is empty; see the
  threat model). CORS is limited to the console origins, which is not access control.
- **Not built.** Event-driven ingestion, tiered serving and the Power BI view do not exist (placeholder files only).
- **Azure is untested (I4 PENDING).** Nothing is deployed, and no Azure latency, cost or equivalence number
  exists (`results/cloud/i4_status.md`).
- **Ablation A-F has not been run.** The configs are implemented in the orchestrator, but the ablation
  experiment still needs an LLM run per config; `run_all` lists it as manual.
- **No release tag.** The no-git rule applies, so all changes since commit 4e586aa are uncommitted and the
  user creates the tag.
