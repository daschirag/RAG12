# GreenRAG-Sec: working rules for Claude

Repo: `D:\Rag\GreenRAG_Azure_Cloud_Project_2026`, branch `feature/dev`, remote `origin` (github.com/daschirag/RAG12).
Spec: `docs/GreenRAG-Sec_Work_Split.docx` (Dr. Priya V work split). Phase list: `docs/PLAN.md`. Status: `PROGRESS.md`.
Read PROGRESS.md first after any context reset.

## How to work (non-negotiable)
1. One phase at a time. At the end of each phase: run that phase's tests, show the real output, update PROGRESS.md,
   commit to feature/dev, then STOP until the user types "next". Never start the next phase unprompted.
2. A phase is done only if its tests ran and passed in this session. On failure: show it, fix, re-run.
3. Never fabricate or estimate results. Every number in a results table comes from a real run in this repo, with
   fixed seeds and raw output saved under `results/`. Report worse-than-baseline results as they are. No cherry-picking.
4. Synthetic/generated data (KB docs, injected faults, attacks) is labelled `synthetic` in metadata and docs.
   No copyrighted text (standards, manuals): write original paraphrased summaries.
5. Reproducibility: chronological splits for time series (never random); split fault data by asset/event; no leakage
   (rolling features use shift, scalers fit on train only); fixed seeds; log package versions.
6. Environment: `uv pip install ...` only (never pip). Use `python`. `.py` modules/scripts only, no notebooks.
   Tests in `tests/` with pytest + FastAPI TestClient (no live server, no --reload). Explicit `encoding="utf-8"`
   for file IO; ASCII-safe console output. No debug prints or log files left behind.
7. Machine: RTX 4050 6GB, 16GB RAM, limited D: space. Small models. Caches under `D:\Rag\cache`
   (`src/backend/runtime_env.configure_cache_dirs()` forces HF_HOME/TORCH_HOME there; call it in any entry point
   that loads HF/torch models). Ask before any download > 1GB. Fault data is huge: develop on Wind Farm A, sampled.
8. Do not break: /health, /system/status, /domains, /domains/{id}, POST /ingest/{id}, /query/{id}, /query/{id}/top,
   /ask/{id}, the BACKEND_MODE local/azure abstraction, the console UI (src/frontend). `python -m pytest` must pass
   after every phase (regression suite: tests/test_api_regression.py, test_backend_mode.py, test_frontend.py,
   test_local_integration.py; test_ollama_integration.py skips if Ollama is down).
9. Ollama (`ollama serve`) local models: qwen2.5:7b-instruct, llama3.2:3b, nomic-embed-text.
10. Report figures: matplotlib, black-and-white/grayscale only.
11. Git: one commit per phase, clear message. Never commit datasets, chroma_db, model weights, .env, logs.

## Commands
- Tests: `python -m pytest` (markers: `integration`, `ollama`). Fast subset: `python -m pytest -m "not integration and not ollama"`.
- API: `uvicorn src.backend.main:app` (no --reload). Ingest chunks: `python -m src.backend.ingestion.pipeline --domain solar`.
- Experiments (from I3): `python -m experiments.run_all`.

## Data facts (verified 2026-09-28 by inspecting files)
- Solar `dataset/solar/`: Plant_1/Plant_2 `*_Generation_Data.csv` (per inverter SOURCE_KEY, 15-min: DC_POWER, AC_POWER,
  DAILY_YIELD, TOTAL_YIELD) and `*_Weather_Sensor_Data.csv` (AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION).
  Range 2020-05-15 00:00 to 2020-06-17 23:45 (~34 days). NO humidity column: do not invent one (ask user about any
  external source). Timestamp formats differ: Plant_1 generation is `dd-mm-yyyy HH:MM`; the other three are ISO
  `yyyy-mm-dd HH:MM:SS`. Parse with explicit formats.
- Wind `dataset/wind/T1.csv`: one turbine, 2018, 10-min, 50,530 rows. Columns: `Date/Time` (`dd mm yyyy HH:MM`),
  `LV ActivePower (kW)`, `Wind Speed (m/s)`, `Theoretical_Power_Curve (KWh)`, `Wind Direction (°)`. File starts
  with a UTF-8 BOM (read with `encoding="utf-8-sig"`).
- Fault `dataset/fault_diagnosis/Wind Farm {A,B,C}/`: `datasets/` per-asset SCADA CSVs (A: 22 files 766MB,
  B: 15 files 1.3GB, C: 58 files 17GB), `event_info.csv` (semicolon-separated; columns asset|asset_id, event_id,
  event_label, event_start, event_start_id, event_end, event_end_id, event_description), `feature_description.csv`.
  Asset column is `asset` in Farm A but `asset_id` in B/C. These are WIND-TURBINE faults (e.g. transformer failure,
  hydraulic group, converter failure), not solar-inverter faults. The plan's fault list (inverter overheating, etc.)
  does not match; derive the taxonomy from event_info.csv (Phase B4). Synthetic solar-inverter faults only if
  labelled synthetic.
- Existing processed chunks: `data/processed/{solar,wind,fault_diagnosis}/chunks.jsonl` (gitignored).

## Code map
- `src/backend/main.py` FastAPI app. `config/` domain manifests (YAML). `ingestion/` loaders + chunkers + pipeline.
  `retrieval/` VectorStore ABC, local (MiniLM + Chroma) and Azure (AI Search) stores, factory by BACKEND_MODE.
  `serving/` generator (Ollama / Azure OpenAI) and deterministic numeric router.
- New packages (empty until their phase): `forecasting/`, `faultdiag/`, `kb/`, `security/`, `trust/`, `evaluation/`.
- `experiments/` runnable experiment modules; `results/<area>/` CSV, markdown tables, grayscale plots, raw outputs.
