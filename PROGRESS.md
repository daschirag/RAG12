# GreenRAG-Sec progress

Phase list: `docs/PLAN.md`. Rules: `CLAUDE.md`.

| Phase | Status | Tests | Commit |
|---|---|---|---|
| 0 Setup | done | 38 passed, 0 skipped (with Ollama running) | e043ac1 + Ollama fix commit |
| A1 Solar preprocessing | done | 19 new (57 total) passed | see git log "A1" |
| A2 Feature engineering | done | 12 new (69 total) passed | 3f2a690, 6063449 + results commit |
| A3-A6 Forecasting | not started | | |
| B1-B7 RAG + fault diagnosis | not started | | |
| C1-C7 Security + trust | not started | | |
| I1-I5 Integration | not started | | |

## Phase 0 - Setup (2026-09-28)
- Spec found at `C:\Users\HP\Downloads\Work Split Up - Minmini - DrPriyaV.docx`, copied locally to
  `docs/GreenRAG-Sec_Work_Split.docx`. The Trust Score formula in the docx is an equation object that does not
  extract as text; using `T = w1*E + w2*R + w3*C + w4*S` from the user's instructions.
- Created packages `src/backend/{forecasting,faultdiag,kb,security,trust,evaluation}`, `experiments/`,
  `results/{forecasting,retrieval,faultdiag,rag,security,ablation,cloud}/`, `tests/`.
- Installed pytest 9.1.1 via uv; added to requirements.txt (also committed the pending xgboost, matplotlib,
  statsmodels additions).
- Added `src/backend/runtime_env.py`: forces HF_HOME/TORCH_HOME under `D:\Rag\cache` (was pointing at another
  project's cache). all-MiniLM-L6-v2 re-downloaded there (~90MB).
- Regression tests: `tests/test_api_regression.py` (all existing endpoints, fakes), `test_backend_mode.py`
  (local/azure factory), `test_frontend.py` (UI assets, ids, design tokens, routes), `test_local_integration.py`
  (real MiniLM + Chroma in a temp dir), `test_ollama_integration.py` (real LLM; skips when Ollama is down).
- Result: `python -m pytest` -> 33 passed, 1 skipped (Ollama unreachable at 127.0.0.1:11434).

## Phase 0 follow-up - Ollama check (2026-09-28)
- `/api/tags` lists qwen2.5:7b-instruct, llama3.2:3b, nomic-embed-text. Ollama 0.34.3, RTX 4050 Laptop, driver 592.82.
- First run of `tests/test_ollama_integration.py` FAILED (not a timeout): Ollama's llama-server crashed while loading
  the model: `exit status 0xc0000409 ... CUDA error: shared object initialization failed` (HTTP 500).
- Reproduced by forcing cold loads (keep_alive=0): 1 failure in 5, then 0 in 8, i.e. 2/14 cold loads including the
  first call. Warm calls never failed. A direct /api/generate call right after the crash succeeded.
- Cause is in Ollama's CUDA runner (outside this repo). Fix: `LocalGenerator` retries exactly once when Ollama reports
  "llama-server process has terminated" (other errors are not retried); applies to /ask answers and numeric
  enrichment. Unit tests in `tests/test_generator_retry.py`. The integration test was not changed.
- After the fix: Ollama test passed on a cold load (9.78s); 10/10 further cold-load runs passed (7.7-27.1s; the 27s
  run may have included a retry, not verifiable from pytest output). Full suite: 38 passed.

## A1 - Solar preprocessing (2026-09-28)
- `src/backend/forecasting/preprocess.py`: load (explicit timestamp formats) -> duplicates -> time sync (15-min grid,
  +/-2 min snap) -> plant aggregation (mean over reporting inverters x 22, coverage kept, NaN below 50%) -> physical
  bounds + night consistency -> causal missing-value handling -> trailing robust-z outlier flags (values kept) ->
  causal smoothed weather copies. `chronological_split` (whole days) and `TrainOnlyScaler` (fit on train only).
  `src/backend/evaluation/provenance.py` records package versions + git commit in every report.
- Run: `python -m experiments.preprocess_solar` -> results/forecasting/preprocessing_{report_plant1,report_plant2}.json,
  preprocessing_summary.md, preprocessing_plant{1,2}.png. Reports are stamped `f64e76f-dirty` = produced by the
  uncommitted A1 code that is committed together with them.
- Measured: Plant 1: 3264 slots, 106 without generation, 82 without weather, 22 night-filled, 118 target NaN,
  2 outlier flags, DC/AC ratio 10.22 (DC scale inconsistency; DC not used), daytime corr(AC, irradiation) 0.9876.
  Plant 2: 904 partial-coverage slots, 6 target NaN, 1 night-inconsistent, 10 outlier flags, corr 0.7319.
  Plant 1 = primary modelling series; Plant 2 = harder secondary series.
- Tests: `tests/test_forecasting_preprocess.py` 19 passed, including a prefix-invariance causality test (verified to
  fail on two injected leaks: centered smoothing and whole-series outlier stats). Full suite 57 passed.
- Design notes: `docs/forecasting.md`.

## A2 - Feature engineering (2026-09-28)
- A1 reports re-generated from committed code: stamp ca8c808, report/config/package blocks identical to the
  previous run (commit 40f716c).
- **Resolution: native 15 min (no resampling). Horizon: 1 hour ahead (4 steps), fixed for all A3 models.**
  Justification in docs/forecasting.md (34 days is too short to downsample; no NWP data for longer horizons).
- `src/backend/forecasting/features.py`: origin-indexed rows; features use data <= t (lags 1/3/6/24 steps, same slot
  yesterday, rolling mean/std over 4 and 16 slots, weather at t, irradiation envelope from previous days, calendar of
  T). Measured weather at T only as opt-in `oracle_irr_target` (labelled upper bound, off by default).
- Split by target day: train 21 / val 4 / calib 4 (reserved for A4) / test 5 days. Plant 1 rows
  1901 / 379 / 384 / 478 (daytime 905 / 188 / 188 / 234).
- Tests: `tests/test_forecasting_features.py` 12 passed (future-corruption leakage test over 3 seeds + positive
  control on the oracle feature). Full suite 69 passed. First run had 5 failures: a test-helper dtype bug and a
  feature-description ordering mismatch; both fixed and re-run.
- Provenance now ignores results/ when marking code dirty (commit 6063449); A2 results stamped 6063449.
- Carried into A3: score every model on the same test rows (train-fitted imputation, no row dropping); small test set.

## Open questions / notes for the user
- The fault dataset contains wind-turbine faults; the plan's solar-inverter fault list does not match (details in B4).
- No humidity in the solar data (plan asks for it); no external source used unless approved.
