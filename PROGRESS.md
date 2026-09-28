# GreenRAG-Sec progress

Phase list: `docs/PLAN.md`. Rules: `CLAUDE.md`.

| Phase | Status | Tests | Commit |
|---|---|---|---|
| 0 Setup | done | 38 passed, 0 skipped (with Ollama running) | e043ac1 + Ollama fix commit |
| A1-A6 Forecasting | not started | | |
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

## Open questions / notes for the user
- The fault dataset contains wind-turbine faults; the plan's solar-inverter fault list does not match (details in B4).
- No humidity in the solar data (plan asks for it); no external source used unless approved.
