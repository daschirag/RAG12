# GreenRAG-Sec progress

Phase list: `docs/PLAN.md`. Rules: `CLAUDE.md`.

| Phase | Status | Tests | Commit |
|---|---|---|---|
| 0 Setup | done | 33 passed, 1 skipped (Ollama not running) | see git log "Phase 0" |
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

## Open questions / notes for the user
- Ollama was not running during Phase 0, so the real-LLM `/ask` test was skipped.
- The fault dataset contains wind-turbine faults; the plan's solar-inverter fault list does not match (details in B4).
- No humidity in the solar data (plan asks for it); no external source used unless approved.
