# Tarunica S (23BIT0185)

**Role:** Track B - knowledge base, hybrid RAG, fault diagnosis, and the frontend.

This folder holds copies of the files for the module area Tarunica S is responsible for: the knowledge base, chunking, hybrid retrieval with rerank, Wind Farm A fault detection and evidence-grounded diagnosis, the RAG evaluation, and the two consoles (src/console, src/frontend).
It lists responsibility for an area of the project, not authorship of each line.

## How it connects to shared/

Her retrieval and diagnosis modules are served by `shared/src/backend/main.py` (/query, /ask, /diagnose) and the catalog router `shared/src/backend/api/catalog.py`; the consoles call the shared API; the secure pipeline in Pritika's folder builds on her retrieval and knowledge base. Dependency: her `src/backend/retrieval/factory.py` imports `azure_store` (AzureVectorStore) when BACKEND_MODE=azure; that file is in Minmini's folder (`minmini/src/backend/retrieval/azure_store.py`).
To run anything, use the full repository (or this folder together with `team_split/shared/`): the code imports
`src.backend.*` packages that live partly in `shared/`.

## Files included (155, 6.99 MB)

- `docs/fault_diagnosis.md/` (1): `fault_diagnosis.md`
- `docs/knowledge_base.md/` (1): `knowledge_base.md`
- `docs/rag_evaluation.md/` (1): `rag_evaluation.md`
- `docs/screenshots/` (31): `about.png`, `about_dark.png`, `about_mobile.png`, `assistant.png`, `assistant_dark.png`, `assistant_mobile.png`, `console_check.json`, `console_check_dark.json`, `console_check_mobile.json`, `diagnosis.png`, `diagnosis_dark.png`, `diagnosis_mobile.png`, ... (+19 more, see ../MANIFEST.md)
- `docs/team/` (1): `tarunica_rag_diagnosis.md`
- `experiments/b6_followup.py/` (1): `b6_followup.py`
- `experiments/b6_followup_report.py/` (1): `b6_followup_report.py`
- `experiments/chunking_comparison.py/` (1): `chunking_comparison.py`
- `experiments/expansion_check.py/` (1): `expansion_check.py`
- `experiments/fault_diagnosis.py/` (1): `fault_diagnosis.py`
- `experiments/fault_rag.py/` (1): `fault_rag.py`
- `experiments/fault_threshold.py/` (1): `fault_threshold.py`
- `experiments/rag_evaluation.py/` (1): `rag_evaluation.py`
- `experiments/retrieval_comparison.py/` (1): `retrieval_comparison.py`
- `experiments/wind_diagnose.py/` (1): `wind_diagnose.py`
- `results/faultdiag/` (21): `.gitkeep`, `b4_attribution.csv`, `b4_event_metrics.csv`, `b4_event_predictions.csv`, `b4_if_threshold.csv`, `b4_if_threshold.md`, `b4_row_metrics.csv`, `b4_run.json`, `b4_summary.md`, `b5_reports.jsonl`, `b5_results.csv`, `b5_run.json`, ... (+9 more, see ../MANIFEST.md)
- `results/kb/` (15): `b2_chunk_stats.csv`, `b2_chunking_metrics.csv`, `b2_chunking_summary.md`, `b2_paired_bootstrap.csv`, `b2_run.json`, `b3_classifier.csv`, `b3_paired_bootstrap.csv`, `b3_retrieval_metrics.csv`, `b3_retrieval_summary.md`, `b3_run.json`, `b6_expansion_check.csv`, `b6_expansion_check.md`, ... (+3 more, see ../MANIFEST.md)
- `results/rag/` (20): `.gitkeep`, `answers.jsonl`, `b6_per_answer.csv`, `b6_run.json`, `b6_summary.md`, `judge_calibration.json`, `judgements.jsonl`, `answers.jsonl`, `b6_followup_deterministic.md`, `b6_followup_summary.md`, `b6_followup_template_claims.csv`, `b6_run2_per_answer.csv`, ... (+8 more, see ../MANIFEST.md)
- `results/retrieval/` (1): `.gitkeep`
- `scripts/capture_screenshots.py/` (1): `capture_screenshots.py`
- `src/backend/` (29): `rag_eval.py`, `rag_systems.py`, `__init__.py`, `data.py`, `diagnosis.py`, `models.py`, `service.py`, `__init__.py`, `chunkers.py`, `loaders.py`, `pipeline.py`, `scada_ingest.py`, ... (+17 more, see ../MANIFEST.md)
- `src/console/` (12): `app.css`, `index.html`, `api.js`, `app.js`, `charts.js`, `about.js`, `assistant.js`, `diagnosis.js`, `forecast.js`, `lab.js`, `results.js`, `ui.js`
- `src/frontend/` (3): `app.js`, `index.html`, `styles.css`
- `tests/test_diagnose_api.py/` (1): `test_diagnose_api.py`
- `tests/test_fault_rag.py/` (1): `test_fault_rag.py`
- `tests/test_faultdiag.py/` (1): `test_faultdiag.py`
- `tests/test_frontend.py/` (1): `test_frontend.py`
- `tests/test_hybrid_retrieval.py/` (1): `test_hybrid_retrieval.py`
- `tests/test_kb.py/` (1): `test_kb.py`
- `tests/test_kb_chunking.py/` (1): `test_kb_chunking.py`
- `tests/test_rag_eval.py/` (1): `test_rag_eval.py`
