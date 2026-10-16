# Pritika Kasinathan (23BIT0232)

**Role:** Track C - security and trust, and documentation.

This folder holds copies of the files for the module area Pritika Kasinathan is responsible for: the threat model, attack simulator, detectors, secure retrieval pipeline, trust-aware decision engine, the security evaluation and lab API, and the report material.
It lists responsibility for an area of the project, not authorship of each line.

## How it connects to shared/

Her secure pipeline and decision engine are served by `shared/src/backend/main.py` (/secure-ask, /decision) and her `src/backend/api/lab.py`; they use Tarunica's retrieval and knowledge base and the forecasts from Minmini's service; the orchestrator in `shared/` calls them for config F.
To run anything, use the full repository (or this folder together with `team_split/shared/`): the code imports
`src.backend.*` packages that live partly in `shared/`.

## Files included (100, 7.40 MB)

- `docs/attack_simulation.md/` (1): `attack_simulation.md`
- `docs/decision_engine.md/` (1): `decision_engine.md`
- `docs/detectors.md/` (1): `detectors.md`
- `docs/screenshots/` (5): `01_assistant_numeric.png`, `02_forecast_interval.png`, `03_fault_diagnosis.png`, `04_security_lab.png`, `05_results_status.png`
- `docs/secure_retrieval.md/` (1): `secure_retrieval.md`
- `docs/security_evaluation.md/` (1): `security_evaluation.md`
- `docs/team/` (1): `pritika_security_trust.md`
- `docs/threat_model.md/` (1): `threat_model.md`
- `experiments/attack_simulation.py/` (1): `attack_simulation.py`
- `experiments/decision_engine_eval.py/` (1): `decision_engine_eval.py`
- `experiments/detector_evaluation.py/` (1): `detector_evaluation.py`
- `experiments/secure_pipeline_eval.py/` (1): `secure_pipeline_eval.py`
- `experiments/security_evaluation.py/` (1): `security_evaluation.py`
- `experiments/security_evaluation_report.py/` (1): `security_evaluation_report.py`
- `literature_survey/.gitkeep/` (1): `.gitkeep`
- `presentation/.gitkeep/` (1): `.gitkeep`
- `results/report_facts.md/` (1): `report_facts.md`
- `results/report_facts2.md/` (1): `report_facts2.md`
- `results/report_numbers.md/` (1): `report_numbers.md`
- `results/security/` (46): `.gitkeep`, `c2_attack_docs.jsonl`, `c2_fault_detection.csv`, `c2_forecast.csv`, `c2_knowledge.csv`, `c2_knowledge_items.csv`, `c2_knowledge_items_llama3.2-3b.csv`, `c2_knowledge_llama3.2-3b.csv`, `c2_knowledge_raw.jsonl`, `c2_knowledge_raw_llama3.2-3b.jsonl`, `c2_manifest.json`, `c2_run_knowledge.json`, ... (+34 more, see ../MANIFEST.md)
- `results/trust/` (9): `c5_decisions.csv`, `c5_raw_llama3.2-3b.jsonl`, `c5_run.json`, `c5_sensitivity_grid.csv`, `c5_sensitivity_thresholds.csv`, `c5_summary.md`, `c5_unsafe_constraint_c2.csv`, `c5_weight_sweep.csv`, `c5_weight_sweep.png`
- `src/backend/` (15): `lab.py`, `__init__.py`, `attack_eval.py`, `attacks.py`, `auth.py`, `doc_trust.py`, `evidence_filter.py`, `injection_detector.py`, `injection_eval_set.py`, `scada_anomaly.py`, `secure_pipeline.py`, `__init__.py`, ... (+3 more, see ../MANIFEST.md)
- `tests/test_attacks.py/` (1): `test_attacks.py`
- `tests/test_decision_engine.py/` (1): `test_decision_engine.py`
- `tests/test_detectors.py/` (1): `test_detectors.py`
- `tests/test_secure_api.py/` (1): `test_secure_api.py`
- `tests/test_secure_pipeline.py/` (1): `test_secure_pipeline.py`
- `tests/test_security_evaluation.py/` (1): `test_security_evaluation.py`
- `tests/test_threat_model.py/` (1): `test_threat_model.py`

Redacted in the copy (personal data removed): `results/report_facts.md`
