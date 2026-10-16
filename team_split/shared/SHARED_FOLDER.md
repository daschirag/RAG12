# shared

Code and files that no single student owns: the FastAPI app and routers, settings, serving, telemetry, provenance, orchestrator, test fixtures and cross-track tests, project-level docs and configuration.
Every student folder depends on these to run.

## Files included (55, 384.0 KB)

- `.env.example`
- `.gitignore`
- `CLAUDE.md`
- `PROGRESS.md`
- `README.md`
- `docs/GreenRAG-Sec_Work_Split.docx`
- `docs/PLAN.md`
- `docs/RESULTS.md`
- `docs/team/README.md`
- `experiments/__init__.py`
- `experiments/run_all.py`
- `pytest.ini`
- `requirements.txt`
- `results/.gitkeep`
- `results/ablation/.gitkeep`
- `results/run_all_status.json`
- `results/run_all_status.md`
- `src/__init__.py`
- `src/backend/__init__.py`
- `src/backend/api/__init__.py`
- `src/backend/api/capture_examples.py`
- `src/backend/api/catalog.py`
- `src/backend/api/examples.py`
- `src/backend/api/openapi_examples.json`
- `src/backend/api/ops.py`
- `src/backend/api/results.py`
- `src/backend/api_support.py`
- `src/backend/config/__init__.py`
- `src/backend/config/domain_manifest.py`
- `src/backend/config/domain_manifests/fault_diagnosis.yaml`
- `src/backend/config/domain_manifests/knowledge_base.yaml`
- `src/backend/config/domain_manifests/solar.yaml`
- `src/backend/config/domain_manifests/wind.yaml`
- `src/backend/deps.py`
- `src/backend/evaluation/__init__.py`
- `src/backend/evaluation/provenance.py`
- `src/backend/main.py`
- `src/backend/orchestrator.py`
- `src/backend/runtime_env.py`
- `src/backend/serving/__init__.py`
- `src/backend/serving/generator.py`
- `src/backend/serving/router.py`
- `src/backend/serving/tiered_router.py`
- `src/backend/settings.py`
- `src/backend/telemetry/__init__.py`
- `src/backend/telemetry/monitor.py`
- `tests/conftest.py`
- `tests/test_api_regression.py`
- `tests/test_catalog_lab_api.py`
- `tests/test_generator_retry.py`
- `tests/test_local_integration.py`
- `tests/test_ollama_integration.py`
- `tests/test_orchestrator.py`
- `tests/test_platform_api.py`
- `tests/test_run_all.py`
