# Minmini K (23BIT0198)

**Role:** Track A - forecasting with uncertainty, and the cloud service (Azure plan A).

This folder holds copies of the files for the module area Minmini K is responsible for: solar and wind forecasting (preprocessing, features, models, CQR intervals, evaluation), the forecast API service, and the Azure integration (Key Vault secrets, Blob storage, App Insights, the I4 local-vs-Azure comparison).
It lists responsibility for an area of the project, not authorship of each line.

## How it connects to shared/

Her forecasting service is served by `shared/src/backend/main.py` (POST /forecast, GET /forecast/series) through `shared/src/backend/deps.py`; her settings live in `shared/src/backend/settings.py`; the Azure vector store `src/backend/retrieval/azure_store.py` (in this folder) is selected by `src/backend/retrieval/factory.py` in Tarunica's folder when BACKEND_MODE=azure, and itself imports Tarunica's `src/backend/retrieval/base.py` (VectorStore), `metrics.py` and `local_store.py` (MiniLM).
To run anything, use the full repository (or this folder together with `team_split/shared/`): the code imports
`src.backend.*` packages that live partly in `shared/`.

## Files included (217, 42.09 MB)

- `docs/azure.md/` (1): `azure.md`
- `docs/forecasting.md/` (1): `forecasting.md`
- `docs/team/` (1): `minmini_forecasting.md`
- `experiments/build_forecast_model.py/` (1): `build_forecast_model.py`
- `experiments/cloud_portability.py/` (1): `cloud_portability.py`
- `experiments/evaluate_forecasting.py/` (1): `evaluate_forecasting.py`
- `experiments/features_solar.py/` (1): `features_solar.py`
- `experiments/i4_harness.py/` (1): `i4_harness.py`
- `experiments/preprocess_solar.py/` (1): `preprocess_solar.py`
- `experiments/rolling_origin.py/` (1): `rolling_origin.py`
- `experiments/train_forecasters.py/` (1): `train_forecasters.py`
- `experiments/uncertainty_solar.py/` (1): `uncertainty_solar.py`
- `experiments/wind_evaluate.py/` (1): `wind_evaluate.py`
- `experiments/wind_rolling.py/` (1): `wind_rolling.py`
- `results/cloud/` (10): `.gitkeep`, `i4_equivalence.csv`, `i4_ingest.json`, `i4_latency.csv`, `i4_query_metrics.csv`, `i4_raw_azure.jsonl`, `i4_raw_local.jsonl`, `i4_status.json`, `i4_status.md`, `i4_summary.json`
- `results/forecasting/` (167): `.gitkeep`, `a3_metrics.csv`, `a3_predictions_plant1.csv`, `a3_predictions_plant2.csv`, `a3_run.json`, `a3_summary.md`, `a4_coverage_by_day.csv`, `a4_coverage_by_tod.csv`, `a4_coverage_tod.png`, `a4_interval_plant1.png`, `a4_intervals_plant1.csv`, `a4_intervals_plant2.csv`, ... (+155 more, see ../MANIFEST.md)
- `src/azure/` (2): `__init__.py`, `clients.py`
- `src/backend/` (15): `__init__.py`, `appinsights.py`, `blob.py`, `secrets.py`, `__init__.py`, `evaluation.py`, `features.py`, `metrics.py`, `models.py`, `preprocess.py`, `schemas.py`, `service.py`, ... (+3 more, see ../MANIFEST.md)
- `tests/test_azure_mode.py/` (1): `test_azure_mode.py`
- `tests/test_backend_mode.py/` (1): `test_backend_mode.py`
- `tests/test_forecast_api.py/` (1): `test_forecast_api.py`
- `tests/test_forecasting_evaluation.py/` (1): `test_forecasting_evaluation.py`
- `tests/test_forecasting_features.py/` (1): `test_forecasting_features.py`
- `tests/test_forecasting_models.py/` (1): `test_forecasting_models.py`
- `tests/test_forecasting_preprocess.py/` (1): `test_forecasting_preprocess.py`
- `tests/test_forecasting_uncertainty.py/` (1): `test_forecasting_uncertainty.py`
- `tests/test_forecasting_wind.py/` (1): `test_forecasting_wind.py`
