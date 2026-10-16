# team_split manifest

Copies of the repository files, one owner each (user's assignment, overridden by `docs/team/*.md` where they
differ). Originals are untouched; no git command was run. `purpose` = the file's own docstring or heading, or
its kind of data.

| file path | owner | one-line purpose |
|---|---|---|
| `docs/azure.md` | minmini | Azure mode (plan A) |
| `docs/forecasting.md` | minmini | Track A - Forecasting: design notes |
| `docs/team/minmini_forecasting.md` | minmini | Track A - Forecasting with uncertainty (Minmini K, 23BIT0198) |
| `experiments/build_forecast_model.py` | minmini | A6: build the deployed forecast models used by POST /forecast. |
| `experiments/cloud_portability.py` | minmini | I4 - Cloud portability: is the Azure configuration usable from here? (never simulated) |
| `experiments/evaluate_forecasting.py` | minmini | A5: evaluation report for Track A (fixed split + rolling origin + bootstrap). |
| `experiments/features_solar.py` | minmini | A2: build 1-hour-ahead features for both plants and describe them. |
| `experiments/i4_harness.py` | minmini | I4 - local vs Azure (plan A): same requests through the app in both modes; only measured values are written. |
| `experiments/preprocess_solar.py` | minmini | A1: run solar preprocessing for both plants and save reports. |
| `experiments/rolling_origin.py` | minmini | A5: rolling-origin (expanding-window) evaluation of point models and interval methods. |
| `experiments/train_forecasters.py` | minmini | A3: train and compare 1-hour-ahead forecasters on both solar plants. |
| `experiments/uncertainty_solar.py` | minmini | A4: night-rule decision (val only) and prediction intervals (calib only) for both plants. |
| `experiments/wind_evaluate.py` | minmini | Wind T1 replication: evaluation report from the rolling-origin month folds. |
| `experiments/wind_rolling.py` | minmini | Wind T1 replication of Track A: rolling-origin evaluation by month (resumable). |
| `results/cloud/.gitkeep` | minmini | empty folder placeholder |
| `results/cloud/i4_equivalence.csv` | minmini | result table (cloud) |
| `results/cloud/i4_ingest.json` | minmini | result/record (JSON) (cloud) |
| `results/cloud/i4_latency.csv` | minmini | result table (cloud) |
| `results/cloud/i4_query_metrics.csv` | minmini | result table (cloud) |
| `results/cloud/i4_raw_azure.jsonl` | minmini | raw outputs (JSON lines) (cloud) |
| `results/cloud/i4_raw_local.jsonl` | minmini | raw outputs (JSON lines) (cloud) |
| `results/cloud/i4_status.json` | minmini | result/record (JSON) (cloud) |
| `results/cloud/i4_status.md` | minmini | I4 cloud portability: local vs Azure (plan A) |
| `results/cloud/i4_summary.json` | minmini | result/record (JSON) (cloud) |
| `results/forecasting/.gitkeep` | minmini | empty folder placeholder |
| `results/forecasting/a3_metrics.csv` | minmini | result table (forecasting) |
| `results/forecasting/a3_predictions_plant1.csv` | minmini | result table (forecasting) |
| `results/forecasting/a3_predictions_plant2.csv` | minmini | result table (forecasting) |
| `results/forecasting/a3_run.json` | minmini | result/record (JSON) (forecasting) |
| `results/forecasting/a3_summary.md` | minmini | A3 forecasting model comparison (1 hour ahead, 15-min resolution) |
| `results/forecasting/a4_coverage_by_day.csv` | minmini | result table (forecasting) |
| `results/forecasting/a4_coverage_by_tod.csv` | minmini | result table (forecasting) |
| `results/forecasting/a4_coverage_tod.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/a4_interval_plant1.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/a4_intervals_plant1.csv` | minmini | result table (forecasting) |
| `results/forecasting/a4_intervals_plant2.csv` | minmini | result table (forecasting) |
| `results/forecasting/a4_metrics.csv` | minmini | result table (forecasting) |
| `results/forecasting/a4_night_rule.csv` | minmini | result table (forecasting) |
| `results/forecasting/a4_run.json` | minmini | result/record (JSON) (forecasting) |
| `results/forecasting/a4_summary.md` | minmini | A4 uncertainty-aware forecasting (1 hour ahead) |
| `results/forecasting/a5_bootstrap_interval.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_bootstrap_interval.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/a5_bootstrap_point.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_coverage_ci.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_error_by_hour.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_error_by_hour.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/a5_error_vs_clearness.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_error_vs_clearness.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/a5_fixed_split.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant1_2020-06-06.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant1_2020-06-07.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant1_2020-06-08.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant1_2020-06-09.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant1_2020-06-10.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant1_2020-06-11.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant1_2020-06-12.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant1_2020-06-13.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant1_2020-06-14.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant1_2020-06-15.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant1_2020-06-16.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant1_2020-06-17.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant2_2020-06-06.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant2_2020-06-07.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant2_2020-06-08.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant2_2020-06-09.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant2_2020-06-10.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant2_2020-06-11.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant2_2020-06-12.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant2_2020-06-13.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant2_2020-06-14.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant2_2020-06-15.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant2_2020-06-16.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/plant2_2020-06-17.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_folds/run_log.json` | minmini | result/record (JSON) (forecasting) |
| `results/forecasting/a5_rolling_coverage.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/a5_rolling_interval_by_day.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_rolling_mae.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/a5_rolling_point_by_day.csv` | minmini | result table (forecasting) |
| `results/forecasting/a5_summary.md` | minmini | A5 forecasting evaluation (1 hour ahead, strict night rule everywhere) |
| `results/forecasting/a6_model_card.json` | minmini | result/record (JSON) (forecasting) |
| `results/forecasting/archive_night_rule_0.05/README.md` | minmini | Archived results (night rule envelope <= 0.05) |
| `results/forecasting/archive_night_rule_0.05/a3_metrics.csv` | minmini | result table (forecasting) |
| `results/forecasting/archive_night_rule_0.05/a3_predictions_plant1.csv` | minmini | result table (forecasting) |
| `results/forecasting/archive_night_rule_0.05/a3_predictions_plant2.csv` | minmini | result table (forecasting) |
| `results/forecasting/archive_night_rule_0.05/a3_run.json` | minmini | result/record (JSON) (forecasting) |
| `results/forecasting/archive_night_rule_0.05/a3_summary.md` | minmini | A3 forecasting model comparison (1 hour ahead, 15-min resolution) |
| `results/forecasting/archive_night_rule_0.05/a4_coverage_by_day.csv` | minmini | result table (forecasting) |
| `results/forecasting/archive_night_rule_0.05/a4_coverage_by_tod.csv` | minmini | result table (forecasting) |
| `results/forecasting/archive_night_rule_0.05/a4_coverage_tod.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/archive_night_rule_0.05/a4_interval_plant1.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/archive_night_rule_0.05/a4_intervals_plant1.csv` | minmini | result table (forecasting) |
| `results/forecasting/archive_night_rule_0.05/a4_intervals_plant2.csv` | minmini | result table (forecasting) |
| `results/forecasting/archive_night_rule_0.05/a4_metrics.csv` | minmini | result table (forecasting) |
| `results/forecasting/archive_night_rule_0.05/a4_night_rule.csv` | minmini | result table (forecasting) |
| `results/forecasting/archive_night_rule_0.05/a4_run.json` | minmini | result/record (JSON) (forecasting) |
| `results/forecasting/archive_night_rule_0.05/a4_summary.md` | minmini | A4 uncertainty-aware forecasting (1 hour ahead) |
| `results/forecasting/features_plant1.json` | minmini | result/record (JSON) (forecasting) |
| `results/forecasting/features_plant2.json` | minmini | result/record (JSON) (forecasting) |
| `results/forecasting/features_summary.md` | minmini | A2 feature engineering summary |
| `results/forecasting/preprocessing_plant1.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/preprocessing_plant2.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/preprocessing_report_plant1.json` | minmini | result/record (JSON) (forecasting) |
| `results/forecasting/preprocessing_report_plant2.json` | minmini | result/record (JSON) (forecasting) |
| `results/forecasting/preprocessing_summary.md` | minmini | A1 solar preprocessing summary |
| `results/forecasting/wind/bootstrap_interval.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/bootstrap_point.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/coverage_ci.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/base.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/daily_persistence__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/lstm__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/lstm__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/lstm__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/persistence__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/random_forest__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/random_forest__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/random_forest__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/tcn__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/tcn__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/tcn__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/xgboost__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/xgboost__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/xgboost__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/xgboost_oracle_weather__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/xgboost_oracle_weather__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-09/xgboost_oracle_weather__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/base.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/daily_persistence__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/lstm__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/lstm__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/lstm__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/persistence__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/random_forest__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/random_forest__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/random_forest__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/tcn__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/tcn__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/tcn__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/xgboost__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/xgboost__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/xgboost__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/xgboost_oracle_weather__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/xgboost_oracle_weather__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-10/xgboost_oracle_weather__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/base.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/daily_persistence__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/lstm__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/lstm__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/lstm__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/persistence__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/random_forest__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/random_forest__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/random_forest__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/tcn__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/tcn__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/tcn__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/xgboost__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/xgboost__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/xgboost__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/xgboost_oracle_weather__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/xgboost_oracle_weather__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-11/xgboost_oracle_weather__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/base.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/daily_persistence__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/lstm__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/lstm__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/lstm__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/persistence__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/random_forest__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/random_forest__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/random_forest__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/tcn__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/tcn__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/tcn__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/xgboost__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/xgboost__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/xgboost__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/xgboost_oracle_weather__seed0.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/xgboost_oracle_weather__seed1.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/2018-12/xgboost_oracle_weather__seed2.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/folds/run_log_20260929_230222.json` | minmini | result/record (JSON) (forecasting) |
| `results/forecasting/wind/interval_by_day.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/interval_metrics.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/point_by_day.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/point_metrics.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/preprocessing_report.json` | minmini | result/record (JSON) (forecasting) |
| `results/forecasting/wind/wind_coverage_regime.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/wind/wind_diagnosis.csv` | minmini | result table (forecasting) |
| `results/forecasting/wind/wind_diagnosis.md` | minmini | Wind T1: where persistence wins (diagnostic, test rows of the 4 rolling-origin months) |
| `results/forecasting/wind/wind_error_by_speed.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/wind/wind_monthly_mae.png` | minmini | figure/screenshot (forecasting) |
| `results/forecasting/wind/wind_summary.md` | minmini | Wind T1 replication of Track A (1 hour ahead, 10-min resolution) |
| `src/azure/__init__.py` | minmini | empty file |
| `src/azure/clients.py` | minmini | empty file |
| `src/backend/cloud/__init__.py` | minmini | Azure integration for BACKEND_MODE=azure (plan A): Key Vault secrets, Blob storage, Application Insights. |
| `src/backend/cloud/appinsights.py` | minmini | Send the structured request log (greenrag.access) to Application Insights. |
| `src/backend/cloud/blob.py` | minmini | Blob storage for the knowledge base and results (Entra auth; the account has shared keys disabled). |
| `src/backend/cloud/secrets.py` | minmini | Secret lookup: environment (or the git-ignored .env) first, then Azure Key Vault. Values are never logged. |
| `src/backend/forecasting/__init__.py` | minmini | Track A: solar/wind generation forecasting with uncertainty (Minmini). |
| `src/backend/forecasting/evaluation.py` | minmini | A5 - Rolling-origin folds and paired day-level bootstrap. |
| `src/backend/forecasting/features.py` | minmini | A2 - Feature engineering for 1-hour-ahead plant AC power forecasting. |
| `src/backend/forecasting/metrics.py` | minmini | Point-forecast error metrics (A3/A5). |
| `src/backend/forecasting/models.py` | minmini | A3 - Forecasting models for 1-hour-ahead plant AC power. |
| `src/backend/forecasting/preprocess.py` | minmini | A1 - Solar preprocessing pipeline (plant level). |
| `src/backend/forecasting/schemas.py` | minmini | Pydantic schemas for the forecasting API (POST /forecast). |
| `src/backend/forecasting/service.py` | minmini | A6 - Forecasting service behind POST /forecast (solar plants 1-2 and wind turbine T1). |
| `src/backend/forecasting/uncertainty.py` | minmini | A4 - Prediction intervals for 1-hour-ahead plant AC power. |
| `src/backend/forecasting/wind.py` | minmini | Track A replicated on wind turbine T1 (dataset/wind/T1.csv). |
| `src/backend/retrieval/azure_store.py` | minmini | Azure vector store: Azure AI Search + embeddings from local MiniLM (plan A) or Azure OpenAI. |
| `tests/test_azure_mode.py` | minmini | BACKEND_MODE=azure plan A with mocked Azure clients only: no network, no Azure calls, no real secrets. |
| `tests/test_backend_mode.py` | minmini | BACKEND_MODE local/azure abstraction for vector stores and generators. |
| `tests/test_forecast_api.py` | minmini | A6: POST /forecast (TestClient), request/response schemas, service consistency and causality. |
| `tests/test_forecasting_evaluation.py` | minmini | A5: rolling-origin folds (chronology, no overlap) and the paired day bootstrap. |
| `tests/test_forecasting_features.py` | minmini | A2 features: exact alignment, no future information, split by target day. |
| `tests/test_forecasting_models.py` | minmini | A3 models: train-only imputation, same scored rows, causal windows, determinism. |
| `tests/test_forecasting_preprocess.py` | minmini | A1 preprocessing: each step on small synthetic frames, causality, split, scaler, real data. |
| `tests/test_forecasting_uncertainty.py` | minmini | A4: conformal quantile, known-coverage sanity checks, calibration/test separation, determinism. |
| `tests/test_forecasting_wind.py` | minmini | Wind T1 replication: preprocessing, features, folds, generalized models/intervals, API. |
| `docs/attack_simulation.md` | pritika | C2 - Attack simulator |
| `docs/decision_engine.md` | pritika | C5 - Trust-aware decision engine |
| `docs/detectors.md` | pritika | C3 - Detectors |
| `docs/screenshots/report/01_assistant_numeric.png` | pritika | figure/screenshot (screenshots) |
| `docs/screenshots/report/02_forecast_interval.png` | pritika | figure/screenshot (screenshots) |
| `docs/screenshots/report/03_fault_diagnosis.png` | pritika | figure/screenshot (screenshots) |
| `docs/screenshots/report/04_security_lab.png` | pritika | figure/screenshot (screenshots) |
| `docs/screenshots/report/05_results_status.png` | pritika | figure/screenshot (screenshots) |
| `docs/secure_retrieval.md` | pritika | C4 - Secure retrieval pipeline |
| `docs/security_evaluation.md` | pritika | C6 - Security evaluation |
| `docs/team/pritika_security_trust.md` | pritika | Track C - Security and trust (Pritika Kasinathan, 23BIT0232) |
| `docs/threat_model.md` | pritika | GreenRAG-Sec threat model (C1) |
| `experiments/attack_simulation.py` | pritika | C2: attack simulator - effect of attacks on the undefended system (local, this repo's data only). |
| `experiments/decision_engine_eval.py` | pritika | C5: trust-aware decision engine on real pipeline outputs, with a weight sensitivity analysis. |
| `experiments/detector_evaluation.py` | pritika | C3: evaluate the detectors against the attacks of C2 (no Ollama needed). |
| `experiments/secure_pipeline_eval.py` | pritika | C4: the secure retrieval pipeline end to end, on clean and attacked KB stores (resumable). |
| `experiments/security_evaluation.py` | pritika | C6: security evaluation of the deployed generator (qwen2.5:7b-instruct) - undefended vs secure vs secure + eng |
| `experiments/security_evaluation_report.py` | pritika | C6 report: attack success per configuration, injection defence, utility cost, decision engine, forecast C. |
| `literature_survey/.gitkeep` | pritika | empty folder placeholder |
| `presentation/.gitkeep` | pritika | empty folder placeholder |
| `results/report_facts.md` | pritika | Report facts (compiled 2026-10-16 from this repository; nothing estimated) (redacted in copy) |
| `results/report_facts2.md` | pritika | Report facts 2 (compiled 2026-10-16 from the repository; read only, nothing estimated) |
| `results/report_numbers.md` | pritika | Report numbers (compiled from existing files only) |
| `results/security/.gitkeep` | pritika | empty folder placeholder |
| `results/security/c2_attack_docs.jsonl` | pritika | raw outputs (JSON lines) (security) |
| `results/security/c2_fault_detection.csv` | pritika | result table (security) |
| `results/security/c2_forecast.csv` | pritika | result table (security) |
| `results/security/c2_knowledge.csv` | pritika | result table (security) |
| `results/security/c2_knowledge_items.csv` | pritika | result table (security) |
| `results/security/c2_knowledge_items_llama3.2-3b.csv` | pritika | result table (security) |
| `results/security/c2_knowledge_llama3.2-3b.csv` | pritika | result table (security) |
| `results/security/c2_knowledge_raw.jsonl` | pritika | raw outputs (JSON lines) (security) |
| `results/security/c2_knowledge_raw_llama3.2-3b.jsonl` | pritika | raw outputs (JSON lines) (security) |
| `results/security/c2_manifest.json` | pritika | result/record (JSON) (security) |
| `results/security/c2_run_knowledge.json` | pritika | result/record (JSON) (security) |
| `results/security/c2_run_knowledge_llama3.2-3b.json` | pritika | result/record (JSON) (security) |
| `results/security/c2_run_telemetry.json` | pritika | result/record (JSON) (security) |
| `results/security/c2_summary.md` | pritika | C2 - Attack simulation: effect on the undefended system |
| `results/security/c3_filter_items.csv` | pritika | result table (security) |
| `results/security/c3_filter_scored.csv` | pritika | result table (security) |
| `results/security/c3_injection.csv` | pritika | result table (security) |
| `results/security/c3_run.json` | pritika | result/record (JSON) (security) |
| `results/security/c3_summary.md` | pritika | C3 - Detectors evaluated against the C2 attacks |
| `results/security/c3_telemetry_days.csv` | pritika | result table (security) |
| `results/security/c3_trust_contexts.csv` | pritika | result table (security) |
| `results/security/c3_trust_docs.csv` | pritika | result table (security) |
| `results/security/c4_raw_llama3.2-3b.jsonl` | pritika | raw outputs (JSON lines) (security) |
| `results/security/c4_raw_none.jsonl` | pritika | raw outputs (JSON lines) (security) |
| `results/security/c4_run_llama3.2-3b.json` | pritika | result/record (JSON) (security) |
| `results/security/c4_run_none.json` | pritika | result/record (JSON) (security) |
| `results/security/c4_summary.md` | pritika | C4 - Secure retrieval pipeline: attacked KB stores, unprotected (C2) vs secure (C4) |
| `results/security/c6_attack_success.json` | pritika | result/record (JSON) (security) |
| `results/security/c6_decision_engine.json` | pritika | result/record (JSON) (security) |
| `results/security/c6_forecast_confidence.json` | pritika | result/record (JSON) (security) |
| `results/security/c6_heldout_items.csv` | pritika | result table (security) |
| `results/security/c6_heldout_raw_qwen2.5-7b-instruct.jsonl` | pritika | raw outputs (JSON lines) (security) |
| `results/security/c6_injection_defence.json` | pritika | result/record (JSON) (security) |
| `results/security/c6_items.csv` | pritika | result table (security) |
| `results/security/c6_judgements.jsonl` | pritika | raw outputs (JSON lines) (security) |
| `results/security/c6_raw_qwen2.5-7b-instruct.jsonl` | pritika | raw outputs (JSON lines) (security) |
| `results/security/c6_run.json` | pritika | result/record (JSON) (security) |
| `results/security/c6_secure_answers_qwen2.5-7b-instruct.jsonl` | pritika | raw outputs (JSON lines) (security) |
| `results/security/c6_sensitivity_floors.csv` | pritika | result table (security) |
| `results/security/c6_sensitivity_grid.csv` | pritika | result table (security) |
| `results/security/c6_sensitivity_thresholds.csv` | pritika | result table (security) |
| `results/security/c6_summary.md` | pritika | C6 - Security evaluation (deployed generator qwen2.5:7b-instruct) |
| `results/security/c6_telemetry_days.csv` | pritika | result table (security) |
| `results/security/c6_telemetry_rerun_metrics.csv` | pritika | result table (security) |
| `results/security/c6_utility.json` | pritika | result/record (JSON) (security) |
| `results/trust/c5_decisions.csv` | pritika | result table (trust) |
| `results/trust/c5_raw_llama3.2-3b.jsonl` | pritika | raw outputs (JSON lines) (trust) |
| `results/trust/c5_run.json` | pritika | result/record (JSON) (trust) |
| `results/trust/c5_sensitivity_grid.csv` | pritika | result table (trust) |
| `results/trust/c5_sensitivity_thresholds.csv` | pritika | result table (trust) |
| `results/trust/c5_summary.md` | pritika | C5 - Trust-aware decision engine on real pipeline outputs |
| `results/trust/c5_unsafe_constraint_c2.csv` | pritika | result table (trust) |
| `results/trust/c5_weight_sweep.csv` | pritika | result table (trust) |
| `results/trust/c5_weight_sweep.png` | pritika | figure/screenshot (trust) |
| `src/backend/api/lab.py` | pritika | Security lab: run the known C2 attacks against the live C3 detectors and C5 engine (all attack data synthetic) |
| `src/backend/security/__init__.py` | pritika | Track C: attack detectors, secure retrieval and input security (Pritika). |
| `src/backend/security/attack_eval.py` | pritika | C2 - Deterministic scoring of attack outcomes (no LLM judge). |
| `src/backend/security/attacks.py` | pritika | C2 - Attack simulator (local, on this repo's own data only). |
| `src/backend/security/auth.py` | pritika | empty file |
| `src/backend/security/doc_trust.py` | pritika | C3 - Document trust verification for knowledge-base evidence. |
| `src/backend/security/evidence_filter.py` | pritika | C3 - Evidence filter between retrieval and generation. |
| `src/backend/security/injection_detector.py` | pritika | C3 - Rule-based prompt-injection detector for user queries and retrieved text. |
| `src/backend/security/injection_eval_set.py` | pritika | C3 - Held-out prompt-injection evaluation texts (SYNTHETIC, written for this project). |
| `src/backend/security/scada_anomaly.py` | pritika | C3 - Telemetry anomaly / spoofing detector for the solar plants and the wind turbine. |
| `src/backend/security/secure_pipeline.py` | pritika | C4 - Secure retrieval pipeline. |
| `src/backend/trust/__init__.py` | pritika | Track C: trust-aware decision engine and trust score (Pritika). |
| `src/backend/trust/decision.py` | pritika | C5 - Trust-aware decision engine. |
| `src/backend/trust/service.py` | pritika | C7 - Service behind POST /decision: run a served pipeline, then the C5 trust-aware decision engine. |
| `src/backend/trust/signals.py` | pritika | C5 - Measured inputs of the trust score, loaded from the repository's saved results (never hard-coded). |
| `tests/test_attacks.py` | pritika | C2: attack simulator - attacks change only what they claim to, attack documents blend in, scoring is exact. |
| `tests/test_decision_engine.py` | pritika | C5: trust-aware decision engine (the LLM recommends; only this pipeline decides). |
| `tests/test_detectors.py` | pritika | C3: telemetry detector, prompt-injection detector, document trust and evidence filter. |
| `tests/test_secure_api.py` | pritika | C7: POST /secure-ask (C4 secure pipeline) and POST /decision (C5 engine) through TestClient with injected |
| `tests/test_secure_pipeline.py` | pritika | C4: secure retrieval pipeline (input security -> injection detection -> retrieval -> trust -> filter -> LLM). |
| `tests/test_security_evaluation.py` | pritika | C6: security evaluation helpers - intervals, schema conversion, markers, forecast gating, judge metrics. |
| `tests/test_threat_model.py` | pritika | C1: docs/threat_model.md is complete, internally consistent and grounded in files that exist. |
| `.env.example` | shared | .example |
| `.gitignore` | shared | file |
| `CLAUDE.md` | shared | GreenRAG-Sec: working rules for Claude |
| `PROGRESS.md` | shared | GreenRAG-Sec progress (redacted in copy) |
| `README.md` | shared | GreenRAG-Sec |
| `docs/GreenRAG-Sec_Work_Split.docx` | shared | Word document |
| `docs/PLAN.md` | shared | GreenRAG-Sec implementation plan |
| `docs/RESULTS.md` | shared | GreenRAG-Sec results |
| `docs/team/README.md` | shared | GreenRAG-Sec team documentation |
| `experiments/__init__.py` | shared | Reproducible experiment entry points (python -m experiments.<name>). |
| `experiments/run_all.py` | shared | I3 - Experiments 1-7 (+ ablation): status of every step, reusing finished results, never re-running them. |
| `pytest.ini` | shared | pytest configuration |
| `requirements.txt` | shared | text |
| `results/.gitkeep` | shared | empty folder placeholder |
| `results/ablation/.gitkeep` | shared | empty folder placeholder |
| `results/run_all_status.json` | shared | result/record (JSON) |
| `results/run_all_status.md` | shared | run_all status |
| `src/__init__.py` | shared | GreenRAG source package root (enables ``python -m src.backend...``). |
| `src/backend/__init__.py` | shared | GreenRAG backend package. |
| `src/backend/api/__init__.py` | shared | API routers (system, results, catalog, security lab) and the recorded OpenAPI examples. |
| `src/backend/api/capture_examples.py` | shared | Record real responses of the running API as OpenAPI examples (src/backend/api/openapi_examples.json). |
| `src/backend/api/catalog.py` | shared | Lookup endpoints the console needs: fault events, forecast assets and day curves, suggested questions and the |
| `src/backend/api/examples.py` | shared | OpenAPI response examples: real responses recorded from the running system, never written by hand. |
| `src/backend/api/openapi_examples.json` | shared | result/record (JSON) |
| `src/backend/api/ops.py` | shared | Operations: GET /ready, POST/GET /warmup, GET /system/info. |
| `src/backend/api/results.py` | shared | GET /results/summary and GET /results/summary/{track}: measured numbers read from results/ (never typed in). |
| `src/backend/api_support.py` | shared | Cross-cutting API behaviour: one JSON error format, request ids and a structured request log. |
| `src/backend/config/__init__.py` | shared | Public exports for the GreenRAG config package. |
| `src/backend/config/domain_manifest.py` | shared | Domain manifest schema and loaders for GreenRAG. |
| `src/backend/config/domain_manifests/fault_diagnosis.yaml` | shared | .yaml |
| `src/backend/config/domain_manifests/knowledge_base.yaml` | shared | .yaml |
| `src/backend/config/domain_manifests/solar.yaml` | shared | .yaml |
| `src/backend/config/domain_manifests/wind.yaml` | shared | .yaml |
| `src/backend/deps.py` | shared | Process-wide services (lazy). ``main`` and the routers share these function objects, so a test's |
| `src/backend/evaluation/__init__.py` | shared | Shared evaluation metrics and experiment helpers. |
| `src/backend/evaluation/provenance.py` | shared | Run provenance: package versions and base commit, saved next to every result. |
| `src/backend/main.py` | shared | GreenRAG-Sec FastAPI application (local mode: FastAPI + ChromaDB + MiniLM/cross-encoder + Ollama). |
| `src/backend/orchestrator.py` | shared | I1 - Orchestrator behind POST /assistant: forecast + diagnosis + security + trust, with ablation configs A-F. |
| `src/backend/runtime_env.py` | shared | Process-wide runtime settings shared by the API, experiments and tests. |
| `src/backend/serving/__init__.py` | shared | Python module |
| `src/backend/serving/generator.py` | shared | LLM answer generation for GreenRAG (local Ollama or Azure OpenAI). |
| `src/backend/serving/router.py` | shared | Deterministic query intent routing and numeric answer helpers for /ask. |
| `src/backend/serving/tiered_router.py` | shared | empty file |
| `src/backend/settings.py` | shared | Single settings class: every runtime option comes from an environment variable (or the project .env). |
| `src/backend/telemetry/__init__.py` | shared | empty file |
| `src/backend/telemetry/monitor.py` | shared | empty file |
| `tests/conftest.py` | shared | Shared fixtures: TestClient with in-memory fakes for the vector store and LLM. |
| `tests/test_api_regression.py` | shared | Regression tests for the pre-existing GreenRAG endpoints (fakes injected). |
| `tests/test_catalog_lab_api.py` | shared | Catalog endpoints (events, forecast assets/series, suggestions, chunks) and the security lab. |
| `tests/test_generator_retry.py` | shared | LocalGenerator retries once when Ollama's runner crashes during model load. |
| `tests/test_local_integration.py` | shared | End-to-end API tests on the real local backend (MiniLM + ChromaDB, temp dir). |
| `tests/test_ollama_integration.py` | shared | Real LLM path through /ask (needs `ollama serve` with the configured model). |
| `tests/test_orchestrator.py` | shared | I1: orchestrator (POST /assistant) - intent routing, ablation configs A-F, trust only in F. Fakes only. |
| `tests/test_platform_api.py` | shared | Platform endpoints: /ready, /warmup, /system/info, error format, request ids, CORS, settings, OpenAPI docs, |
| `tests/test_run_all.py` | shared | I3: run_all reuses finished results and never re-runs a complete step (subprocess is mocked). |
| `docs/fault_diagnosis.md` | tarunica | Track B - Fault diagnosis (B4) |
| `docs/knowledge_base.md` | tarunica | Track B - Knowledge base (B1) |
| `docs/rag_evaluation.md` | tarunica | Track B - RAG evaluation (B6) |
| `docs/screenshots/about.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/about_dark.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/about_mobile.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/assistant.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/assistant_dark.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/assistant_mobile.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/console_check.json` | tarunica | result/record (JSON) (screenshots) |
| `docs/screenshots/console_check_dark.json` | tarunica | result/record (JSON) (screenshots) |
| `docs/screenshots/console_check_mobile.json` | tarunica | result/record (JSON) (screenshots) |
| `docs/screenshots/diagnosis.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/diagnosis_dark.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/diagnosis_mobile.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/forecast.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/forecast_dark.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/forecast_mobile.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/lab.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/lab_dark.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/lab_heldout.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/lab_heldout_dark.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/lab_heldout_mobile.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/lab_injection.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/lab_injection_dark.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/lab_injection_mobile.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/lab_mobile.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/lab_poisoning.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/lab_poisoning_dark.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/lab_poisoning_mobile.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/openapi_docs.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/results.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/results_dark.png` | tarunica | figure/screenshot (screenshots) |
| `docs/screenshots/results_mobile.png` | tarunica | figure/screenshot (screenshots) |
| `docs/team/tarunica_rag_diagnosis.md` | tarunica | Track B - Knowledge base, hybrid RAG and fault diagnosis (Tarunica S, 23BIT0185) |
| `experiments/b6_followup.py` | tarunica | B6 follow-up: (1) hybrid RAG with sibling-chunk expansion on the fault events; (2) corrected judge (v2). |
| `experiments/b6_followup_report.py` | tarunica | B6 follow-up: re-score saved judgements with deterministic verification of template statements (no LLM calls). |
| `experiments/chunking_comparison.py` | tarunica | B2: compare fixed-size, semantic and metadata-aware chunking on the knowledge base. |
| `experiments/expansion_check.py` | tarunica | B6 fix check: does sibling-chunk expansion regress general retrieval on the 64 labelled KB queries? |
| `experiments/fault_diagnosis.py` | tarunica | B4: fault-diagnosis models on Wind Farm A SCADA (protocol in docs/fault_diagnosis.md). |
| `experiments/fault_rag.py` | tarunica | B5: diagnose all Wind Farm A events with the fault-to-RAG pipeline (production retrieval + local LLM). |
| `experiments/fault_threshold.py` | tarunica | B4 follow-up: Isolation Forest operating threshold by nested leave-one-asset-out. |
| `experiments/rag_evaluation.py` | tarunica | B6: evaluate LLM-only vs basic RAG vs hybrid RAG vs GreenRAG (resumable). |
| `experiments/retrieval_comparison.py` | tarunica | B3: compare keyword (BM25), vector, hybrid (RRF) and hybrid + cross-encoder reranking. |
| `experiments/wind_diagnose.py` | tarunica | Wind T1: where does persistence beat the learned models? (diagnostic only, no tuning) |
| `results/faultdiag/.gitkeep` | tarunica | empty folder placeholder |
| `results/faultdiag/b4_attribution.csv` | tarunica | result table (faultdiag) |
| `results/faultdiag/b4_event_metrics.csv` | tarunica | result table (faultdiag) |
| `results/faultdiag/b4_event_predictions.csv` | tarunica | result table (faultdiag) |
| `results/faultdiag/b4_if_threshold.csv` | tarunica | result table (faultdiag) |
| `results/faultdiag/b4_if_threshold.md` | tarunica | B4 follow-up: Isolation Forest threshold chosen by nested leave-one-asset-out |
| `results/faultdiag/b4_row_metrics.csv` | tarunica | result table (faultdiag) |
| `results/faultdiag/b4_run.json` | tarunica | result/record (JSON) (faultdiag) |
| `results/faultdiag/b4_summary.md` | tarunica | B4 fault diagnosis - Wind Farm A |
| `results/faultdiag/b5_reports.jsonl` | tarunica | raw outputs (JSON lines) (faultdiag) |
| `results/faultdiag/b5_results.csv` | tarunica | result table (faultdiag) |
| `results/faultdiag/b5_run.json` | tarunica | result/record (JSON) (faultdiag) |
| `results/faultdiag/b5_run1_reports.jsonl` | tarunica | raw outputs (JSON lines) (faultdiag) |
| `results/faultdiag/b5_run1_results.csv` | tarunica | result table (faultdiag) |
| `results/faultdiag/b5_run1_run.json` | tarunica | result/record (JSON) (faultdiag) |
| `results/faultdiag/b5_run1_summary.md` | tarunica | B5 fault diagnosis -> RAG, Wind Farm A |
| `results/faultdiag/b5_run2_reports.jsonl` | tarunica | raw outputs (JSON lines) (faultdiag) |
| `results/faultdiag/b5_run2_results.csv` | tarunica | result table (faultdiag) |
| `results/faultdiag/b5_run2_run.json` | tarunica | result/record (JSON) (faultdiag) |
| `results/faultdiag/b5_run2_summary.md` | tarunica | B5 fault diagnosis -> RAG, Wind Farm A |
| `results/faultdiag/b5_summary.md` | tarunica | B5 fault diagnosis -> RAG, Wind Farm A |
| `results/kb/b2_chunk_stats.csv` | tarunica | result table (kb) |
| `results/kb/b2_chunking_metrics.csv` | tarunica | result table (kb) |
| `results/kb/b2_chunking_summary.md` | tarunica | B2 chunking comparison |
| `results/kb/b2_paired_bootstrap.csv` | tarunica | result table (kb) |
| `results/kb/b2_run.json` | tarunica | result/record (JSON) (kb) |
| `results/kb/b3_classifier.csv` | tarunica | result table (kb) |
| `results/kb/b3_paired_bootstrap.csv` | tarunica | result table (kb) |
| `results/kb/b3_retrieval_metrics.csv` | tarunica | result table (kb) |
| `results/kb/b3_retrieval_summary.md` | tarunica | B3 retrieval comparison |
| `results/kb/b3_run.json` | tarunica | result/record (JSON) (kb) |
| `results/kb/b6_expansion_check.csv` | tarunica | result table (kb) |
| `results/kb/b6_expansion_check.md` | tarunica | Sibling-chunk expansion: retrieval regression check (64 labelled KB queries) |
| `results/kb/fault_taxonomy.csv` | tarunica | result table (kb) |
| `results/kb/kb_build.json` | tarunica | result/record (JSON) (kb) |
| `results/kb/kb_summary.md` | tarunica | B1 knowledge base build |
| `results/rag/.gitkeep` | tarunica | empty folder placeholder |
| `results/rag/answers.jsonl` | tarunica | raw outputs (JSON lines) (rag) |
| `results/rag/b6_per_answer.csv` | tarunica | result table (rag) |
| `results/rag/b6_run.json` | tarunica | result/record (JSON) (rag) |
| `results/rag/b6_summary.md` | tarunica | B6 RAG evaluation |
| `results/rag/judge_calibration.json` | tarunica | result/record (JSON) (rag) |
| `results/rag/judgements.jsonl` | tarunica | raw outputs (JSON lines) (rag) |
| `results/rag/run2/answers.jsonl` | tarunica | raw outputs (JSON lines) (rag) |
| `results/rag/run2/b6_followup_deterministic.md` | tarunica | B6 follow-up: template statements verified by code (judge v1 for everything else) |
| `results/rag/run2/b6_followup_summary.md` | tarunica | B6 follow-up: fair expansion for hybrid RAG, and a corrected judge (v2) |
| `results/rag/run2/b6_followup_template_claims.csv` | tarunica | result table (rag) |
| `results/rag/run2/b6_run2_per_answer.csv` | tarunica | result table (rag) |
| `results/rag/run2/b6_run2_summary.md` | tarunica | B6 run 2 - fault events after the two fixes |
| `results/rag/run2/hybrid_expanded_answers.jsonl` | tarunica | raw outputs (JSON lines) (rag) |
| `results/rag/run2/judge_v2_calibration.json` | tarunica | result/record (JSON) (rag) |
| `results/rag/run2/judgements.jsonl` | tarunica | raw outputs (JSON lines) (rag) |
| `results/rag/run2/judgements_v1_hybrid_expanded.jsonl` | tarunica | raw outputs (JSON lines) (rag) |
| `results/rag/run2/judgements_v2_greenrag_run2.jsonl` | tarunica | raw outputs (JSON lines) (rag) |
| `results/rag/run2/judgements_v2_hybrid_expanded.jsonl` | tarunica | raw outputs (JSON lines) (rag) |
| `results/rag/run2/judgements_v2_hybrid_run1.jsonl` | tarunica | raw outputs (JSON lines) (rag) |
| `results/retrieval/.gitkeep` | tarunica | empty folder placeholder |
| `scripts/capture_screenshots.py` | tarunica | Capture console screenshots with Playwright and check the browser console for errors. |
| `src/backend/evaluation/rag_eval.py` | tarunica | B6 - RAG answer evaluation: claim-level faithfulness, hallucination, relevance, numeric exact match. |
| `src/backend/evaluation/rag_systems.py` | tarunica | B6 - The four systems compared: LLM-only, basic RAG, hybrid RAG, GreenRAG. |
| `src/backend/faultdiag/__init__.py` | tarunica | Track B: SCADA fault diagnosis models (Tarunica). |
| `src/backend/faultdiag/data.py` | tarunica | B4 - Loading and normalising the per-event SCADA datasets (CARE-style, Wind Farm A). |
| `src/backend/faultdiag/diagnosis.py` | tarunica | B5 - Fault diagnosis connected to RAG: anomaly signal -> retrieval -> LLM -> validated report. |
| `src/backend/faultdiag/models.py` | tarunica | B4 - Anomaly detection, fault probability and component attribution (Wind Farm A). |
| `src/backend/faultdiag/service.py` | tarunica | B7 - Service behind POST /diagnose (fault signal -> evidence-grounded DiagnosisReport). |
| `src/backend/ingestion/__init__.py` | tarunica | GreenRAG ingestion package: loaders, chunkers, and local pipeline. |
| `src/backend/ingestion/chunkers.py` | tarunica | Tabular chunking and natural-language text builders for GreenRAG. |
| `src/backend/ingestion/loaders.py` | tarunica | Dataset loaders for GreenRAG ingestion. |
| `src/backend/ingestion/pipeline.py` | tarunica | Local ingestion pipeline: load → chunk → write JSONL. |
| `src/backend/ingestion/scada_ingest.py` | tarunica | SCADA ingest helpers (placeholder — not used by the local JSONL pipeline yet). |
| `src/backend/ingestion/weather_ingest.py` | tarunica | Weather ingest helpers (placeholder — not used by the local JSONL pipeline yet). |
| `src/backend/kb/__init__.py` | tarunica | Track B: knowledge-base construction with provenance metadata (Tarunica). |
| `src/backend/kb/build.py` | tarunica | B1 - Build, validate and store the GreenRAG-Sec knowledge base. |
| `src/backend/kb/chunking.py` | tarunica | B2 - Chunking strategies for the knowledge base. |
| `src/backend/kb/content.py` | tarunica | B1 - Hand-written source material for the SYNTHETIC knowledge-base documents. |
| `src/backend/kb/queries.py` | tarunica | B2 - Labelled query set with relevance known by construction. |
| `src/backend/kb/retrieval_eval.py` | tarunica | B2 - Retrieval evaluation against document-level relevance. |
| `src/backend/kb/schema.py` | tarunica | B1 - Knowledge-base document schema, provenance and validation. |
| `src/backend/kb/taxonomy.py` | tarunica | B1 - Fault taxonomy derived from the real event logs (event_info.csv, farms A-C). |
| `src/backend/retrieval/__init__.py` | tarunica | GreenRAG retrieval package: pluggable local / Azure vector stores. |
| `src/backend/retrieval/base.py` | tarunica | Abstract vector-store interface shared by local and Azure backends. |
| `src/backend/retrieval/corrective_retrieval.py` | tarunica | empty file |
| `src/backend/retrieval/factory.py` | tarunica | Factory that selects local vs Azure vector store from BACKEND_MODE. |
| `src/backend/retrieval/hybrid_retriever.py` | tarunica | B3 - Keyword (BM25), hybrid (reciprocal rank fusion), reranked and routed retrieval. |
| `src/backend/retrieval/local_store.py` | tarunica | Local vector store: sentence-transformers + ChromaDB (no cloud account). |
| `src/backend/retrieval/metrics.py` | tarunica | Shared helpers for numeric metadata / metric ranking. |
| `src/backend/retrieval/query_classifier.py` | tarunica | B3 - Query intent classifier: conceptual vs quantitative. |
| `src/console/css/app.css` | tarunica | console styles |
| `src/console/index.html` | tarunica | console page |
| `src/console/js/api.js` | tarunica | console script |
| `src/console/js/app.js` | tarunica | console script |
| `src/console/js/charts.js` | tarunica | console script |
| `src/console/js/pages/about.js` | tarunica | console script |
| `src/console/js/pages/assistant.js` | tarunica | console script |
| `src/console/js/pages/diagnosis.js` | tarunica | console script |
| `src/console/js/pages/forecast.js` | tarunica | console script |
| `src/console/js/pages/lab.js` | tarunica | console script |
| `src/console/js/pages/results.js` | tarunica | console script |
| `src/console/js/ui.js` | tarunica | console script |
| `src/frontend/app.js` | tarunica | console script |
| `src/frontend/index.html` | tarunica | console page |
| `src/frontend/styles.css` | tarunica | console styles |
| `tests/test_diagnose_api.py` | tarunica | B7: POST /diagnose (TestClient) with an injected service; one real-model test without the LLM. |
| `tests/test_fault_rag.py` | tarunica | B5: fault signal -> retrieval -> LLM -> validated DiagnosisReport (fake retriever/LLM; one real-Ollama test). |
| `tests/test_faultdiag.py` | tarunica | B4: SCADA loading safeguards, leave-one-asset-out protocol, metrics and attribution. |
| `tests/test_frontend.py` | tarunica | Console UI contract: static assets, element ids, design tokens, API routes. |
| `tests/test_hybrid_retrieval.py` | tarunica | B3: query classifier, BM25, rank fusion, reranking, routing (no model downloads). |
| `tests/test_kb.py` | tarunica | B1 knowledge base: taxonomy rules, provenance, synthetic labelling, validation, ingestion format. |
| `tests/test_kb_chunking.py` | tarunica | B2 setup: chunkers, labelled query set and retrieval metrics (mocked embeddings, no model download). |
| `tests/test_rag_eval.py` | tarunica | B6: claim-level evaluation, numeric checks, judge parsing/calibration, systems (fake LLMs, no Ollama). |

## Not copied

| path | reason |
|---|---|
| `chroma_db/` | generated vector index (git-ignored, rebuilt by POST /ingest) |
| `data/` | generated processed data (git-ignored, rebuilt by the pipelines) |
| `dataset/` | dataset folder (rule) |
| `models/` | model weights (rule; git-ignored) |
| `.env.azure` | .env file (rule; may hold settings) |
| `results/security/c6_telemetry_slots.csv` | over 5 MB (5.3 MB) |

## Unclassified (owner to be decided by the user; not copied)

- `architecture/.gitkeep`
- `docs/.gitkeep`
- `references/.gitkeep`
- `src/ai_model/.gitkeep`
