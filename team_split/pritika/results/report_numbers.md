# Report numbers (compiled from existing files only)

Compiled 2026-10-15 from files already in this repo. Nothing was re-run. Every value names its source file.
Anything with no record in the repo is marked **NOT RECORDED**.

Re-checked 2026-10-16 (no re-run): forecasting rows match `a5_summary.md` and `wind/wind_summary.md`; the repo
still has no Kaggle URL or licence text, no Dockerfile/CI/infrastructure files and no `.env`.

## 1. Forecasting metrics (1 hour ahead)

### Solar (fixed split, test 2020-06-13 .. 06-17, strict night rule)

Source: `results/forecasting/a5_summary.md` section 1 (raw values in `a5_fixed_split.csv`).
- Values are mean +/- sample std over seeds 0-2. The persistence baselines are deterministic.
- MAE and RMSE are in kW.
- MAPE is computed only on rows with y >= 1650 kW (5% of capacity): 224 rows for Plant 1, 223 for Plant 2.
- Scored test rows: 478 for Plant 1 and 480 for Plant 2 (`a3_summary.md`).

Plant 1:

| model | MAE (kW) | RMSE (kW) | MAPE (%) | R2 |
|---|---|---|---|---|
| persistence | 2458.0 | 4080.3 | 59.0 | 0.7530 |
| daily_persistence | 1961.5 | 3814.5 | 38.2 | 0.7841 |
| random_forest | 1444.1 +/- 13.2 | 2803.0 +/- 22.9 | 33.4 +/- 0.5 | 0.8834 +/- 0.0019 |
| xgboost | 1486.8 +/- 19.5 | 2848.9 +/- 23.8 | 34.2 +/- 0.6 | 0.8796 +/- 0.0020 |
| xgboost_no_day_month (ablation) | 1460.1 +/- 18.2 | 2817.1 +/- 43.8 | 33.3 +/- 0.4 | 0.8822 +/- 0.0037 |
| lstm | 1537.5 +/- 31.9 | 3032.8 +/- 83.6 | 33.5 +/- 1.3 | 0.8634 +/- 0.0075 |
| tcn | 1497.2 +/- 43.0 | 2858.2 +/- 116.6 | 31.9 +/- 2.5 | 0.8786 +/- 0.0100 |
| xgboost_oracle_weather (upper bound, not deployable) | 293.7 +/- 22.3 | 583.1 +/- 39.1 | 5.5 +/- 0.8 | 0.9949 +/- 0.0007 |

Plant 2:

| model | MAE (kW) | RMSE (kW) | MAPE (%) | R2 |
|---|---|---|---|---|
| persistence | 1964.5 | 3261.9 | 57.2 | 0.6671 |
| daily_persistence | 1623.0 | 3108.5 | 42.3 | 0.6976 |
| random_forest | 1230.2 +/- 14.4 | 2393.8 +/- 26.4 | 35.4 +/- 0.6 | 0.8207 +/- 0.0040 |
| xgboost | 1241.7 +/- 21.4 | 2402.9 +/- 55.2 | 35.5 +/- 0.9 | 0.8193 +/- 0.0083 |
| xgboost_no_day_month (ablation) | 1251.9 +/- 15.5 | 2417.3 +/- 21.0 | 35.4 +/- 0.5 | 0.8172 +/- 0.0032 |
| lstm | 1216.7 +/- 20.2 | 2317.8 +/- 14.3 | 32.5 +/- 0.5 | 0.8319 +/- 0.0021 |
| tcn | 1227.7 +/- 37.1 | 2319.3 +/- 51.1 | 32.3 +/- 0.9 | 0.8316 +/- 0.0074 |
| xgboost_oracle_weather (upper bound, not deployable) | 775.5 +/- 15.0 | 1869.6 +/- 3.8 | 18.6 +/- 0.4 | 0.8906 +/- 0.0004 |

The solar rolling-origin evaluation (12 test days, `a5_summary.md` section 2) records MAE and RMSE only. Its MAPE and R2 are **NOT RECORDED**.

### Wind T1 (rolling origin by month, test 2018-09 .. 2018-12, 10-min resolution)

Source: `results/forecasting/wind/wind_summary.md` (raw values in `point_metrics.csv`).
- 16,330 scored rows over 116 days.
- Values are mean +/- std over seeds 0-2.
- MAPE is computed only on rows with y >= 180 kW (5% of the rated 3600 kW).

| model | MAE (kW) | RMSE (kW) | MAPE (%) | R2 |
|---|---|---|---|---|
| persistence | 298.1 | 490.9 | 33.3 | 0.8619 |
| daily_persistence | 1157.1 | 1573.1 | 111.3 | -0.4184 |
| random_forest | 343.2 +/- 1.1 | 486.4 +/- 0.8 | 33.8 +/- 0.1 | 0.8644 +/- 0.0004 |
| xgboost | 329.6 +/- 1.7 | 480.3 +/- 1.6 | 33.8 +/- 0.1 | 0.8678 +/- 0.0009 |
| lstm | 365.1 +/- 6.0 | 523.8 +/- 14.5 | 35.7 +/- 1.2 | 0.8427 +/- 0.0088 |
| tcn | 339.8 +/- 8.2 | 498.5 +/- 5.6 | 35.3 +/- 0.7 | 0.8576 +/- 0.0032 |
| xgboost_oracle_weather (upper bound, not deployable) | 114.8 +/- 0.9 | 216.9 +/- 0.5 | 10.5 +/- 0.0 | 0.9730 +/- 0.0001 |

On wind, persistence has a lower MAE than every learned model.

## 2. Datasets

Sizes are the on-disk bytes of `dataset/` (1 MB = 2^20 bytes), measured with `du -sb`.
- Raw row counts come from the recorded preprocessing reports where those exist.
- Fault row counts were counted from the CSV files (lines minus one header line per file); no fault row count was recorded before this.

| dataset | files | size (MB) | raw rows | raw columns | model features |
|---|---|---|---|---|---|
| Solar Plant 1 | generation + weather CSV | 4.89 (4.61 + 0.27) | generation 68,778; weather 3,182 | generation 7, weather 6 | 26 (A2, excluding the oracle feature) |
| Solar Plant 2 | generation + weather CSV | 5.82 (5.54 + 0.29) | generation 67,698; weather 3,259 | generation 7, weather 6 | 26 |
| Solar total | 4 CSV | 10.71 | | | |
| Wind T1 | `T1.csv` | 3.79 | 50,530 | 5 | 26 (excluding the 2 oracle features) |
| Fault, Wind Farm A | 22 event CSV + event_info + feature_description | 765.7 | 1,196,747 | 86 (5 id/status + 81 sensor) | 85 (B4; `status_type_id` excluded) |
| Fault, Wind Farm B | 15 event CSV + metadata | 1302.6 | 859,065 | 257 | not modelled |
| Fault, Wind Farm C | 58 event CSV + metadata | 16993.3 | 3,187,136 | 957 | not modelled |

Solar row sources:
- Raw rows: `preprocessing_report_plant{1,2}.json`.
- Both plants have a 15-min grid of 3,264 slots.
- Feature rows: 3,142 for Plant 1 (train/val/calib/test 1901/379/384/478) and 3,254 for Plant 2 (2006/384/384/480), from `features_plant{1,2}.json`.

Wind row sources:
- `wind_summary.md`: 52,560 grid slots and 2,030 missing.
- Feature count: `WIND_FEATURES` in `src/backend/forecasting/wind.py`.

Fault metadata:
- Events in `event_info.csv`: A 22, B 15, C 58 (95 in total, 45 anomaly / 50 normal).
- `feature_description.csv` rows: A 54, B 63, C 238.
- B4 and B5 use Farm A only.

Licences and sources:

| dataset | licence text | source URL |
|---|---|---|
| Solar (Plant 1/2 generation + weather) | **NOT RECORDED** | **NOT RECORDED** (no Kaggle URL anywhere in the repo) |
| Wind T1 | **NOT RECORDED** | **NOT RECORDED** |
| Fault (Wind Farms A-C) | **NOT RECORDED**: the bundled `dataset/fault_diagnosis/README.md` states no licence | Kaggle URL **NOT RECORDED**. The README gives only the Farm A upstream source, the EDP open data platform: https://www.edp.com/en/innovation/open-data/data |

## 3. Environment

Source: the `environment` block of `results/forecasting/a3_run.json`.

- Python: 3.11.15, on Windows-10-10.0.26200-SP0.
- LSTM/TCN framework: PyTorch (`torch` 2.13.0; `torch.nn` modules in `src/backend/forecasting/models.py`). It runs on CPU because the installed torch build has no CUDA (PROGRESS.md A3).
- Other recorded packages:
  - numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, xgboost 3.2.0, statsmodels 0.15.0, matplotlib 3.11.2
  - sentence-transformers 5.7.0, chromadb 1.5.9, fastapi 0.141.1, pydantic 2.13.4

## 4. Azure deployment

There is no Azure deployment.
- Project decision (2026-09-30, CLAUDE.md rule 12, `docs/threat_model.md`): local mode only, with no Azure setup, implementation or testing.
- The `BACKEND_MODE=azure` code path exists but has never been deployed or run.
- `results/cloud/` is empty.

| item | status |
|---|---|
| Services actually deployed | None. The code path targets Azure OpenAI (chat completions, and embeddings via `api_version` 2024-02-01) and Azure AI Search (one index per domain) (`serving/generator.py`, `retrieval/azure_store.py`). Azure Storage and Azure Cosmos DB appear only as empty variables in `.env.example`; no code uses them. |
| Region | **NOT RECORDED** |
| Image build and push | **NOT RECORDED** (the repo has no Dockerfile, container registry config or CI pipeline) |
| Private endpoints | **NOT RECORDED** (no infrastructure code; the code expects plain endpoint URLs) |
| Scaling rules | **NOT RECORDED** |
| Authentication | As written in code, not deployed: API keys from environment variables (`AzureKeyCredential` for AI Search, an API key for Azure OpenAI). No managed identity or Entra ID code. The local API itself has no authentication (`src/backend/security/auth.py` is empty; threat model C1). |
| Azure smoke-test result | **NOT RECORDED** |
| Azure latency | **NOT RECORDED** (PLAN I4 is marked PENDING without Azure keys and is never simulated) |
| Azure spend | **NOT RECORDED** |

## 5. Knowledge-base documents

Source: `results/kb/kb_summary.md` and `docs/knowledge_base.md`.
- The KB (domain `knowledge_base`) has 189 documents in 210 build chunks.
- Production retrieval uses the B2 semantic chunks with hybrid BM25 + MiniLM retrieval and a cross-encoder rerank.

| doc_type | count | real or synthetic | content |
|---|---|---|---|
| fault_description | 45 | real (from the dataset) | one per anomaly event in `event_info.csv`, farms A-C |
| scada_knowledge | 54 | real (from the dataset) | per-farm event overviews, component event histories, sensor catalogues from `feature_description.csv` |
| equipment_manual | 19 | synthetic | 13 wind components + 6 solar-inverter fault types from the plan |
| troubleshooting_procedure | 19 | synthetic | ordered checks and corrective actions |
| maintenance_record | 45 | synthetic | one invented work order per real anomaly event |
| standard_summary | 7 | synthetic | original scope overviews of IEC 61400-1, 61400-12-1, 61400-25, IEC 61724-1, IEC 62446-1, ISO 17359, ISO 55000 (no standard text copied) |

Totals:
- 99 real and 90 synthetic documents.
- Synthetic documents carry `synthetic: true` and the `[SYNTHETIC]` marker in both title and text.

Two other text sources are not KB documents:
- The C2 attack documents (`results/security/c2_attack_docs.jsonl`) are synthetic and are injected only in attack runs.
- The older per-domain chunks (`data/processed/{solar,wind,fault_diagnosis}/chunks.jsonl`) are built from the raw datasets.

## 6. Phase-I features

| feature | implemented? | evidence |
|---|---|---|
| Event-driven ingestion | No | `src/backend/ingestion/scada_ingest.py` and `weather_ingest.py` contain only a docstring: "placeholder - not used by the local JSONL pipeline yet". Ingestion is the batch CLI / `POST /ingest/{id}`. |
| Tiered serving | No | `src/backend/serving/tiered_router.py` is an empty file (0 bytes). The only router is the deterministic numeric router (`serving/router.py`). |
| Power BI view | No | No Power BI file, export or code anywhere in the repo. |

Also empty, for completeness: `src/backend/telemetry/monitor.py` and `src/backend/retrieval/corrective_retrieval.py` (both 0 bytes).
