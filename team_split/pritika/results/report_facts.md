# Report facts (compiled 2026-10-16 from this repository; nothing estimated)

Every value names its source. Anything with no record is marked **NOT RECORDED**.

## Part A - Azure evidence

Checked on this machine on 2026-10-16:
- **No Azure CLI.** The `az` command is not installed, so `az account show`, `az resource list` and `az group show`
  could not be run.
- **No configuration.** The repository has no `.env` file, and no `AZURE_*` variables are set in the environment.
- **Local-only record.** `results/cloud/i4_status.md` (PENDING) and the 2026-09-30 decision in CLAUDE.md rule 12
  record the project as local-only.

| item | status | source |
|---|---|---|
| Azure region | **NOT RECORDED** | no CLI, no configuration |
| Resource group | **NOT RECORDED** | no CLI, no configuration |
| Azure OpenAI | **NOT DEPLOYED**: exists only as code (`src/backend/serving/generator.py`, `BACKEND_MODE=azure` path) | code |
| Azure AI Search | **NOT DEPLOYED**: exists only as code (`src/backend/retrieval/azure_store.py`) | code |
| Blob Storage, Cosmos DB | **NOT DEPLOYED**: empty variables in `.env.example` only; no code uses them | `.env.example` |
| Key Vault, App Service, Application Insights | **NOT DEPLOYED**: no code, no configuration | repository search |
| `BACKEND_MODE=azure` run end to end | **No.** No endpoint (/ask, /forecast, /diagnose, /secure-ask, /decision) has been tested against Azure | `results/cloud/i4_status.md`, PROGRESS.md |
| I4 local-vs-Azure harness | **Not run; cannot run here.** No Azure CLI, credentials or endpoints, so there is no latency or equivalence measurement | `results/cloud/i4_status.md` |
| Azure cost (actual, period to date) | **NOT RECORDED** (Cost Management is not reachable without an account and CLI) | - |
| One-month cost estimate | **NOT RECORDED** (no pricing source was consulted; no estimate was made) | - |
| Azure-hosted app URL | **None**: no app is hosted on Azure | - |
| Portal screenshots 06-08 | **Not taken.** No signed-in portal session exists, and the console's status pill shows "local mode" (`/system/status`) | - |

## Part B - Facts for the report placeholders

### 1. Dataset rows and sizes (`results/report_numbers.md`, section 2; preprocessing reports)

| dataset | raw rows | size (MB, 2^20 bytes) |
|---|---|---|
| Solar Plant 1 | generation 68,778 + weather 3,182 = 71,960 | 4.89 |
| Solar Plant 2 | generation 67,698 + weather 3,259 = 70,957 | 5.82 |
| Wind T1 (`T1.csv`) | 50,530 | 3.79 |
| Wind Farm A (fault data, 22 event CSVs) | 1,196,747 | 765.7 |

- **Total of these four:** 1,390,194 rows and 780.2 MB.
- **Other fault farms, not modelled:** Wind Farm B (859,065 rows, 1302.6 MB) and Wind Farm C (3,187,136 rows,
  16993.3 MB).
- **Solar time grid:** both plants have 3,264 slots of 15 min (2020-05-15 to 2020-06-17)
  (`results/forecasting/preprocessing_report_plant{1,2}.json`).

### 2. Dataset URLs and licences

| dataset | URL | licence | source |
|---|---|---|---|
| Solar Plant 1/2 | **NOT RECORDED**: no Kaggle URL anywhere in the repository | **NOT RECORDED** | `results/report_numbers.md` |
| Wind T1 | **NOT RECORDED** | **NOT RECORDED** | `results/report_numbers.md` |
| Fault data (Wind Farms A-C) | Kaggle URL **NOT RECORDED**. The bundled README gives the Farm A upstream source: https://www.edp.com/en/innovation/open-data/data | **NOT RECORDED**: the README states no licence | `dataset/fault_diagnosis/README.md`, `README.txt` |

The licence text has to be copied from each dataset's download page; it is not in the repository.

### 3. Software versions (`uv pip freeze` in the project venv, 2026-10-16; `python --version`)

| package | version |
|---|---|
| Python | 3.11.15 |
| torch (PyTorch) | 2.13.0 (CPU build; no CUDA, per PROGRESS.md A3) |
| xgboost | 3.2.0 |
| scikit-learn | 1.9.0 |
| sentence-transformers | 5.7.0 |
| chromadb | 1.5.9 |
| fastapi | 0.141.1 |

The deep-learning framework is **PyTorch** (`torch.nn` LSTM and TCN in `src/backend/forecasting/models.py`).

### 4. Forecasting metrics (1 hour ahead; MAE/RMSE in kW; mean over seeds 0-2)

Sources: `results/forecasting/a5_fixed_split.csv` (solar, fixed test split) and
`results/forecasting/wind/point_metrics.csv` (wind, rolling origin Sep-Dec 2018).

| model | P1 MAE | P1 RMSE | P1 MAPE % | P1 R2 | P2 MAE | P2 RMSE | P2 MAPE % | P2 R2 | Wind MAE | Wind RMSE | Wind MAPE % | Wind R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| persistence | 2458.0 | 4080.3 | 59.0 | 0.7530 | 1964.5 | 3261.9 | 57.2 | 0.6671 | 298.1 | 490.9 | 33.3 | 0.8619 |
| daily persistence | 1961.5 | 3814.5 | 38.2 | 0.7841 | 1623.0 | 3108.5 | 42.3 | 0.6976 | 1157.1 | 1573.1 | 111.3 | -0.4184 |
| random forest | 1444.1 | 2803.0 | 33.4 | 0.8834 | 1230.2 | 2393.8 | 35.4 | 0.8207 | 343.2 | 486.4 | 33.8 | 0.8644 |
| XGBoost | 1486.8 | 2848.9 | 34.2 | 0.8796 | 1241.7 | 2402.9 | 35.5 | 0.8193 | 329.6 | 480.3 | 33.8 | 0.8678 |
| LSTM | 1537.5 | 3032.8 | 33.5 | 0.8634 | 1216.7 | 2317.8 | 32.5 | 0.8319 | 365.1 | 523.8 | 35.7 | 0.8427 |
| TCN | 1497.2 | 2858.2 | 31.9 | 0.8786 | 1227.7 | 2319.3 | 32.3 | 0.8316 | 339.8 | 498.5 | 35.3 | 0.8576 |

- **MAPE scope.** MAPE is computed only on rows with an observed value >= 5% of capacity.
- **Wind result.** On wind, persistence has a lower MAE than every learned model.

### 5. Knowledge-base sources

Sources: `results/kb/kb_summary.md`, `docs/knowledge_base.md`, `data/processed/knowledge_base/documents.jsonl`.
The knowledge base has 189 documents (99 real, 90 synthetic) in 210 chunks.

**Real documents (99)**
- **45 fault descriptions:** one per anomaly event in `dataset/fault_diagnosis/Wind Farm {A,B,C}/event_info.csv`.
  Each gives the event window, turbine, label and the logged description quoted verbatim.
- **54 SCADA knowledge documents**, built from the same dataset:
  - per-farm event overviews;
  - per-component event histories;
  - sensor catalogues from `feature_description.csv`.

**Synthetic documents (90)**, written for this project and marked `synthetic: true` and `[SYNTHETIC]`:
- 19 equipment manuals (13 wind components + 6 solar-inverter fault types from the plan);
- 19 troubleshooting procedures;
- 45 maintenance records (one invented work order per real anomaly event);
- 7 standard overviews (IEC 61400-1, 61400-12-1, 61400-25, IEC 61724-1, IEC 62446-1, ISO 17359, ISO 55000).
  These are original summaries; no standard text is copied.

### 6. Forecasting steps (`docs/forecasting.md`)

**A1 preprocessing.** Every step is causal: the output at t uses only rows at or before t.
1. Duplicates: drop exact duplicates; drop all rows of a (timestamp, inverter) key that has conflicting values.
2. Time sync: parse timestamps with explicit formats and align them to the 15-min grid.
3. Plant aggregation: plant AC = mean AC over the reporting inverters x 22, keeping coverage.
4. Physical bounds: AC within [0, 22 x 1500 kW], plus a night-consistency check.
5. Missing values: forward-fill weather for at most 2 slots; a missing target stays missing.
6. Statistical outliers: flag a trailing robust z-score (values kept).
7. Noise: keep causal 3-slot rolling-median copies of the weather inputs.
8. Normalisation: `TrainOnlyScaler`, fitted on the training frame only.

**A2 features.**
- Native 15-min grid, 1-hour horizon (4 steps).
- Lags 1/3/6/24 and the same slot yesterday.
- Rolling mean and std over 4 and 16 slots.
- Weather at the origin, an irradiation envelope from previous days, and the calendar of the target time.
- Split by target day: train 21 / val 4 / calib 4 / test 5 days.

**A3 models.**
- Persistence, daily persistence, random forest, XGBoost, LSTM and TCN.
- Every model is scored on the same rows, with train-only imputation.
- Predictions are clipped to [0, 33,000 kW], and night slots are set to 0.
- Models are selected on validation only; seeds 0-2.

**A4 uncertainty.**
- Multi-quantile XGBoost plus conformal calibration on the calibration days.
- Methods: qr_raw, gaussian, split_conformal, cqr_pooled and cqr_mondrian (CQR per day/night group; deployed).

**A5 evaluation.**
- Fixed split, plus a rolling origin over 12 test days per plant.
- Paired day-level bootstrap for model differences; coverage, width, Winkler and pinball for intervals.

**A6 API.** `POST /forecast` returns the prediction, lower and upper bounds, and the confidence (= nominal coverage).

**Wind T1.** The same pipeline on a 10-min grid with a 6-step horizon. It uses power-regime groups instead of
day/night, and a rolling origin by month (Sep-Dec 2018).

### 7. Latest test count

`python -m pytest`, run on **2026-10-16** with Ollama up: **410 passed**, 0 failed, 0 skipped
(PROGRESS.md, "Product polish" section).

### 8. Git (read-only; only the three requested commands, run 2026-10-16)

- `git shortlog -sne --all`: `9  daschirag <<email redacted>>`
- `git branch -a`: `develop`, `* feature/dev`, `main`, `remotes/origin/feature/dev`
- `git log --oneline | wc -l` (current branch): `9`

All work after commit 4e586aa is uncommitted (the no-git rule), so these counts do not include it.

### 9. Project dates (PROGRESS.md section headings)

| phase | date |
|---|---|
| Phase 0 setup (project start in PROGRESS.md) | 2026-09-28 |
| A1-A4 forecasting | 2026-09-28 |
| A5-A6, wind replication, B1-B4 | 2026-09-29 |
| B5-B7, C1, C2 (first runs) | 2026-09-30 |
| C3, C4 | 2026-10-01 |
| C5 decision engine | 2026-10-07 |
| C2 qwen run completed | 2026-10-15 |
| C6, C7, I1-I5, console and team docs | 2026-10-16 |

PROGRESS.md does not record a project start before Phase 0. The earliest commit date was not looked up (only the
three git commands above were run).
