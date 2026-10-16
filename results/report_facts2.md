# Report facts 2 (compiled 2026-10-16 from the repository; read only, nothing estimated)

Anything not found in the repository is marked **NOT FOUND**.

## 1. Columns per dataset (raw files) and features as used

| dataset | file | raw columns | main columns | model features as used |
|---|---|---|---|---|
| Solar Plant 1, generation | `dataset/solar/Plant_1_Generation_Data.csv` | 7 | DATE_TIME, PLANT_ID, SOURCE_KEY (inverter), DC_POWER, AC_POWER, DAILY_YIELD, TOTAL_YIELD | 26 combined with weather (`results/report_numbers.md`) |
| Solar Plant 1, weather | `dataset/solar/Plant_1_Weather_Sensor_Data.csv` | 6 | DATE_TIME, PLANT_ID, SOURCE_KEY, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION | (in the 26) |
| Solar Plant 2, generation | `dataset/solar/Plant_2_Generation_Data.csv` | 7 | same as Plant 1 | 26 combined with weather |
| Solar Plant 2, weather | `dataset/solar/Plant_2_Weather_Sensor_Data.csv` | 6 | same as Plant 1 | (in the 26) |
| Wind T1 | `dataset/wind/T1.csv` | 5 | Date/Time, LV ActivePower (kW), Wind Speed (m/s), Theoretical_Power_Curve (KWh), Wind Direction (deg) | 26, excluding 2 oracle features (`results/report_numbers.md`) |
| Wind Farm A SCADA | `dataset/fault_diagnosis/Wind Farm A/datasets/0.csv` (one per event, semicolon-separated) | 86 | time_stamp, asset_id, id, train_test, status_type_id, then 81 sensor columns (e.g. sensor_0_avg, wind_speed_3_avg/max/min/std, sensor_5_avg) | 85 (B4; `status_type_id` excluded because it leaks the label) |

## 2. Forecasting metrics (1 hour ahead; MAE and RMSE in kW; mean over the seeds shown)

Solar: `results/forecasting/a5_fixed_split.csv` (fixed test split). Wind: `results/forecasting/wind/point_metrics.csv`
(rolling origin, Sep-Dec 2018).

```
asset         model                        MAE     RMSE  MAPE%       R2 seeds
solar_plant1  persistence               2458.0   4080.3   59.0   0.7530     1
solar_plant1  daily_persistence         1961.5   3814.5   38.2   0.7841     1
solar_plant1  random_forest             1444.1   2803.0   33.4   0.8834     3
solar_plant1  xgboost                   1486.8   2848.9   34.2   0.8796     3
solar_plant1  xgboost_no_day_month      1460.1   2817.1   33.3   0.8822     3
solar_plant1  lstm                      1537.5   3032.8   33.5   0.8634     3
solar_plant1  tcn                       1497.2   2858.2   31.9   0.8786     3
solar_plant1  xgboost_oracle_weather     293.7    583.1    5.5   0.9949     3
solar_plant2  persistence               1964.5   3261.9   57.2   0.6671     1
solar_plant2  daily_persistence         1623.0   3108.5   42.3   0.6976     1
solar_plant2  random_forest             1230.2   2393.8   35.4   0.8207     3
solar_plant2  xgboost                   1241.7   2402.9   35.5   0.8193     3
solar_plant2  xgboost_no_day_month      1251.9   2417.3   35.4   0.8172     3
solar_plant2  lstm                      1216.7   2317.8   32.5   0.8319     3
solar_plant2  tcn                       1227.7   2319.3   32.3   0.8316     3
solar_plant2  xgboost_oracle_weather     775.5   1869.6   18.6   0.8906     3
wind_t1       daily_persistence         1157.1   1573.1  111.3  -0.4184     1
wind_t1       lstm                       365.1    523.8   35.7   0.8427     3
wind_t1       persistence                298.1    490.9   33.3   0.8619     1
wind_t1       random_forest              343.2    486.4   33.8   0.8644     3
wind_t1       tcn                        339.8    498.5   35.3   0.8576     3
wind_t1       xgboost_oracle_weather     114.8    216.9   10.5   0.9730     3
wind_t1       xgboost                    329.6    480.3   33.8   0.8678     3
```

- The persistence baselines are deterministic, so they have 1 seed.
- `xgboost_oracle_weather` uses the measured weather at the target time. It is an upper bound and cannot be
  deployed.

## 3. Phase dates (`PROGRESS.md` section headings)

| phase | start | end |
|---|---|---|
| Phase 0 setup | 2026-09-28 | 2026-09-28 |
| A1-A4 forecasting (preprocessing, features, models, uncertainty) | 2026-09-28 | 2026-09-28 |
| A5 evaluation, A6 API, wind T1 replication | 2026-09-29 | 2026-09-29 |
| B1-B4 (knowledge base, chunking, retrieval, fault diagnosis) | 2026-09-29 | 2026-09-29 |
| B5-B7 (diagnosis to RAG, RAG evaluation, /diagnose) | 2026-09-30 | 2026-09-30 |
| C1 threat model | 2026-09-30 | 2026-09-30 |
| C2 attack simulator | 2026-09-30 | 2026-10-15 (qwen run completed) |
| C3 detectors, C4 secure retrieval | 2026-10-01 | 2026-10-01 |
| C5 decision engine | 2026-10-07 | 2026-10-07 |
| C6 security evaluation | 2026-10-15 (capture run, `results/security/c6_run.json`) | 2026-10-16 |
| C7 API, I1-I5 integration, console and team docs | 2026-10-16 | 2026-10-16 |
| Azure plan A (stages 1-3) | 2026-10-16 | 2026-10-16 |

These five activities are not recorded as separate entries in `PROGRESS.md`:

| activity | dates |
|---|---|
| literature survey | **NOT FOUND** |
| research gap and novelty | **NOT FOUND** |
| architecture design | **NOT FOUND** (the nearest record is Phase 0 setup, 2026-09-28) |
| implementation and integration | **NOT FOUND** as a named activity (the phases above run 2026-09-28 to 2026-10-16) |
| testing and report | **NOT FOUND** as a named activity (each phase ends with its tests; final suite 2026-10-16) |

## 4. Ownership in `docs/team/*.md`

The team files assign the three tracks:
- `docs/team/README.md` line 10: "| Minmini K (23BIT0198) | A - forecasting with uncertainty | ..."
- line 11: "| Tarunica S (23BIT0185) | B - knowledge base, hybrid RAG, fault diagnosis | ..."
- line 12: "| Pritika Kasinathan (23BIT0232) | C - security and trust | ..."

| role | owner |
|---|---|
| frontend (console) | **NOT FOUND**. No file names an owner; the API map lists shared items as "shared". |
| cloud services (Azure) | **NOT FOUND**. No owner is named. The team files predate the Azure deployment: `docs/team/README.md` lines 4-5 still say "no cloud service is used". |
| security | Pritika Kasinathan, `docs/team/README.md` line 12 (track C - security and trust) |
| documentation and presentation | **NOT FOUND** |

## 5. Phase-I items

| item | implemented? | evidence |
|---|---|---|
| event-driven streaming ingestion | **no** | `src/backend/ingestion/scada_ingest.py` (88 bytes) and `weather_ingest.py` (90 bytes) contain only a placeholder docstring. No Event Hubs, Kafka or streaming code was found. |
| tiered model serving | **no** | `src/backend/serving/tiered_router.py` is empty (0 bytes) |
| Power BI telemetry view | **no** | **NOT FOUND**: no Power BI file or code in the repository. Request telemetry goes to Application Insights (`src/backend/cloud/appinsights.py`), not Power BI. |
