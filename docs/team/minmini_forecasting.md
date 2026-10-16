# Track A - Forecasting with uncertainty (Minmini K, 23BIT0198)

## Scope

1-hour-ahead AC power forecasts with a calibrated prediction interval for three assets:
- solar Plant 1 and Plant 2 (15-min grid);
- wind turbine T1 (10-min grid).

The work covers six phases:
- A1 preprocessing;
- A2 features;
- A3 models (persistence, daily persistence, random forest, XGBoost, LSTM, TCN);
- A4 uncertainty (quantile regression + conformal calibration, CQR per group);
- A5 evaluation (fixed split, rolling origin, paired day-level bootstrap);
- A6 API.

Wind T1 repeats the same pipeline.

## Data (from `results/report_numbers.md` and the preprocessing reports)

| series | raw rows | grid | model features |
|---|---|---|---|
| Solar Plant 1 | generation 68,778; weather 3,182 | 3,264 slots of 15 min (2020-05-15 to 2020-06-17) | 26 |
| Solar Plant 2 | generation 67,698; weather 3,259 | 3,264 slots | 26 |
| Wind T1 | 50,530 | 10 min, 2018 | 26 |

Plant 1 feature rows by split (`features_plant1.json`): train 1901, val 379, calib 384, test 478.

How the splits are made:
- **Chronological.** Splits are made by whole days, never at random.
- **Train-only fitting.** Scalers and imputers are fitted on train only.
- **No look-ahead.** Rolling features use data up to the origin only. A prefix-invariance test fails on two
  injected leaks.
- **No humidity.** The solar data has no humidity column, and none was invented.

## Files and modules

| file | role |
|---|---|
| `src/backend/forecasting/preprocess.py` | A1: explicit timestamp formats, 15-min grid, plant aggregation, causal cleaning, train-only scaler |
| `src/backend/forecasting/features.py` | A2: lags 1/3/6/24, same slot yesterday, rolling mean/std, weather at origin, calendar of target |
| `src/backend/forecasting/models.py` | A3: persistence, daily persistence, RF, XGBoost, LSTM, TCN; shared post-processing (night rule, capacity clip) |
| `src/backend/forecasting/uncertainty.py` | A4: multi-quantile XGBoost, conformal quantile, `cqr_mondrian` (CQR calibrated per day/night or power regime) |
| `src/backend/forecasting/evaluation.py`, `experiments/rolling_origin.py`, `experiments/evaluate_forecasting.py` | A5 |
| `src/backend/forecasting/wind.py`, `experiments/wind_rolling.py`, `experiments/wind_evaluate.py` | wind replication |
| `src/backend/forecasting/service.py`, `schemas.py` | A6: deployed models, `/forecast`, day series |
| `results/forecasting/` | all outputs (`a3`..`a6`, `wind/`), grayscale figures |

## Endpoints

- `POST /forecast {asset | plant, origin, confidence in {0.8, 0.9, 0.95}}` returns:
  - `prediction`, `lower_bound`, `upper_bound`, `confidence`;
  - `target_time`, `observed_kw`, `in_sample`, `calibration_group`, `model_version`.
- `GET /forecast/assets`: the grid, the valid origin range and the days the deployed model was fit on.
- `GET /forecast/series?asset&start&end&confidence`: every origin in a window, with the interval and the observed
  value. It is the same computation as `/forecast`; bounds agree to under 0.01 kW (float32 batching; tested).
- `POST /decision {kind: "forecast"}`: the same forecast scored by the trust engine.

## Results (out-of-sample)

Solar, fixed test split (`results/forecasting/a5_fixed_split.csv`, mean over seeds 0-2), MAE in kW:

| model | Plant 1 MAE | Plant 1 R2 | Plant 2 MAE | Plant 2 R2 |
|---|---|---|---|---|
| persistence | 2458.0 | 0.7530 | 1964.5 | 0.6671 |
| daily persistence | 1961.5 | 0.7841 | 1623.0 | 0.6976 |
| random forest | 1444.1 | 0.8834 | 1230.2 | 0.8207 |
| XGBoost | 1486.8 | 0.8796 | 1241.7 | 0.8193 |
| LSTM | 1537.5 | 0.8634 | 1216.7 | 0.8319 |
| TCN | 1497.2 | 0.8786 | 1227.7 | 0.8316 |
| XGBoost + measured weather at target (upper bound, not deployable) | 293.7 | 0.9949 | 775.5 | 0.8906 |

Solar, rolling origin (`results/forecasting/a5_summary.md`, 12 test days per plant):
- **Learned models vs persistence.** Every learned model beats persistence on 12/12 days on Plant 1. Random forest
  minus persistence MAE is -1039.7 kW, 95% CI [-1203.8, -888.3].
- **Comparisons between learned models.** On Plant 1, random forest beats XGBoost (-63.5 kW, CI [-99.4, -32.7]).
  On Plant 2 the same comparison is not resolved (-17.4 kW, CI [-68.0, 27.7]).

Wind T1, rolling origin Sep-Dec 2018 (`results/forecasting/wind/point_metrics.csv`): **persistence wins on MAE.**

| model | MAE (kW) | RMSE (kW) | R2 |
|---|---|---|---|
| persistence | 298.1 | 490.9 | 0.8619 |
| XGBoost | 329.6 | 480.3 | 0.8678 |
| TCN | 339.8 | 498.5 | 0.8576 |
| random forest | 343.2 | 486.4 | 0.8644 |
| LSTM | 365.1 | 523.8 | 0.8427 |

90% interval coverage, `cqr_mondrian` = deployed method (`a5_coverage_ci.csv`, `wind/coverage_ci.csv`):

| asset | coverage | 95% CI | days | qr_raw (no calibration) |
|---|---|---|---|---|
| Solar Plant 1 (daylight) | 0.893 | [0.853, 0.927] | 12 | 0.846 |
| Solar Plant 2 (daylight) | 0.906 | [0.873, 0.938] | 12 | 0.889 |
| Wind T1 | 0.890 | [0.869, 0.909] | 116 | 0.866 |

The Plant 1 CI includes 0.90. In the wind high-power regime, coverage is 0.698 for `cqr_mondrian` and 0.559 for
`qr_raw` (`wind/wind_summary.md`).

## Limitations (honest)

- **Short solar record.** The solar data covers 34 days per plant, so every CI is wide; 12 rolling test days is a
  small sample.
- **Wind loses to persistence on MAE.** On the 10-min wind series, persistence beats every learned model on MAE.
  The learned models win slightly on RMSE (ramps) and lose on flat hours and at 0 / rated power.
- **High-regime under-coverage.** Every quantile method under-covers in the wind high-power regime.
- **Served forecasts are in-sample.** The deployed models are fit on the whole record (train / val / calib), so
  `in_sample` is true for every target inside the record. The console says so; judge accuracy by the rolling-origin
  results.
- **Night rule fixed on validation data.** The night rule forces dawn/dusk slots to 0 kW; its threshold was
  chosen on validation days (A4), not tuned on test.
- **Not in the served path.** No live telemetry check guards `/forecast` (C3 is day-level and offline). So
  `/decision` can at best return "review" for a served forecast: S = 0.
- **No humidity or weather forecasts.** The oracle-weather row shows how much a real weather forecast could add.

## Likely viva questions

1. **Why 15 min and a 1-hour horizon?**
   - The record is 34 days, so down-sampling would leave too few samples.
   - There is no NWP (weather forecast) data, so a longer horizon would rely on persistence of the weather.
2. **How do you prevent leakage?**
   - The splits are chronological, and scalers and imputers are fitted on train only.
   - Features use data up to the origin only.
   - A test shows the forecast does not change when data after the origin is corrupted, and does change when data
     before it is corrupted (`tests/test_forecast_api.py`).
3. **What does `confidence = 0.9` mean?**
   - It is the nominal coverage of the interval: over many forecasts, about 90% should contain the observation.
   - It is not the probability that this forecast is right.
   - Measured: 0.893 on Plant 1, CI [0.853, 0.927].
4. **Why CQR per group (Mondrian)?** Errors differ by day vs night (solar) and by power regime (wind). Calibrating
   per group keeps coverage closer to nominal: on Plant 1, 0.893 vs 0.846 for raw quantile regression.
5. **Why is wind worse than persistence?** At 10 minutes the next value is very close to the current one.
   Squared-error models smooth flat hours and the 0 / rated-power limits, and that costs MAE.
6. **Is the RF vs XGBoost difference real?** On Plant 1 yes (bootstrap CI excludes 0); on Plant 2 no.
7. **What would you do next?** Add weather forecasts (the oracle row shows the headroom), use a longer solar record,
   and calibrate per regime with more wind data.

## Demo script (live)

1. Start the API and open `http://127.0.0.1:8000/console/#forecast`. Wait for the header pill to say `ready`.
2. Pick `solar_plant1` and a day, then click **Show day**. Show:
   - the prediction line, the 90% band and the observed values;
   - the coverage KPI for that day;
   - the yellow `in_sample` explanation.
3. Switch the level to 80% and 95%: the band narrows and widens.
4. On the right, show the model comparison against persistence and the measured coverage table (both loaded from
   `GET /results/summary/forecasting`).
5. Switch to `wind_t1`: the comparison panel shows "Persistence wins on MAE here".
6. Click **Check with the decision engine**: REVIEW, because S = 0 (no telemetry check on the served path).
7. Optional: `/docs` -> `POST /forecast` -> "Try it out" with the recorded example.

## What is not done

- No weather-forecast input and no humidity (no source in the data).
- No live telemetry gate in `/forecast`.
- No re-training pipeline or drift monitoring. The deployed artifacts are built once by
  `experiments/build_forecast_model.py`.
- No probabilistic evaluation beyond coverage, Winkler and pinball (e.g. CRPS).
- Wind: no model beats persistence on MAE; not fixed.
