"""Wind T1 replication of Track A: rolling-origin evaluation by month (resumable).

    python -m experiments.wind_rolling            # skips (fold, model, seed) files already saved
    python -m experiments.wind_rolling --fresh

Test months 2018-09 .. 2018-12. For test month m: train = all months before
m-2, val = m-2 (hyperparameters / early stopping), calib = m-1 (interval
calibration), test = m. Horizon 1 hour (6 x 10 min). Seeds 0-2.

Output: results/forecasting/wind/folds/<YYYY-MM>/base.csv and one CSV per
model and seed (``<model>__seed<k>.csv``). The xgboost file also holds the
interval bounds of every method and level. Aggregate with
``python -m experiments.wind_evaluate``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.backend.evaluation.provenance import environment_info
from src.backend.forecasting.models import (
    DailyPersistence,
    LSTMForecaster,
    Persistence,
    RandomForest,
    TCNForecaster,
    TrainConfig,
    XGBoost,
    eval_mask,
    postprocess,
)
from src.backend.forecasting.uncertainty import LEVELS, QuantileXGBoost, build_intervals, quantiles_for
from src.backend.forecasting.wind import (
    WIND_CAPACITY_KW,
    WIND_CONTEXT_COLS,
    WIND_SEQ_COLUMNS,
    WindFeatureConfig,
    build_wind_features,
    power_regime,
    preprocess_wind_file,
    rolling_origin_month_folds,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "forecasting" / "wind" / "folds"
TEST_MONTHS = ["2018-09", "2018-10", "2018-11", "2018-12"]
SEEDS = (0, 1, 2)
KW = {"night_threshold": None, "capacity": WIND_CAPACITY_KW}          # wind: no night rule
# Scaled for ~15x more data than solar (CPU only, 16 GB RAM); documented in docs/forecasting.md.
RF_GRID = {"max_depth": [12, 20], "min_samples_leaf": [1, 5], "max_features": [0.5]}
RF_TREES = 200
NN_CFG = TrainConfig(lookback=24, hidden=32, batch_size=256, max_epochs=60, patience=8)
MODELS = ("persistence", "daily_persistence", "random_forest", "xgboost", "lstm", "tcn", "xgboost_oracle_weather")


def make(name: str, seed: int):
    if name == "persistence":
        return Persistence()
    if name == "daily_persistence":
        return DailyPersistence()
    if name == "random_forest":
        rf = RandomForest(seed=seed, **KW)
        rf.grid, rf.n_estimators = RF_GRID, RF_TREES
        return rf
    if name in ("xgboost", "xgboost_oracle_weather"):
        return XGBoost(seed=seed, **KW)
    seq = {"train_cfg": NN_CFG, "seq_columns": WIND_SEQ_COLUMNS, "context_cols": WIND_CONTEXT_COLS, **KW}
    if name == "lstm":
        return LSTMForecaster(seed=seed, **seq)
    if name == "tcn":
        return TCNForecaster(seed=seed, **seq)
    raise ValueError(name)


def run_one(name, seed, frame, parts) -> pd.DataFrame:
    train, val, calib, test = (parts[k] for k in ("train", "val", "calib", "test"))
    model = make(name, seed).fit(train, val, frame)
    point = postprocess(model.predict(test, frame), test, None, WIND_CAPACITY_KW)
    cols = {"pred": point}
    if name == "xgboost":
        qr = QuantileXGBoost(quantiles_for(LEVELS), seed=seed).fit(train, val)
        pc = postprocess(model.predict(calib, frame), calib, None, WIND_CAPACITY_KW)
        ints = build_intervals(pc, qr.predict(calib), calib, point, qr.predict(test), test, LEVELS,
                               night_threshold=None, groups_cal=power_regime(pc), groups_te=power_regime(point),
                               capacity=WIND_CAPACITY_KW)
        cols["regime"] = power_regime(point)
        for (method, level), (lo, hi) in ints.bounds.items():
            cols[f"{method}_{int(level * 100)}_lo"] = lo
            cols[f"{method}_{int(level * 100)}_hi"] = hi
    return pd.DataFrame(cols, index=test.X.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame, _ = preprocess_wind_file()
    data = build_wind_features(frame, WindFeatureConfig())
    data_oracle = build_wind_features(frame, WindFeatureConfig(include_oracle_weather=True))
    folds = dict(rolling_origin_month_folds(data, TEST_MONTHS))
    folds_oracle = dict(rolling_origin_month_folds(data_oracle, TEST_MONTHS))
    log = []
    for month, parts in folds.items():
        d = OUT / str(month)
        d.mkdir(exist_ok=True)
        test = parts["test"]
        pd.DataFrame({"origin": test.X.index, "target_time": test.meta["target_time"].to_numpy(),
                      "y_true": test.y.to_numpy(), "scored": eval_mask(test),
                      "down": test.meta["target_down"].to_numpy()}).to_csv(d / "base.csv", index=False,
                                                                          encoding="utf-8")
        for name in MODELS:
            use = folds_oracle[month] if name == "xgboost_oracle_weather" else parts
            for seed in ((0,) if "persistence" in name else SEEDS):
                path = d / f"{name}__seed{seed}.csv"
                if path.exists() and not args.fresh:
                    continue
                start = time.perf_counter()
                out = run_one(name, seed, frame, use)
                out.to_csv(path, index_label="origin", encoding="utf-8")
                secs = time.perf_counter() - start
                log.append({"month": str(month), "model": name, "seed": seed, "fit_seconds": round(secs, 1),
                            "train_rows": len(use["train"].y)})
                print(f"{month} {name:<24} seed {seed}: {secs:6.0f}s", flush=True)
    (OUT / f"run_log_{pd.Timestamp.now():%Y%m%d_%H%M%S}.json").write_text(
        json.dumps({"environment": environment_info(), "runs": log}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
