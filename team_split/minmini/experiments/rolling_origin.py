"""A5: rolling-origin (expanding-window) evaluation of point models and interval methods.

    python -m experiments.rolling_origin            # resumes: skips folds already saved
    python -m experiments.rolling_origin --fresh    # recompute every fold

For each plant and each test day d (12 days, 2020-06-06 .. 2020-06-17):
train = all days before d-8, val = d-8..d-5, calib = d-4..d-1, test = d
(``evaluation.rolling_origin_folds``). Every model is refit per fold with
seeds 0-2 under the strict night rule; interval methods use that fold's
XGBoost point model and a multi-quantile XGBoost, calibrated on the fold's
calib days only.

Raw per-row output: results/forecasting/a5_folds/plant{p}_{YYYY-MM-DD}.csv
(point predictions per model/seed and interval bounds per method/level/seed).
Aggregation: ``python -m experiments.evaluate_forecasting``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.backend.evaluation.provenance import environment_info
from src.backend.forecasting.evaluation import FoldSpec, rolling_origin_folds
from src.backend.forecasting.features import build_features
from src.backend.forecasting.models import (
    DEFAULT_NIGHT_THRESHOLD,
    DailyPersistence,
    LSTMForecaster,
    Persistence,
    RandomForest,
    TCNForecaster,
    XGBoost,
    eval_mask,
    postprocess,
)
from src.backend.forecasting.preprocess import preprocess_plant
from src.backend.forecasting.uncertainty import LEVELS, QuantileXGBoost, build_intervals, quantiles_for

ROOT = Path(__file__).resolve().parents[1]
FOLD_DIR = ROOT / "results" / "forecasting" / "a5_folds"
SEEDS = (0, 1, 2)
PLANTS = (1, 2)
POINT_MODELS = ("persistence", "daily_persistence", "random_forest", "xgboost",
                "xgboost_no_day_month", "lstm", "tcn")


def make_model(name: str, seed: int, feature_names: list[str]):
    if name == "persistence":
        return Persistence()
    if name == "daily_persistence":
        return DailyPersistence()
    if name == "random_forest":
        return RandomForest(seed=seed)
    if name == "xgboost":
        return XGBoost(seed=seed)
    if name == "xgboost_no_day_month":
        return XGBoost(seed=seed, features=[f for f in feature_names if f not in ("day", "month")])
    if name == "lstm":
        return LSTMForecaster(seed=seed)
    if name == "tcn":
        return TCNForecaster(seed=seed)
    raise ValueError(name)


def run_fold(frame, parts) -> tuple[pd.DataFrame, dict]:
    train, val, calib, test = (parts[k] for k in ("train", "val", "calib", "test"))
    base = pd.DataFrame({
        "origin": test.X.index, "target_time": test.meta["target_time"].to_numpy(),
        "y_true": test.y.to_numpy(), "scored": eval_mask(test),
        "daylight": test.meta["target_daytime"].to_numpy(),
    })
    # Collect prediction columns in a dict and join once (avoids ~100 incremental inserts).
    cols: dict[str, object] = {}
    fit_seconds: dict[str, float] = {}
    names = train.feature_names
    for name in POINT_MODELS:
        seeds = (0,) if name in ("persistence", "daily_persistence") else SEEDS
        for seed in seeds:
            start = time.perf_counter()
            model = make_model(name, seed, names).fit(train, val, frame)
            fit_seconds[f"{name}__seed{seed}"] = round(time.perf_counter() - start, 2)
            point = postprocess(model.predict(test, frame), test)
            cols[f"{name}__seed{seed}"] = point
            if name == "xgboost":
                qr = QuantileXGBoost(quantiles_for(LEVELS), seed=seed).fit(train, val)
                pc = postprocess(model.predict(calib, frame), calib)
                ints = build_intervals(pc, qr.predict(calib), calib, point, qr.predict(test), test, LEVELS)
                for (method, level), (lo, hi) in ints.bounds.items():
                    cols[f"{method}_{int(level * 100)}_lo__seed{seed}"] = lo
                    cols[f"{method}_{int(level * 100)}_hi__seed{seed}"] = hi
    return pd.concat([base, pd.DataFrame(cols, index=base.index)], axis=1), fit_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="recompute folds that already exist")
    args = parser.parse_args()
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    env = environment_info()
    spec = FoldSpec()
    log = {"environment": env, "fold_spec": spec.__dict__, "seeds": SEEDS,
           "night_threshold": DEFAULT_NIGHT_THRESHOLD, "folds": []}
    for plant in PLANTS:
        frame = preprocess_plant(plant).frame
        data = build_features(frame)
        for test_day, parts in rolling_origin_folds(data, spec):
            path = FOLD_DIR / f"plant{plant}_{test_day:%Y-%m-%d}.csv"
            if path.exists() and not args.fresh:
                print(f"plant {plant} {test_day:%Y-%m-%d}: exists, skipped")
                continue
            start = time.perf_counter()
            out, fit_seconds = run_fold(frame, parts)
            out.to_csv(path, index=False, encoding="utf-8")
            log["folds"].append({"plant": plant, "test_day": f"{test_day:%Y-%m-%d}",
                                 "train_rows": len(parts["train"].y), "fit_seconds": fit_seconds})
            print(f"plant {plant} {test_day:%Y-%m-%d}: train rows {len(parts['train'].y)}, "
                  f"{time.perf_counter() - start:.0f}s")
    (FOLD_DIR / "run_log.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
