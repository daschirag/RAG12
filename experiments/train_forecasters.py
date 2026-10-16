"""A3: train and compare 1-hour-ahead forecasters on both solar plants.

    python -m experiments.train_forecasters

Every model uses the A2 features, horizon (4 steps = 1 h) and split, the same
train-only imputation, the same post-processing, and is scored on the same rows
(observed targets). Hyperparameters are chosen on val only; test is scored once.

Outputs (results/forecasting/):
- a3_predictions_plant{1,2}.csv  raw per-row predictions (val + test) for every model/seed
- a3_metrics.csv                 per model/seed/split metrics
- a3_summary.md                  mean +/- std over seeds
- a3_run.json                    chosen hyperparameters, fit times, environment
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backend.evaluation.provenance import environment_info
from src.backend.forecasting.features import FeatureConfig, build_features, split_by_target_day
from src.backend.forecasting.metrics import mae, r2, rmse
from src.backend.forecasting.models import (
    DailyPersistence,
    LSTMForecaster,
    ModelSpec,
    Persistence,
    RandomForest,
    TCNForecaster,
    XGBoost,
    eval_mask,
    postprocess,
)
from src.backend.forecasting.preprocess import preprocess_plant

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "forecasting"
SEEDS = (0, 1, 2)
PLANTS = (1, 2)


def model_specs(feature_names: list[str]) -> list[ModelSpec]:
    no_calendar_trend = [f for f in feature_names if f not in ("day", "month")]
    return [
        ModelSpec("persistence", lambda s: Persistence(), (0,), notes="baseline: y(t)"),
        ModelSpec("daily_persistence", lambda s: DailyPersistence(), (0,), notes="baseline: y(T-24h)"),
        ModelSpec("random_forest", lambda s: RandomForest(seed=s), SEEDS),
        ModelSpec("xgboost", lambda s: XGBoost(seed=s), SEEDS),
        ModelSpec("lstm", lambda s: LSTMForecaster(seed=s), SEEDS),
        ModelSpec("tcn", lambda s: TCNForecaster(seed=s), SEEDS, notes="advanced temporal model"),
        ModelSpec("xgboost_no_day_month", lambda s: XGBoost(seed=s, features=no_calendar_trend), SEEDS,
                  notes="ablation: drop day/month"),
        ModelSpec("xgboost_oracle_weather", lambda s: XGBoost(seed=s), SEEDS, oracle=True,
                  notes="UPPER BOUND ONLY: uses measured irradiation at T (future), not deployable"),
    ]


def score(y: np.ndarray, pred: np.ndarray, daytime: np.ndarray) -> dict:
    return {
        "rows": int(len(y)),
        "mae": mae(y, pred),
        "rmse": rmse(y, pred),
        "r2": r2(y, pred),
        "daytime_rows": int(daytime.sum()),
        "daytime_mae": mae(y[daytime], pred[daytime]),
        "daytime_rmse": rmse(y[daytime], pred[daytime]),
    }


def run_plant(plant: int) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    frame = preprocess_plant(plant).frame
    data = build_features(frame, FeatureConfig())
    data_oracle = build_features(frame, FeatureConfig(include_oracle_weather=True))
    if not data.X.index.equals(data_oracle.X.index):
        raise RuntimeError("oracle feature set changed the rows")
    parts = split_by_target_day(data)
    parts_oracle = split_by_target_day(data_oracle)

    pred_frames = []
    for split in ("val", "test"):
        p = parts[split]
        pred_frames.append(pd.DataFrame({
            "origin": p.X.index, "target_time": p.meta["target_time"].to_numpy(), "split": split,
            "y_true": p.y.to_numpy(), "scored": eval_mask(p), "daytime": p.meta["target_daytime"].to_numpy(),
        }))
    preds = pd.concat(pred_frames, ignore_index=True)

    metrics_rows, run_rows = [], []
    for spec in model_specs(data.feature_names):
        use = parts_oracle if spec.oracle else parts
        for seed in spec.seeds:
            model = spec.factory(seed)
            start = time.perf_counter()
            model.fit(use["train"], use["val"], frame)
            fit_seconds = time.perf_counter() - start
            col = f"{spec.label}__seed{seed}"
            values = []
            for split in ("val", "test"):
                p = use[split]
                pred = postprocess(model.predict(p, frame), p)
                values.append(pred)
                m = eval_mask(p)
                day = p.meta["target_daytime"].to_numpy()[m]
                metrics_rows.append({"plant": plant, "model": spec.label, "seed": seed, "split": split,
                                     **score(p.y.to_numpy()[m], pred[m], day)})
            preds[col] = np.concatenate(values)
            run_rows.append({"plant": plant, "model": spec.label, "seed": seed, "notes": spec.notes,
                             "fit_seconds": round(fit_seconds, 2), "params": getattr(model, "params_", {})})
            print(f"plant {plant} {spec.label:<24} seed {seed}  fit {fit_seconds:6.1f}s  "
                  f"test MAE {metrics_rows[-1]['mae']:9.1f}")
    return preds, metrics_rows, run_rows


def _fmt(series: pd.Series, digits: int) -> str:
    if len(series) == 1:
        return f"{series.iloc[0]:.{digits}f}"
    return f"{series.mean():.{digits}f} +/- {series.std(ddof=1):.{digits}f}"


def summary_markdown(metrics: pd.DataFrame, runs: list[dict], env: dict) -> str:
    notes = {(r["model"]): r["notes"] for r in runs}
    lines = [
        "# A3 forecasting model comparison (1 hour ahead, 15-min resolution)",
        "",
        f"Generated by `python -m experiments.train_forecasters` at base commit {env['git_commit']}",
        "(working tree has uncommitted changes by project rule).",
        "Scored rows = observed targets only (night slots filled by A1 are excluded), identical for every model.",
        "Values are mean +/- sample std over seeds 0, 1, 2 (persistence baselines are deterministic).",
        "MAE/RMSE in kW. Skill = 1 - MAE / MAE(persistence) on the same split.",
        "Hyperparameters were chosen on val; test was not used for any choice.",
        "",
    ]
    for plant in sorted(metrics["plant"].unique()):
        m = metrics[metrics["plant"] == plant]
        persist = {s: m[(m.model == "persistence") & (m.split == s)]["mae"].iloc[0] for s in ("val", "test")}
        test = m[m.split == "test"]
        lines += [
            f"## Plant {plant} (test rows scored: {int(test['rows'].iloc[0])}, daytime: "
            f"{int(test['daytime_rows'].iloc[0])})",
            "",
            "| model | val MAE | test MAE | test RMSE | test R2 | test daytime MAE | test skill vs persistence | notes |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for model in m["model"].unique():
            v = m[(m.model == model) & (m.split == "val")]
            t = m[(m.model == model) & (m.split == "test")]
            skill = 1 - t["mae"] / persist["test"]
            lines.append(
                f"| {model} | {_fmt(v['mae'], 1)} | {_fmt(t['mae'], 1)} | {_fmt(t['rmse'], 1)} | "
                f"{_fmt(t['r2'], 4)} | {_fmt(t['daytime_mae'], 1)} | {_fmt(skill, 3)} | {notes[model]} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = environment_info()
    all_metrics, all_runs = [], []
    for plant in PLANTS:
        preds, metrics_rows, run_rows = run_plant(plant)
        preds.to_csv(RESULTS / f"a3_predictions_plant{plant}.csv", index=False, encoding="utf-8")
        all_metrics += metrics_rows
        all_runs += run_rows
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(RESULTS / "a3_metrics.csv", index=False, encoding="utf-8")
    (RESULTS / "a3_summary.md").write_text(summary_markdown(metrics, all_runs, env), encoding="utf-8")
    (RESULTS / "a3_run.json").write_text(
        json.dumps({"runs": all_runs, "environment": env, "seeds": SEEDS, "horizon_steps": 4},
                   indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
