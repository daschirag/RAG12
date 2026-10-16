"""B4 - Anomaly detection, fault probability and component attribution (Wind Farm A).

Protocol (fixed before training; docs/fault_diagnosis.md):
- Supervised models (logistic regression, random forest, XGBoost) are trained
  on window rows of all *other* turbines and applied to the held-out turbine
  (leave-one-asset-out): no turbine is in train and test at once.
- Isolation Forest is unsupervised and fitted per dataset on that turbine's
  own train part (no labels); a row is flagged above the 99th percentile of
  its training scores.
- Event probability = mean row score over the window; an event is called
  anomalous when it is >= 0.5 (supervised: mean probability; Isolation
  Forest: fraction of flagged rows).
- Component attribution is unsupervised: rank taxonomy components by the mean
  absolute robust z-score of their sensor group inside the window.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from ..kb.taxonomy import COMPONENTS, load_features, sensors_for
from .data import EventData, base_sensor

Z_CLIP = 50.0                    # robust z beyond +/-50 carries no extra information and destabilises LR
EVENT_THRESHOLD = 0.5
IF_QUANTILE = 0.99


def make_logreg(seed: int):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=0.1, max_iter=2000, random_state=seed)


def make_random_forest(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed)


def make_xgboost(seed: int):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, n_jobs=4, eval_metric="logloss", tree_method="hist")


SUPERVISED: dict[str, Callable] = {"logistic_regression": make_logreg, "random_forest": make_random_forest,
                                   "xgboost": make_xgboost}


def _matrix(events: list[EventData]) -> tuple[np.ndarray, np.ndarray]:
    X = np.clip(np.vstack([e.X.to_numpy() for e in events]), -Z_CLIP, Z_CLIP)
    return X, np.concatenate([e.y for e in events])


def leave_one_asset_out(events: list[EventData], name: str, seed: int = 0) -> pd.DataFrame:
    """Row-level out-of-fold probabilities for every window row."""
    rows = []
    for asset in sorted({e.asset_id for e in events}):
        train = [e for e in events if e.asset_id != asset]
        test = [e for e in events if e.asset_id == asset]
        X, y = _matrix(train)
        model = SUPERVISED[name](seed)
        model.fit(X, y)
        if name == "random_forest":
            model.set_params(n_jobs=1)          # deterministic summation order at predict time
        for e in test:
            p = model.predict_proba(np.clip(e.X.to_numpy(), -Z_CLIP, Z_CLIP))[:, 1]
            rows.append(pd.DataFrame({"event_id": e.event_id, "asset_id": e.asset_id, "y": e.y, "score": p,
                                      "flag": (p >= 0.5).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def isolation_forest(events: list[EventData], seed: int = 0) -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest

    rows = []
    for e in events:
        model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=1)
        model.fit(np.clip(e.train_X.to_numpy(), -Z_CLIP, Z_CLIP))
        train_scores = -model.score_samples(np.clip(e.train_X.to_numpy(), -Z_CLIP, Z_CLIP))
        thr = np.quantile(train_scores, IF_QUANTILE)
        s = -model.score_samples(np.clip(e.X.to_numpy(), -Z_CLIP, Z_CLIP))
        rows.append(pd.DataFrame({"event_id": e.event_id, "asset_id": e.asset_id, "y": e.y, "score": s,
                                  "flag": (s > thr).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def classification_metrics(y, flag, score) -> dict[str, float]:
    y, flag, score = map(np.asarray, (y, flag, score))
    return {"n": int(len(y)), "accuracy": accuracy_score(y, flag),
            "precision": precision_score(y, flag, zero_division=0), "recall": recall_score(y, flag, zero_division=0),
            "f1": f1_score(y, flag, zero_division=0),
            "auc": roc_auc_score(y, score) if len(set(y)) == 2 else float("nan")}


def event_table(rows: pd.DataFrame, model: str) -> pd.DataFrame:
    """One row per event: mean score (supervised: probability; IF: raw score) and mean flag."""
    g = rows.groupby("event_id").agg(asset_id=("asset_id", "first"), y=("y", "first"),
                                     mean_score=("score", "mean"), flag_rate=("flag", "mean")).reset_index()
    # Event probability: mean probability for supervised models, fraction of flagged rows for Isolation Forest.
    g["event_prob"] = g["flag_rate"] if model == "isolation_forest" else g["mean_score"]
    g["event_flag"] = (g["event_prob"] >= EVENT_THRESHOLD).astype(int)
    return g


def bootstrap_event_metric(events: pd.DataFrame, metric: str, n_resamples: int = 10_000, seed: int = 0):
    """Percentile CI over events (resamples with one class only are skipped for AUC)."""
    rng = np.random.default_rng(seed)
    vals = []
    y, flag, prob = events["y"].to_numpy(), events["event_flag"].to_numpy(), events["event_prob"].to_numpy()
    for _ in range(n_resamples):
        i = rng.integers(0, len(y), len(y))
        if metric == "auc":
            if len(set(y[i])) < 2:
                continue
            vals.append(roc_auc_score(y[i], prob[i]))
        else:
            vals.append(classification_metrics(y[i], flag[i], prob[i])[metric])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def select_threshold(scores, y) -> float:
    """Event threshold t (event flagged if score > t) maximising F1 on labelled validation events.

    Candidates: below the minimum, midpoints between sorted unique scores, and the maximum.
    Ties are broken towards the higher threshold (fewer alarms). Rule fixed before use.
    """
    s, y = np.asarray(scores, dtype=float), np.asarray(y, dtype=int)
    u = np.unique(s)
    candidates = np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1]]])
    best_t, best_f1 = candidates[-1], -1.0
    for t in candidates:
        f1 = f1_score(y, (s > t).astype(int), zero_division=0)
        if f1 > best_f1 or (f1 == best_f1 and t > best_t):
            best_t, best_f1 = float(t), f1
    return best_t


def nested_threshold_eval(events: pd.DataFrame, score_col: str = "event_prob") -> pd.DataFrame:
    """Leave-one-asset-out threshold selection: each asset's events use a threshold chosen on the other assets."""
    out = []
    for asset in sorted(events["asset_id"].unique()):
        val = events[events["asset_id"] != asset]
        test = events[events["asset_id"] == asset].copy()
        t = select_threshold(val[score_col], val["y"])
        test["threshold"] = t
        test["event_flag_nested"] = (test[score_col] > t).astype(int)
        out.append(test)
    return pd.concat(out, ignore_index=True)


def component_columns(columns, farm: str = "A") -> dict[str, list[str]]:
    """Feature columns belonging to each taxonomy component (via B1's sensor mapping)."""
    feats = load_features(farm)
    out = {}
    for comp in COMPONENTS:
        names = set(sensors_for(comp, feats)["sensor_name"])
        cols = [c for c in columns if base_sensor(c) in names]
        if cols:
            out[comp.key] = cols
    return out


def attribute_components(event: EventData, farm: str = "A") -> pd.Series:
    """Components ranked by mean |robust z| of their sensors inside the window (largest first)."""
    groups = component_columns(event.X.columns, farm)
    z = event.X.abs().clip(upper=Z_CLIP)
    score = {k: float(z[cols].to_numpy().mean()) for k, cols in groups.items()}
    return pd.Series(score).sort_values(ascending=False, kind="mergesort")
