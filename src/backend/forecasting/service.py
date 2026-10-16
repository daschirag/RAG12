"""A6 - Forecasting service behind POST /forecast (solar plants 1-2 and wind turbine T1).

Deployed model per asset (the A4/A5 proposed configuration):
XGBoost point forecast + multi-quantile XGBoost + CQR calibrated per group
(``cqr_mondrian``), horizon 1 hour. Solar: day/night groups and the strict
night rule. Wind: power-regime groups of the point forecast, no night rule.

Deployment split follows the rolling-origin protocol aimed at the period after
the record. Solar: train = all days except the last 8, val = the 4 days before
the last 4, calib = the last 4 days. Wind: train Jan-Oct, val Nov, calib Dec
2018. Build once with ``python -m experiments.build_forecast_model``; the API
loads the artifact on first use.

Artifacts are pickled with joblib and are only ever loaded from the local
``models/forecasting`` folder written by the build script (pickle files must
never be loaded from untrusted sources).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .evaluation import target_days
from .features import build_features, SupervisedSet
from .models import DEFAULT_NIGHT_THRESHOLD, PLANT_CAPACITY_KW, XGBoost, eval_mask, night_mask, postprocess
from .preprocess import preprocess_plant
from .uncertainty import LEVELS, PROPOSED, QuantileXGBoost, cqr_offsets, finalize_interval, quantiles_for
from .wind import (
    REGIMES,
    WIND_CAPACITY_KW,
    WIND_STEP,
    build_wind_features,
    deployment_month_split,
    power_regime,
    preprocess_wind_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "forecasting"
PLANTS = (1, 2)
HORIZON_MINUTES = 60


class ModelNotBuiltError(RuntimeError):
    """Raised when the forecast artifact for an asset does not exist."""


class OriginError(ValueError):
    """Raised when the requested forecast origin cannot be served."""


@dataclass(frozen=True)
class AssetSpec:
    kind: str                 # "solar" or "wind"
    step: pd.Timedelta        # native grid of the origin
    capacity_kw: float
    plant: int | None = None


ASSETS: dict[str, AssetSpec] = {
    "solar_plant1": AssetSpec("solar", pd.Timedelta("15min"), PLANT_CAPACITY_KW, plant=1),
    "solar_plant2": AssetSpec("solar", pd.Timedelta("15min"), PLANT_CAPACITY_KW, plant=2),
    "wind_t1": AssetSpec("wind", WIND_STEP, WIND_CAPACITY_KW),
}


def asset_id(plant_or_asset: int | str) -> str:
    """Accept the legacy solar plant number (1, 2) or an asset id."""
    if isinstance(plant_or_asset, str):
        if plant_or_asset not in ASSETS:
            raise OriginError(f"Unknown asset {plant_or_asset!r}; expected one of {list(ASSETS)}")
        return plant_or_asset
    if plant_or_asset not in PLANTS:
        raise OriginError(f"Unknown plant {plant_or_asset}; expected one of {PLANTS}")
    return f"solar_plant{plant_or_asset}"


@dataclass
class PlantModel:
    plant: int | None
    point: XGBoost
    quantile: QuantileXGBoost
    offsets: dict[float, dict]                       # level -> {calibration group: q}
    night_threshold: float | None
    feature_config: object
    segments: dict[str, tuple[str, str]]             # split -> (first, last) target day
    info: dict = field(default_factory=dict)
    asset: str = ""
    kind: str = "solar"
    capacity_kw: float = PLANT_CAPACITY_KW


def _subsets(data: SupervisedSet, masks: dict) -> dict[str, SupervisedSet]:
    return {k: SupervisedSet(X=data.X[m], y=data.y[m], meta=data.meta[m], config=data.config)
            for k, m in masks.items()}


def deployment_split(data: SupervisedSet, val_days: int = 4, calib_days: int = 4) -> dict[str, SupervisedSet]:
    """train | val | calib ending at the last day of the record (no test day)."""
    days = target_days(data)
    day_of_row = data.meta["target_time"].dt.normalize()
    cal_start = len(days) - calib_days
    val_start = cal_start - val_days
    segs = {"train": days[:val_start], "val": days[val_start:cal_start], "calib": days[cal_start:]}
    return _subsets(data, {k: day_of_row.isin(d).to_numpy() for k, d in segs.items()})


def _span(p: SupervisedSet) -> tuple[str, str]:
    t = p.meta["target_time"].dt.normalize()
    return (f"{t.min():%Y-%m-%d}", f"{t.max():%Y-%m-%d}")


def _solar_groups(data: SupervisedSet, point):
    return data.meta["target_daytime"].to_numpy()


def _wind_groups(data: SupervisedSet, point):
    return power_regime(point)


def _fit_and_calibrate(parts, frame, point, qr, group_fn, groups, capacity, threshold):
    """Fit point + quantile models on train/val; per-group CQR offsets on calib (scored rows)."""
    train, val, calib = parts["train"], parts["val"], parts["calib"]
    point.fit(train, val, frame)
    qr.fit(train, val)
    m = eval_mask(calib)
    y_cal = calib.y.to_numpy()[m]
    pc = postprocess(point.predict(calib, frame), calib, threshold, capacity)
    g_cal = group_fn(calib, pc)[m]
    q_cal = qr.predict(calib)
    offsets = {}
    for level in LEVELS:
        a = 1 - level
        lo_c = q_cal[round(a / 2, 6)].to_numpy()[m]
        hi_c = q_cal[round(1 - a / 2, 6)].to_numpy()[m]
        offsets[level] = cqr_offsets(lo_c, hi_c, y_cal, g_cal, level, groups)
    counts = {str(g): int((g_cal == g).sum()) for g in groups}
    return offsets, counts


def _info(seed, point, qr, counts) -> dict:
    return {"trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": seed,
            "method": PROPOSED, "point_params": point.params_, "quantile_params": qr.params_,
            "calib_rows": counts}


def build_plant_model(plant: int, frame: pd.DataFrame | None = None, seed: int = 0,
                      point_grid: dict | None = None, quantile_grid: dict | None = None) -> PlantModel:
    """Train and calibrate the deployed model for one solar plant (grids overridable for tests)."""
    frame = preprocess_plant(plant).frame if frame is None else frame
    data = build_features(frame)
    parts = deployment_split(data)
    point = XGBoost(seed=seed)
    qr = QuantileXGBoost(quantiles_for(LEVELS), seed=seed)
    if point_grid is not None:
        point.grid = point_grid
    if quantile_grid is not None:
        qr.grid = quantile_grid
    offsets, counts = _fit_and_calibrate(parts, frame, point, qr, _solar_groups, (True, False),
                                         PLANT_CAPACITY_KW, DEFAULT_NIGHT_THRESHOLD)
    counts = {"daylight": counts["True"], "night": counts["False"]}
    return PlantModel(
        plant=plant, point=point, quantile=qr, offsets=offsets, night_threshold=DEFAULT_NIGHT_THRESHOLD,
        feature_config=data.config, segments={k: _span(v) for k, v in parts.items()},
        info=_info(seed, point, qr, counts), asset=f"solar_plant{plant}", kind="solar",
        capacity_kw=PLANT_CAPACITY_KW,
    )


def build_wind_model(frame: pd.DataFrame | None = None, seed: int = 0,
                     point_grid: dict | None = None, quantile_grid: dict | None = None) -> PlantModel:
    """Deployed wind T1 model: train Jan-Oct, val Nov, calib Dec 2018; power-regime CQR groups."""
    frame = preprocess_wind_file()[0] if frame is None else frame
    data = build_wind_features(frame)
    parts = deployment_month_split(data)
    point = XGBoost(seed=seed, night_threshold=None, capacity=WIND_CAPACITY_KW)
    qr = QuantileXGBoost(quantiles_for(LEVELS), seed=seed)
    if point_grid is not None:
        point.grid = point_grid
    if quantile_grid is not None:
        qr.grid = quantile_grid
    offsets, counts = _fit_and_calibrate(parts, frame, point, qr, _wind_groups, REGIMES, WIND_CAPACITY_KW, None)
    return PlantModel(
        plant=None, point=point, quantile=qr, offsets=offsets, night_threshold=None,
        feature_config=data.config, segments={k: _span(v) for k, v in parts.items()},
        info=_info(seed, point, qr, counts), asset="wind_t1", kind="wind", capacity_kw=WIND_CAPACITY_KW,
    )


def artifact_path(plant_or_asset: int | str, model_dir: Path) -> Path:
    return model_dir / f"{asset_id(plant_or_asset)}.joblib"


def save_plant_model(model: PlantModel, model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    import joblib

    model_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_path(model.asset or model.plant, model_dir)
    joblib.dump(model, path)
    return path


class ForecastService:
    """Loads asset models lazily and serves 1-hour-ahead interval forecasts."""

    def __init__(self, model_dir: Path | str | None = None,
                 frames: dict[int | str, pd.DataFrame] | None = None) -> None:
        env_dir = os.getenv("GREENRAG_FORECAST_MODEL_DIR")
        self.model_dir = Path(model_dir or env_dir or DEFAULT_MODEL_DIR)
        self._frames = {asset_id(k): v for k, v in (frames or {}).items()}
        self._models: dict[str, PlantModel] = {}
        self._features: dict[str, SupervisedSet] = {}
        self._lock = threading.Lock()

    def _model(self, plant_or_asset: int | str) -> PlantModel:
        asset = asset_id(plant_or_asset)
        with self._lock:
            if asset not in self._models:
                path = artifact_path(asset, self.model_dir)
                if not path.is_file():
                    raise ModelNotBuiltError(
                        f"Forecast model for {asset} not found at {path}. "
                        "Build it with: python -m experiments.build_forecast_model")
                import joblib

                self._models[asset] = joblib.load(path)
            return self._models[asset]

    def _feature_rows(self, asset: str, cfg) -> SupervisedSet:
        with self._lock:
            if asset not in self._features:
                spec = ASSETS[asset]
                frame = self._frames.get(asset)
                if frame is None:
                    frame = preprocess_plant(spec.plant).frame if spec.kind == "solar" else preprocess_wind_file()[0]
                    self._frames[asset] = frame
                build = build_features if spec.kind == "solar" else build_wind_features
                self._features[asset] = build(frame, cfg, drop_missing_target=False)
            return self._features[asset]

    def forecast(self, plant_or_asset: int | str, origin: pd.Timestamp, confidence: float = 0.9) -> dict:
        asset = asset_id(plant_or_asset)
        spec = ASSETS[asset]
        if confidence not in LEVELS:
            raise OriginError(f"confidence must be one of {LEVELS}")
        model = self._model(asset)
        rows = self._feature_rows(asset, model.feature_config)
        if origin not in rows.X.index:
            first, last = rows.X.index[0], rows.X.index[-1]
            minutes = int(spec.step / pd.Timedelta("1min"))
            raise OriginError(f"origin {origin} is not a {minutes}-min slot between {first} and {last}")
        one = SupervisedSet(X=rows.X.loc[[origin]], y=rows.y.loc[[origin]], meta=rows.meta.loc[[origin]],
                            config=rows.config)
        frame = self._frames[asset]
        point = postprocess(model.point.predict(one, frame), one, model.night_threshold, model.capacity_kw)
        q = model.quantile.predict(one)
        a = 1 - confidence
        if spec.kind == "solar":
            group = bool(one.meta["target_daytime"].iloc[0])
            group_label = "daylight" if group else "night"
        else:
            group = str(power_regime(point)[0])
            group_label = f"power regime {group}"
        off = model.offsets[confidence][group]
        lo = q[round(a / 2, 6)].to_numpy() - off
        hi = q[round(1 - a / 2, 6)].to_numpy() + off
        lo, hi = finalize_interval(lo, hi, point, night_mask(one, model.night_threshold), model.capacity_kw)

        target_time = one.meta["target_time"].iloc[0]
        observed = one.y.iloc[0]
        first_fit_day, last_fit_day = model.segments["train"][0], model.segments["calib"][1]
        return {
            "prediction": float(point[0]), "lower_bound": float(lo[0]), "upper_bound": float(hi[0]),
            "confidence": confidence, "asset": asset, "plant": spec.plant, "origin": origin.to_pydatetime(),
            "target_time": target_time.to_pydatetime(), "horizon_minutes": HORIZON_MINUTES,
            "observed_kw": None if pd.isna(observed) else float(observed),
            "in_sample": f"{target_time:%Y-%m-%d}" <= last_fit_day,
            "calibration_group": group_label,
            "daylight_group": group if spec.kind == "solar" else None,
            "method": PROPOSED, "unit": "kW",
            "model_version": f"{asset} {model.info.get('trained_at_utc', '')} "
                             f"(fit on {first_fit_day}..{last_fit_day})",
        }


def _series_rows(service: "ForecastService", asset: str, start: pd.Timestamp, end: pd.Timestamp,
                 confidence: float) -> list[dict]:
    import numpy as np

    spec = ASSETS[asset]
    if confidence not in LEVELS:
        raise OriginError(f"confidence must be one of {LEVELS}")
    model = service._model(asset)
    rows = service._feature_rows(asset, model.feature_config)
    idx = rows.X.index[(rows.X.index >= start) & (rows.X.index <= end)]
    if not len(idx):
        raise OriginError(f"no {asset} forecast origins between {start} and {end}; the record covers "
                          f"{rows.X.index[0]} to {rows.X.index[-1]}")
    sub = SupervisedSet(X=rows.X.loc[idx], y=rows.y.loc[idx], meta=rows.meta.loc[idx], config=rows.config)
    point = postprocess(model.point.predict(sub, service._frames[asset]), sub, model.night_threshold,
                        model.capacity_kw)
    q = model.quantile.predict(sub)
    a = 1 - confidence
    if spec.kind == "solar":
        groups = [bool(g) for g in sub.meta["target_daytime"]]
        labels = ["daylight" if g else "night" for g in groups]
    else:
        groups = [str(g) for g in power_regime(point)]
        labels = [f"power regime {g}" for g in groups]
    off = np.array([model.offsets[confidence][g] for g in groups], dtype=float)
    lo = q[round(a / 2, 6)].to_numpy() - off
    hi = q[round(1 - a / 2, 6)].to_numpy() + off
    lo, hi = finalize_interval(lo, hi, point, night_mask(sub, model.night_threshold), model.capacity_kw)
    last_fit_day = model.segments["calib"][1]
    out = []
    for i, origin in enumerate(idx):
        target = sub.meta["target_time"].iloc[i]
        y = sub.y.iloc[i]
        out.append({"origin": origin.to_pydatetime(), "target_time": target.to_pydatetime(),
                    "prediction": float(point[i]), "lower_bound": float(lo[i]), "upper_bound": float(hi[i]),
                    "observed_kw": None if pd.isna(y) else float(y), "calibration_group": labels[i],
                    "in_sample": f"{target:%Y-%m-%d}" <= last_fit_day})
    return out


def _asset_info(service: "ForecastService", asset: str) -> dict:
    spec = ASSETS[asset]
    model = service._model(asset)
    rows = service._feature_rows(asset, model.feature_config)
    return {"asset": asset, "kind": spec.kind, "plant": spec.plant,
            "step_minutes": int(spec.step / pd.Timedelta("1min")), "horizon_minutes": HORIZON_MINUTES,
            "capacity_kw": float(model.capacity_kw), "first_origin": rows.X.index[0].to_pydatetime(),
            "last_origin": rows.X.index[-1].to_pydatetime(),
            "segments": {k: list(v) for k, v in model.segments.items()}, "levels": list(LEVELS),
            "method": PROPOSED, "trained_at_utc": model.info.get("trained_at_utc", "")}


ForecastService.series = lambda self, asset, start, end, confidence=0.9: _series_rows(  # type: ignore[attr-defined]
    self, asset_id(asset), pd.Timestamp(start), pd.Timestamp(end), confidence)
ForecastService.asset_info = lambda self, asset: _asset_info(self, asset_id(asset))  # type: ignore[attr-defined]
