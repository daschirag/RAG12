"""B4 - Loading and normalising the per-event SCADA datasets (CARE-style, Wind Farm A).

Each ``datasets/<event_id>.csv`` holds one turbine's 10-minute SCADA history:
a ``train`` part followed by a ``prediction`` part that contains the event
window (event_start_id .. event_end_id, from event_info.csv).

Decisions (fixed before any model was trained; see docs/fault_diagnosis.md):
- ``status_type_id`` is EXCLUDED everywhere (not a feature, not a row filter).
  In Farm A it is 0% "normal" inside every anomaly window but 100% inside every
  normal window, while rotor speed and power show the turbine running: it
  follows the fault logbook and would leak the label.
- Features are the sensor columns only. Angles become sin/cos.
- Each value is expressed as a robust z-score against the same turbine's own
  ``train`` part (median / interquartile range), which precedes the window and
  uses no labels. Constant columns get scale 1 (their z is 0).
- Rows are labelled 1 inside anomaly windows and 0 inside normal windows; rows
  outside the window are not used for supervised learning or scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..kb.taxonomy import farm_dir, load_events, load_features

META_COLS = ("time_stamp", "asset_id", "id", "train_test", "status_type_id")
EXCLUDED = ("status_type_id",)          # label leakage, see module docstring


def dataset_path(farm: str, event_id: int) -> Path:
    return farm_dir(farm) / "datasets" / f"{event_id}.csv"


STATS = ("avg", "min", "max", "std")


def base_sensor(column: str) -> str:
    """Sensor name of a feature column: 'sensor_18_std' -> 'sensor_18', 'sin_sensor_1_avg' -> 'sensor_1'."""
    for prefix in ("sin_", "cos_"):
        if column.startswith(prefix):
            column = column[len(prefix):]
    head, _, tail = column.rpartition("_")
    return head if head and tail in STATS else column


def sensor_columns(columns, farm: str = "A") -> tuple[list[str], list[str]]:
    """(numeric sensor columns, angle columns) from a dataset header.

    For angle sensors only the average (or the bare column) is treated as an
    angle (encoded as sin/cos); their min/max/std stay ordinary numbers.
    """
    angle_names = set(load_features(farm).query("is_angle")["sensor_name"])
    sensors, angles = [], []
    for c in columns:
        if c in META_COLS:
            continue
        is_angle_value = base_sensor(c) in angle_names and (c.endswith("_avg") or c in angle_names)
        (angles if is_angle_value else sensors).append(c)
    return sensors, angles


@dataclass
class EventData:
    farm: str
    event_id: int
    asset_id: int
    label: str                     # "anomaly" | "normal"
    X: pd.DataFrame                # robust-z features of the window rows (sampled)
    y: np.ndarray                  # 1 anomaly window, 0 normal window
    train_X: pd.DataFrame          # robust-z features of (sampled) train rows, for unsupervised models
    scale: pd.DataFrame            # per-column median and IQR from the train part


def _angle_features(df: pd.DataFrame, angles: list[str]) -> pd.DataFrame:
    out = {}
    for c in angles:
        rad = np.deg2rad(df[c].astype(float))
        out[f"sin_{c}"] = np.sin(rad)
        out[f"cos_{c}"] = np.cos(rad)
    return pd.DataFrame(out, index=df.index)


def robust_scale(train: pd.DataFrame) -> pd.DataFrame:
    med = train.median()
    iqr = train.quantile(0.75) - train.quantile(0.25)
    iqr = iqr.where(iqr > 0, other=1.0).fillna(1.0)
    return pd.DataFrame({"median": med.fillna(0.0), "iqr": iqr})


def load_event(farm: str, event_id: int, row_step: int = 3, train_step: int = 6,
               events: pd.DataFrame | None = None) -> EventData:
    """Read one dataset, normalise against its own train part, return the (sampled) window rows."""
    events = load_events(farm) if events is None else events
    e = events.set_index("event_id").loc[event_id]
    df = pd.read_csv(dataset_path(farm, event_id), sep=";", encoding="utf-8")
    sensors, angles = sensor_columns(df.columns, farm)
    raw = pd.concat([df[sensors].astype(float), _angle_features(df, angles)], axis=1)
    is_train = (df["train_test"] == "train").to_numpy()
    scale = robust_scale(raw[is_train])
    z = ((raw - scale["median"]) / scale["iqr"]).astype("float32")
    z = z.fillna(0.0)                                   # missing reading = no deviation information
    in_window = ((df["id"] >= e["event_start_id"]) & (df["id"] <= e["event_end_id"])).to_numpy()
    win_idx = np.flatnonzero(in_window)[::row_step]
    train_idx = np.flatnonzero(is_train)[::train_step]
    y = np.full(len(win_idx), 1 if e["event_label"] == "anomaly" else 0, dtype=int)
    return EventData(farm=farm, event_id=int(event_id), asset_id=int(e["asset_id"]), label=str(e["event_label"]),
                     X=z.iloc[win_idx].reset_index(drop=True), y=y,
                     train_X=z.iloc[train_idx].reset_index(drop=True), scale=scale)


def load_farm(farm: str = "A", **kwargs) -> list[EventData]:
    events = load_events(farm)
    out = [load_event(farm, int(eid), events=events, **kwargs) for eid in events["event_id"]]
    cols = {tuple(e.X.columns) for e in out}
    if len(cols) != 1:
        raise ValueError(f"datasets of farm {farm} have different feature columns")
    return out
