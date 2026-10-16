"""Track A replicated on wind turbine T1 (dataset/wind/T1.csv).

Reuses the solar pipeline (models, uncertainty, evaluation, API) with the
wind-specific pieces kept here:

- preprocessing: one turbine, native 10-min grid, no plant aggregation;
  small negative power (auxiliary consumption, >= -10 kW) clipped to 0 and
  flagged; downtime/curtailment (power ~0 while the theoretical curve says
  >= 500 kW) flagged but kept, like the solar statistical outliers.
- features: 1-hour horizon (6 steps); wind speed/direction/theoretical power
  at or before the origin instead of irradiation; no night rule.
- splits: whole calendar months of the target time (a full year of data).
- Mondrian calibration groups: power regime of the *point forecast*
  (known at the origin) instead of solar day/night.

Every step is causal (tested as for solar).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features import SupervisedSet
from .preprocess import TARGET, causal_smooth, remove_duplicates, robust_trailing_z, snap_to_grid

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WIND_FILE = PROJECT_ROOT / "dataset" / "wind" / "T1.csv"

WIND_STEP = pd.Timedelta("10min")
WIND_STEPS_PER_DAY = int(pd.Timedelta("1D") / WIND_STEP)      # 144
RATED_KW = 3600.0                  # theoretical power curve saturates at exactly 3600 kW
WIND_CAPACITY_KW = RATED_KW * 1.05  # clip limit; the largest measured value is 3618.7 kW
TIME_FORMAT = "%d %m %Y %H:%M"
RAW_RENAME = {
    "LV ActivePower (kW)": TARGET,
    "Wind Speed (m/s)": "wind_speed",
    "Theoretical_Power_Curve (KWh)": "theoretical_power",
}
WEATHER_VARS = ("wind_speed", "wind_dir", "theoretical_power")


# ---------------------------------------------------------------- preprocessing

@dataclass(frozen=True)
class WindPreprocessConfig:
    sync_tolerance: pd.Timedelta = pd.Timedelta("2min")
    negative_clip_kw: float = -10.0          # [-10, 0) -> 0 (auxiliary consumption); below -> NaN
    power_max_kw: float = WIND_CAPACITY_KW
    wind_speed_range: tuple[float, float] = (0.0, 40.0)
    weather_ffill_limit: int = 2             # 20 min
    down_power_kw: float = 10.0              # "down": power <= this ...
    down_theoretical_kw: float = 500.0       # ... while the power curve says >= this
    stat_window: str = "7D"
    stat_min_periods: int = 144
    stat_z_threshold: float = 5.0
    smooth_window: int = 3


@dataclass
class WindPreprocessReport:
    raw_rows: int = 0
    duplicate_rows_removed: int = 0
    conflicting_keys_removed: int = 0
    offgrid_snapped: int = 0
    offgrid_dropped: int = 0
    grid_start: str = ""
    grid_end: str = ""
    grid_slots: int = 0
    slots_missing: int = 0
    longest_gap: str = ""
    negative_clipped: int = 0
    bound_violations: dict[str, int] = field(default_factory=dict)
    weather_ffilled: int = 0
    target_missing_final: int = 0
    turbine_down_flagged: int = 0
    stat_outliers_flagged: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def load_wind_raw(path: Path = WIND_FILE) -> pd.DataFrame:
    """Read T1.csv (UTF-8 BOM, ``dd mm yyyy HH:MM``) with explicit parsing."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    dir_col = [c for c in raw.columns if c.startswith("Wind Direction")]
    missing = [c for c in ["Date/Time", *RAW_RENAME] if c not in raw.columns]
    if missing or len(dir_col) != 1:
        raise ValueError(f"T1.csv is missing columns: {missing or 'Wind Direction'}")
    out = raw.rename(columns={**RAW_RENAME, dir_col[0]: "wind_dir"})
    out["timestamp"] = pd.to_datetime(out["Date/Time"], format=TIME_FORMAT)
    return out[["timestamp", TARGET, "wind_speed", "wind_dir", "theoretical_power"]]


def preprocess_wind(raw: pd.DataFrame, cfg: WindPreprocessConfig | None = None) -> tuple[pd.DataFrame, WindPreprocessReport]:
    cfg = cfg or WindPreprocessConfig()
    rep = WindPreprocessReport(raw_rows=len(raw))
    df, exact, conflicting = remove_duplicates(raw, ["timestamp"])
    rep.duplicate_rows_removed, rep.conflicting_keys_removed = exact, conflicting
    df, snapped, dropped = snap_to_grid_step(df, cfg.sync_tolerance)
    df, _, more = remove_duplicates(df, ["timestamp"])
    rep.conflicting_keys_removed += more
    rep.offgrid_snapped, rep.offgrid_dropped = snapped, dropped

    grid = pd.date_range(df["timestamp"].min().floor("D"), df["timestamp"].max(), freq=WIND_STEP, name="timestamp")
    frame = df.set_index("timestamp").reindex(grid)
    rep.grid_start, rep.grid_end, rep.grid_slots = str(grid[0]), str(grid[-1]), len(grid)
    rep.slots_missing = int(frame[TARGET].isna().sum())
    present = df["timestamp"].sort_values()
    rep.longest_gap = str(present.diff().max())

    # Physical bounds. Small negatives are auxiliary consumption, not errors.
    p = frame[TARGET]
    neg_small = (p < 0) & (p >= cfg.negative_clip_kw)
    frame["flag_negative_clipped"] = neg_small
    frame.loc[neg_small, TARGET] = 0.0
    rep.negative_clipped = int(neg_small.sum())
    bounds = {
        TARGET: (0.0, cfg.power_max_kw),
        "wind_speed": cfg.wind_speed_range,
        "wind_dir": (0.0, 360.0),
        "theoretical_power": (0.0, cfg.power_max_kw),
    }
    for col, (lo, hi) in bounds.items():
        bad = frame[col].notna() & ((frame[col] < lo) | (frame[col] > hi))
        frame[f"flag_bound_{col}"] = bad
        frame.loc[bad, col] = np.nan
        rep.bound_violations[col] = int(bad.sum())

    # Missing values: causal forward fill of the weather inputs only; target left NaN.
    filled_any = pd.Series(False, index=frame.index)
    for col in WEATHER_VARS:
        miss = frame[col].isna()
        frame[col] = frame[col].ffill(limit=cfg.weather_ffill_limit)
        filled_any |= miss & frame[col].notna()
    frame["flag_imputed_weather"] = filled_any
    rep.weather_ffilled = int(filled_any.sum())
    frame["flag_imputed_target"] = False          # no physical fill rule for wind (no "night")
    rep.target_missing_final = int(frame[TARGET].isna().sum())

    # Downtime / curtailment and statistical outliers: flagged, values kept.
    down = (frame[TARGET] <= cfg.down_power_kw) & (frame["theoretical_power"] >= cfg.down_theoretical_kw)
    frame["flag_turbine_down"] = down
    rep.turbine_down_flagged = int(down.sum())
    resid = (frame[TARGET] - frame["theoretical_power"]).dropna()
    z = robust_trailing_z(resid, cfg.stat_window, cfg.stat_min_periods)
    frame["stat_z_power_residual"] = z.reindex(frame.index)
    frame["flag_stat_outlier"] = frame["stat_z_power_residual"].abs() > cfg.stat_z_threshold
    rep.stat_outliers_flagged = int(frame["flag_stat_outlier"].sum())

    frame["wind_speed_smooth"] = causal_smooth(frame["wind_speed"], cfg.smooth_window)
    frame.index.name = "timestamp"
    return frame, rep


def snap_to_grid_step(df: pd.DataFrame, tolerance: pd.Timedelta) -> tuple[pd.DataFrame, int, int]:
    """``preprocess.snap_to_grid`` for the 10-min grid."""
    snapped = df["timestamp"].dt.round(WIND_STEP)
    offset = (df["timestamp"] - snapped).abs()
    keep = offset <= tolerance
    out = df.loc[keep].copy()
    out["timestamp"] = snapped[keep]
    return out, int(((offset > pd.Timedelta(0)) & keep).sum()), int((~keep).sum())


def preprocess_wind_file() -> tuple[pd.DataFrame, WindPreprocessReport]:
    return preprocess_wind(load_wind_raw())


# -------------------------------------------------------------------- features

@dataclass(frozen=True)
class WindFeatureConfig:
    horizon: int = 6                          # 10-min steps -> 1 hour ahead
    lags: tuple[int, ...] = (1, 3, 6, 24)     # 10 min, 30 min, 1 h, 4 h before the origin
    roll_windows: tuple[int, ...] = (6, 24)   # 1 h and 4 h ending at the origin
    include_oracle_weather: bool = False


WIND_FEATURES: dict[str, str] = {
    "ac_now": "power at origin t",
    **{f"ac_lag_{k}": f"power at t - {k} steps" for k in WindFeatureConfig.lags},
    **{
        name: desc
        for w in WindFeatureConfig.roll_windows
        for name, desc in (
            (f"ac_roll_mean_{w}", f"mean power over the {w} slots ending at t"),
            (f"ac_roll_std_{w}", f"std of power over the {w} slots ending at t"),
        )
    },
    "ac_same_slot_yesterday": "power at T - 24h",
    "ws_now": "wind speed at t",
    "ws_smooth_now": "causal 3-slot median wind speed at t",
    "ws_roll_mean_6": "mean wind speed over the 6 slots ending at t",
    "ws_roll_std_6": "std of wind speed over the 6 slots ending at t (gustiness)",
    "ws_delta_6": "wind speed change over the last hour, ws(t) - ws(t-6)",
    "ws_same_slot_yesterday": "wind speed at T - 24h",
    "tp_now": "theoretical power-curve value at t (function of ws(t))",
    "wd_sin_now": "sin of wind direction at t",
    "wd_cos_now": "cos of wind direction at t",
    "hour": "hour of T",
    "tod_sin": "sin of time of day of T",
    "tod_cos": "cos of time of day of T",
    "doy_sin": "sin of day of year of T",
    "doy_cos": "cos of day of year of T",
    "month": "month of T",
    "dayofweek": "day of week of T",
}
WIND_ORACLE = {
    "oracle_ws_target": "ORACLE: measured wind speed at T (future; upper bound only)",
    "oracle_tp_target": "ORACLE: theoretical power at T (future; upper bound only)",
}
WIND_SEQ_COLUMNS = (TARGET, "wind_speed", "theoretical_power", "wind_dir")
WIND_CONTEXT_COLS = ("tod_sin", "tod_cos", "doy_sin", "doy_cos", "ac_same_slot_yesterday")


def build_wind_features(
    frame: pd.DataFrame, cfg: WindFeatureConfig | None = None, drop_missing_target: bool = True
) -> SupervisedSet:
    """Origin-indexed rows: features use data at times <= t; target = power at t + horizon."""
    cfg = cfg or WindFeatureConfig()
    diffs = frame.index.to_series().diff().dropna()
    if not (diffs == WIND_STEP).all():
        raise ValueError("wind features need the regular 10-min grid from preprocess_wind")
    h = cfg.horizon
    y, ws, wd = frame[TARGET], frame["wind_speed"], frame["wind_dir"]
    target_time = frame.index + h * WIND_STEP

    X = pd.DataFrame(index=frame.index)
    X["ac_now"] = y
    for k in cfg.lags:
        X[f"ac_lag_{k}"] = y.shift(k)
    for w in cfg.roll_windows:
        roll = y.rolling(w, min_periods=w)
        X[f"ac_roll_mean_{w}"] = roll.mean()
        X[f"ac_roll_std_{w}"] = roll.std()
    X["ac_same_slot_yesterday"] = y.shift(WIND_STEPS_PER_DAY - h)
    X["ws_now"] = ws
    X["ws_smooth_now"] = frame["wind_speed_smooth"]
    X["ws_roll_mean_6"] = ws.rolling(6, min_periods=6).mean()
    X["ws_roll_std_6"] = ws.rolling(6, min_periods=6).std()
    X["ws_delta_6"] = ws - ws.shift(6)
    X["ws_same_slot_yesterday"] = ws.shift(WIND_STEPS_PER_DAY - h)
    X["tp_now"] = frame["theoretical_power"]
    X["wd_sin_now"] = np.sin(np.deg2rad(wd))
    X["wd_cos_now"] = np.cos(np.deg2rad(wd))
    minutes = target_time.hour * 60 + target_time.minute
    doy = target_time.dayofyear
    X["hour"] = target_time.hour
    X["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["month"] = target_time.month
    X["dayofweek"] = target_time.dayofweek
    if cfg.include_oracle_weather:
        X["oracle_ws_target"] = ws.shift(-h)
        X["oracle_tp_target"] = frame["theoretical_power"].shift(-h)

    meta = pd.DataFrame(index=frame.index)
    meta["target_time"] = target_time
    meta["target_imputed"] = False
    meta["target_stat_outlier"] = frame["flag_stat_outlier"].shift(-h, fill_value=False).astype(bool)
    meta["target_down"] = frame["flag_turbine_down"].shift(-h, fill_value=False).astype(bool)
    meta["target_daytime"] = True             # no day/night structure for wind; kept for shared code
    target = y.shift(-h).rename("target")
    keep = target.notna() if drop_missing_target else pd.Series(True, index=frame.index)
    return SupervisedSet(X=X[keep], y=target[keep], meta=meta[keep], config=cfg)


# --------------------------------------------------------------- splits, groups

def _subset(data: SupervisedSet, mask: np.ndarray) -> SupervisedSet:
    return SupervisedSet(X=data.X[mask], y=data.y[mask], meta=data.meta[mask], config=data.config)


def target_months(data: SupervisedSet) -> pd.PeriodIndex:
    return pd.PeriodIndex(data.meta["target_time"].dt.to_period("M").unique()).sort_values()


def rolling_origin_month_folds(
    data: SupervisedSet, test_months: list[str], val_months: int = 1, calib_months: int = 1
) -> Iterator[tuple[pd.Period, dict[str, SupervisedSet]]]:
    """For each test month m: train = months before m-(val+calib), val, calib, test = m."""
    months = target_months(data)
    month_of_row = data.meta["target_time"].dt.to_period("M")
    for label in test_months:
        m = pd.Period(label, "M")
        i = months.get_loc(m)
        cal_start, val_start = i - calib_months, i - calib_months - val_months
        if val_start < 1:
            raise ValueError(f"not enough months before {m} for train/val/calib")
        segs = {"train": months[:val_start], "val": months[val_start:cal_start],
                "calib": months[cal_start:i], "test": months[i:i + 1]}
        yield m, {k: _subset(data, month_of_row.isin(v).to_numpy()) for k, v in segs.items()}


def deployment_month_split(data: SupervisedSet, val_months: int = 1, calib_months: int = 1) -> dict[str, SupervisedSet]:
    """train | val | calib ending at the last month of the record (for the API model)."""
    months = target_months(data)
    month_of_row = data.meta["target_time"].dt.to_period("M")
    cal_start = len(months) - calib_months
    val_start = cal_start - val_months
    segs = {"train": months[:val_start], "val": months[val_start:cal_start], "calib": months[cal_start:]}
    return {k: _subset(data, month_of_row.isin(v).to_numpy()) for k, v in segs.items()}


REGIME_EDGES = (0.10 * RATED_KW, 0.90 * RATED_KW)   # fixed in advance, not tuned
REGIMES = ("low", "mid", "high")


def power_regime(point_kw: np.ndarray) -> np.ndarray:
    """Mondrian group from the point forecast (known at the origin): low / mid / high output."""
    p = np.asarray(point_kw, dtype=float)
    out = np.full(p.shape, "mid", dtype=object)
    out[p < REGIME_EDGES[0]] = "low"
    out[p >= REGIME_EDGES[1]] = "high"
    return out
