"""A1 - Solar preprocessing pipeline (plant level).

Raw inverter + weather CSVs -> one regular 15-min frame per plant:

    load -> duplicates -> time sync -> plant aggregation -> physical bounds
         -> missing values -> statistical outliers -> noise handling

Normalization is a separate, train-only step (``TrainOnlyScaler``) applied after
``chronological_split``, so no statistic ever comes from validation/test rows.

Every step is causal: the value produced for time t depends only on rows at or
before t. ``tests/test_forecasting_preprocess.py`` checks this by running the
pipeline on a truncated copy of the data and comparing with the full run.

Dataset facts this module relies on (verified on the files, see CLAUDE.md):
- 15-min sampling, 2020-05-15 00:00 to 2020-06-17 23:45, 22 inverters per plant.
- Plant_1 generation timestamps are ``dd-mm-yyyy HH:MM``; the other files are ISO.
- Plant_1 DC_POWER is ~10x AC_POWER (Plant_2 ~1.02x): a scale inconsistency.
  DC is therefore not used; the target is AC power.
- There is no humidity column.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOLAR_DIR = PROJECT_ROOT / "dataset" / "solar"

STEP = pd.Timedelta("15min")
GEN_TIME_FORMATS = {1: "%d-%m-%Y %H:%M", 2: "%Y-%m-%d %H:%M:%S"}
WEATHER_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

GEN_COLUMNS = ("DATE_TIME", "PLANT_ID", "SOURCE_KEY", "DC_POWER", "AC_POWER")
WEATHER_COLUMNS = ("DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION")
WEATHER_RENAME = {
    "IRRADIATION": "irradiation",
    "AMBIENT_TEMPERATURE": "ambient_temp",
    "MODULE_TEMPERATURE": "module_temp",
}
WEATHER_VARS = ("irradiation", "ambient_temp", "module_temp")
TARGET = "ac_power_kw"


@dataclass(frozen=True)
class PreprocessConfig:
    """Tunable thresholds. Defaults are justified in docs/forecasting.md."""

    # Snap timestamps within this distance of a 15-min slot; drop the rest.
    sync_tolerance: pd.Timedelta = pd.Timedelta("2min")
    # Inverters in the plant (both plants have 22 distinct SOURCE_KEYs).
    n_inverters: int = 22
    # Below this share of reporting inverters the plant total is unreliable.
    min_inverter_coverage: float = 0.5
    # Physical bounds. Inverter nameplate is not in the dataset; 1500 kW sits
    # above the largest observed per-inverter AC value (1410.95 kW).
    inverter_ac_max_kw: float = 1500.0
    irradiation_range: tuple[float, float] = (0.0, 1.5)
    ambient_range: tuple[float, float] = (-10.0, 60.0)
    module_range: tuple[float, float] = (-10.0, 90.0)
    # AC above this while irradiation == 0 is physically inconsistent.
    night_ac_tolerance_kw: float = 1.0
    # Causal forward fill for weather gaps (steps of 15 min).
    weather_ffill_limit: int = 2
    # Statistical outliers on AC/irradiation (daytime only), trailing window.
    daytime_irradiation: float = 0.1
    stat_window: str = "7D"
    stat_min_periods: int = 20
    stat_z_threshold: float = 5.0
    # Causal rolling median for sensor noise (steps).
    smooth_window: int = 3


@dataclass
class PreprocessReport:
    """Counts for every cleaning decision, saved to results/ as JSON."""

    plant: int | None = None
    raw_generation_rows: int = 0
    raw_weather_rows: int = 0
    duplicate_rows_removed: dict[str, int] = field(default_factory=dict)
    conflicting_keys_removed: dict[str, int] = field(default_factory=dict)
    offgrid_snapped: dict[str, int] = field(default_factory=dict)
    offgrid_dropped: dict[str, int] = field(default_factory=dict)
    grid_start: str = ""
    grid_end: str = ""
    grid_slots: int = 0
    slots_without_generation: int = 0
    slots_without_weather: int = 0
    slots_partial_coverage: int = 0
    slots_low_coverage_target_nan: int = 0
    dc_ac_ratio_median: float | None = None
    dc_scale_suspect: bool = False
    bound_violations: dict[str, int] = field(default_factory=dict)
    night_inconsistent: int = 0
    target_night_filled: int = 0
    weather_ffilled: dict[str, int] = field(default_factory=dict)
    weather_still_missing: dict[str, int] = field(default_factory=dict)
    target_missing_final: int = 0
    stat_outliers_flagged: int = 0
    noise_mean_abs_diff: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreprocessResult:
    frame: pd.DataFrame
    report: PreprocessReport


# ----------------------------------------------------------------------------- load

def load_plant_raw(plant: int, solar_dir: Path = SOLAR_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one plant's generation and weather CSVs with explicit timestamp formats."""
    if plant not in GEN_TIME_FORMATS:
        raise ValueError(f"Unknown plant {plant!r}; expected 1 or 2")
    gen = pd.read_csv(solar_dir / f"Plant_{plant}_Generation_Data.csv", encoding="utf-8")
    weather = pd.read_csv(solar_dir / f"Plant_{plant}_Weather_Sensor_Data.csv", encoding="utf-8")
    _require_columns(gen, GEN_COLUMNS, "generation")
    _require_columns(weather, WEATHER_COLUMNS, "weather")
    gen["timestamp"] = pd.to_datetime(gen["DATE_TIME"], format=GEN_TIME_FORMATS[plant])
    weather["timestamp"] = pd.to_datetime(weather["DATE_TIME"], format=WEATHER_TIME_FORMAT)
    return gen, weather


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} data is missing columns: {missing}")


# ----------------------------------------------------------------------- pipeline

def remove_duplicates(df: pd.DataFrame, key: list[str]) -> tuple[pd.DataFrame, int, int]:
    """Drop exact duplicate rows, then rows whose key repeats with different values.

    Conflicting duplicates are all dropped (no way to tell which is right) and
    counted separately. Returns (frame, exact_removed, conflicting_removed).
    """
    before = len(df)
    df = df.drop_duplicates()
    exact = before - len(df)
    conflict_mask = df.duplicated(subset=key, keep=False)
    conflicting = int(conflict_mask.sum())
    return df.loc[~conflict_mask].copy(), exact, conflicting


def snap_to_grid(
    df: pd.DataFrame, tolerance: pd.Timedelta, column: str = "timestamp"
) -> tuple[pd.DataFrame, int, int]:
    """Snap timestamps to the nearest 15-min slot within ``tolerance``; drop others."""
    snapped = df[column].dt.round(STEP)
    offset = (df[column] - snapped).abs()
    offgrid = offset > pd.Timedelta(0)
    keep = offset <= tolerance
    out = df.loc[keep].copy()
    out[column] = snapped[keep]
    return out, int((offgrid & keep).sum()), int((~keep).sum())


def full_grid(timestamps: pd.Series) -> pd.DatetimeIndex:
    """Regular 15-min index from the first day's 00:00 to the last observed slot."""
    start = timestamps.min().floor("D")
    end = timestamps.max()
    return pd.date_range(start, end, freq=STEP, name="timestamp")


def aggregate_plant(gen: pd.DataFrame, grid: pd.DatetimeIndex, cfg: PreprocessConfig) -> pd.DataFrame:
    """Plant AC per slot = mean over reporting inverters x n_inverters.

    Scaling the mean (rather than summing) keeps the total comparable when
    some inverter rows are absent; ``inverter_coverage`` records how many
    reported. Below ``min_inverter_coverage`` the estimate is set to NaN.
    """
    grouped = gen.groupby("timestamp").agg(
        ac_mean=("AC_POWER", "mean"),
        n_reporting=("SOURCE_KEY", "nunique"),
    )
    agg = grouped.reindex(grid)
    agg["n_reporting"] = agg["n_reporting"].fillna(0).astype(int)
    agg["inverter_coverage"] = agg["n_reporting"] / cfg.n_inverters
    agg[TARGET] = agg["ac_mean"] * cfg.n_inverters
    low = (agg["n_reporting"] > 0) & (agg["inverter_coverage"] < cfg.min_inverter_coverage)
    agg.loc[low, TARGET] = np.nan
    agg["flag_low_coverage"] = low
    return agg.drop(columns=["ac_mean"])


def _dc_ac_ratio(gen: pd.DataFrame) -> float | None:
    day = gen[gen["AC_POWER"] > 0]
    if day.empty:
        return None
    return float((day["DC_POWER"] / day["AC_POWER"]).median())


def apply_physical_bounds(frame: pd.DataFrame, cfg: PreprocessConfig) -> dict[str, int]:
    """Set physically impossible values to NaN (in place) and flag them."""
    bounds = {
        TARGET: (0.0, cfg.inverter_ac_max_kw * cfg.n_inverters),
        "irradiation": cfg.irradiation_range,
        "ambient_temp": cfg.ambient_range,
        "module_temp": cfg.module_range,
    }
    counts: dict[str, int] = {}
    for col, (lo, hi) in bounds.items():
        bad = frame[col].notna() & ((frame[col] < lo) | (frame[col] > hi))
        frame[f"flag_bound_{col}"] = bad
        frame.loc[bad, col] = np.nan
        counts[col] = int(bad.sum())
    return counts


def robust_trailing_z(values: pd.Series, window: str, min_periods: int) -> pd.Series:
    """Robust z-score of each value against the trailing window that excludes it."""
    roll = values.rolling(window, min_periods=min_periods, closed="left")
    median = roll.median()
    mad = roll.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    scale = 1.4826 * mad
    return (values - median) / scale.replace(0, np.nan)


def causal_smooth(series: pd.Series, window: int) -> pd.Series:
    """Trailing rolling median (uses t and the window-1 previous slots only)."""
    return series.rolling(window, min_periods=1).median()


def preprocess(
    gen_raw: pd.DataFrame,
    weather_raw: pd.DataFrame,
    cfg: PreprocessConfig | None = None,
    plant: int | None = None,
) -> PreprocessResult:
    """Run the full A1 pipeline on raw frames that carry a parsed ``timestamp``."""
    cfg = cfg or PreprocessConfig()
    rep = PreprocessReport(
        plant=plant, raw_generation_rows=len(gen_raw), raw_weather_rows=len(weather_raw)
    )

    gen = gen_raw[["timestamp", "SOURCE_KEY", "DC_POWER", "AC_POWER"]].copy()
    weather = weather_raw[["timestamp", *WEATHER_RENAME]].rename(columns=WEATHER_RENAME)

    # 1. Duplicates.
    gen, g_exact, g_conf = remove_duplicates(gen, ["timestamp", "SOURCE_KEY"])
    weather, w_exact, w_conf = remove_duplicates(weather, ["timestamp"])
    rep.duplicate_rows_removed = {"generation": g_exact, "weather": w_exact}
    rep.conflicting_keys_removed = {"generation": g_conf, "weather": w_conf}

    # 2. Time synchronisation onto one regular 15-min grid.
    gen, g_snap, g_drop = snap_to_grid(gen, cfg.sync_tolerance)
    weather, w_snap, w_drop = snap_to_grid(weather, cfg.sync_tolerance)
    # Snapping can create new key collisions; resolve them the same way.
    gen, _, g_conf2 = remove_duplicates(gen, ["timestamp", "SOURCE_KEY"])
    weather, _, w_conf2 = remove_duplicates(weather, ["timestamp"])
    rep.conflicting_keys_removed["generation"] += g_conf2
    rep.conflicting_keys_removed["weather"] += w_conf2
    rep.offgrid_snapped = {"generation": g_snap, "weather": w_snap}
    rep.offgrid_dropped = {"generation": g_drop, "weather": w_drop}

    grid = full_grid(pd.concat([gen["timestamp"], weather["timestamp"]]))
    rep.grid_start, rep.grid_end, rep.grid_slots = str(grid[0]), str(grid[-1]), len(grid)

    # 3. Plant-level aggregation (+ DC scale check; DC is not carried forward).
    ratio = _dc_ac_ratio(gen)
    rep.dc_ac_ratio_median = None if ratio is None else round(ratio, 4)
    rep.dc_scale_suspect = ratio is not None and not (0.9 <= ratio <= 1.5)

    frame = aggregate_plant(gen, grid, cfg)
    frame = frame.join(weather.set_index("timestamp").reindex(grid))
    rep.slots_without_generation = int((frame["n_reporting"] == 0).sum())
    rep.slots_without_weather = int(frame["irradiation"].isna().sum())
    rep.slots_partial_coverage = int(
        ((frame["n_reporting"] > 0) & (frame["n_reporting"] < cfg.n_inverters)).sum()
    )
    rep.slots_low_coverage_target_nan = int(frame["flag_low_coverage"].sum())

    # 4. Physical bounds and day/night consistency.
    rep.bound_violations = apply_physical_bounds(frame, cfg)
    night_bad = (frame["irradiation"] == 0) & (frame[TARGET] > cfg.night_ac_tolerance_kw)
    frame["flag_night_inconsistent"] = night_bad
    frame.loc[night_bad, TARGET] = np.nan
    rep.night_inconsistent = int(night_bad.sum())

    # 5. Missing values (causal only).
    for col in WEATHER_VARS:
        missing = frame[col].isna()
        frame[col] = frame[col].ffill(limit=cfg.weather_ffill_limit)
        filled = missing & frame[col].notna()
        frame[f"flag_imputed_{col}"] = filled
        rep.weather_ffilled[col] = int(filled.sum())
        rep.weather_still_missing[col] = int(frame[col].isna().sum())
    # No sun -> no output: a missing target at observed zero irradiation is 0.
    # Only slots with no generation rows are filled; values rejected above
    # (bounds, night inconsistency, low coverage) stay NaN.
    night_fill = (
        frame[TARGET].isna()
        & (frame["n_reporting"] == 0)
        & (frame["irradiation"] == 0)
        & ~frame["flag_imputed_irradiation"]
    )
    frame.loc[night_fill, TARGET] = 0.0
    frame["flag_imputed_target"] = night_fill
    rep.target_night_filled = int(night_fill.sum())
    rep.target_missing_final = int(frame[TARGET].isna().sum())

    # 6. Statistical outliers: flagged, value kept (may be real events).
    daytime = frame["irradiation"] > cfg.daytime_irradiation
    ratio_series = (frame[TARGET] / frame["irradiation"]).where(daytime).dropna()
    z = robust_trailing_z(ratio_series, cfg.stat_window, cfg.stat_min_periods)
    frame["stat_z_ac_per_irradiation"] = z.reindex(frame.index)
    frame["flag_stat_outlier"] = frame["stat_z_ac_per_irradiation"].abs() > cfg.stat_z_threshold
    rep.stat_outliers_flagged = int(frame["flag_stat_outlier"].sum())

    # 7. Noise: causal smoothed copies of sensor inputs; raw kept, target untouched.
    for col in WEATHER_VARS:
        smooth = causal_smooth(frame[col], cfg.smooth_window)
        frame[f"{col}_smooth"] = smooth
        rep.noise_mean_abs_diff[col] = {
            "raw": round(float(frame[col].diff().abs().mean()), 6),
            "smooth": round(float(smooth.diff().abs().mean()), 6),
        }

    frame.index.name = "timestamp"
    return PreprocessResult(frame=frame, report=rep)


def preprocess_plant(plant: int, cfg: PreprocessConfig | None = None) -> PreprocessResult:
    gen, weather = load_plant_raw(plant)
    return preprocess(gen, weather, cfg, plant=plant)


# ---------------------------------------------------------- split + normalization

def chronological_split(
    frame: pd.DataFrame, train_days: int, val_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split on whole calendar days, in time order: train | val | test (rest)."""
    days = pd.Index(frame.index.normalize().unique()).sort_values()
    if train_days < 1 or val_days < 0 or train_days + val_days >= len(days):
        raise ValueError(
            f"Need train_days>=1 and train_days+val_days < {len(days)} available days"
        )
    val_start = days[train_days]
    test_start = days[train_days + val_days]
    train = frame.loc[frame.index < val_start]
    val = frame.loc[(frame.index >= val_start) & (frame.index < test_start)]
    test = frame.loc[frame.index >= test_start]
    return train, val, test


class TrainOnlyScaler:
    """Column scaler whose statistics come from the training frame only."""

    def __init__(self, method: Literal["standard", "minmax"] = "standard") -> None:
        if method not in ("standard", "minmax"):
            raise ValueError("method must be 'standard' or 'minmax'")
        self.method = method
        self.params_: dict[str, tuple[float, float]] | None = None

    def fit(self, train: pd.DataFrame, columns: list[str]) -> "TrainOnlyScaler":
        params: dict[str, tuple[float, float]] = {}
        for col in columns:
            values = train[col].dropna()
            if values.empty:
                raise ValueError(f"Column {col!r} has no training values")
            if self.method == "standard":
                center, scale = float(values.mean()), float(values.std(ddof=0))
            else:
                center, scale = float(values.min()), float(values.max() - values.min())
            params[col] = (center, scale if scale > 0 else 1.0)
        self.params_ = params
        return self

    def _check(self) -> dict[str, tuple[float, float]]:
        if self.params_ is None:
            raise RuntimeError("TrainOnlyScaler.transform called before fit")
        return self.params_

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col, (center, scale) in self._check().items():
            out[col] = (out[col] - center) / scale
        return out

    def inverse_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col, (center, scale) in self._check().items():
            out[col] = out[col] * scale + center
        return out

    def to_dict(self) -> dict:
        return {"method": self.method, "params": self._check()}
