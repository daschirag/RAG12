"""A2 - Feature engineering for 1-hour-ahead plant AC power forecasting.

Resolution: native 15-min grid from A1 (no resampling). Horizon: H = 4 steps
(1 hour), identical for every model in A3.

Each supervised row is indexed by its forecast origin t (the last observed
slot). The target is AC power at T = t + H. Every feature uses data at times
<= t, with two kinds of exception that are still known at t:

- calendar features of T (deterministic),
- the irradiation envelope at T's time-of-day slot, built only from days
  strictly before T's date (all such times are <= T - 24h < t).

Actual weather at T is future information. It is only added when
``FeatureConfig.include_oracle_weather`` is set, under an ``oracle_`` prefix,
as a labelled upper bound that is not deployable.

Plan lags 1/3/6/24 are in 15-min steps before the origin (15 min, 45 min,
1.5 h, 6 h). ``ac_same_slot_yesterday`` adds the daily lag y(T - 24h).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .preprocess import STEP, TARGET

STEPS_PER_DAY = int(pd.Timedelta("1D") / STEP)  # 96


@dataclass(frozen=True)
class FeatureConfig:
    horizon: int = 4                         # steps of 15 min -> 1 hour ahead
    lags: tuple[int, ...] = (1, 3, 6, 24)    # steps before the origin
    roll_windows: tuple[int, ...] = (4, 16)  # 1 h and 4 h trailing windows
    daytime_envelope: float = 0.05           # envelope above this = daytime slot
    include_oracle_weather: bool = False


# (name, description) in output order; oracle feature appended when enabled.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "ac_now": "AC power at origin t",
    **{f"ac_lag_{k}": f"AC power at t - {k} steps" for k in FeatureConfig.lags},
    **{
        name: desc
        for w in FeatureConfig.roll_windows
        for name, desc in (
            (f"ac_roll_mean_{w}", f"mean AC power over the {w} slots ending at t"),
            (f"ac_roll_std_{w}", f"std of AC power over the {w} slots ending at t"),
        )
    },
    "ac_same_slot_yesterday": "AC power at T - 24h (same time of day, previous day)",
    "irr_now": "irradiation at t",
    "irr_smooth_now": "causal 3-slot median irradiation at t",
    "irr_roll_mean_4": "mean irradiation over the 4 slots ending at t",
    "irr_delta_4": "irradiation change over the last hour, irr(t) - irr(t-4)",
    "irr_same_slot_yesterday": "irradiation at T - 24h",
    "irr_envelope_target": "clear-sky proxy for T's slot: max irradiation at that slot on previous days",
    "irr_clearness_now": "irr(t) / envelope at t's slot (cloudiness proxy)",
    "ambient_now": "ambient temperature at t",
    "module_now": "module temperature at t",
    "module_minus_ambient_now": "module - ambient temperature at t (irradiance heating proxy)",
    "hour": "hour of T",
    "tod_sin": "sin of time of day of T",
    "tod_cos": "cos of time of day of T",
    "day": "day of month of T",
    "month": "month of T",
    "dayofweek": "day of week of T",
}
ORACLE_DESCRIPTIONS = {
    "oracle_irr_target": "ORACLE: measured irradiation at T (future; upper bound only, not deployable)",
}


@dataclass
class SupervisedSet:
    X: pd.DataFrame          # index = forecast origin t
    y: pd.Series             # AC power at T = t + horizon
    meta: pd.DataFrame       # target_time, target_imputed, target_daytime, target_stat_outlier
    config: FeatureConfig

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


def _require_regular_grid(frame: pd.DataFrame) -> None:
    diffs = frame.index.to_series().diff().dropna()
    if not (diffs == STEP).all():
        raise ValueError("Feature engineering needs the regular 15-min grid produced by A1")


def irradiation_envelope(irr: pd.Series) -> pd.Series:
    """Max irradiation at the same time-of-day slot over all *previous* days."""
    slot = irr.index.hour * 60 + irr.index.minute
    return irr.groupby(slot).transform(lambda s: s.shift(1).expanding().max())


def build_features(
    frame: pd.DataFrame, cfg: FeatureConfig | None = None, drop_missing_target: bool = True
) -> SupervisedSet:
    """Turn an A1 frame into (X, y, meta) for horizon ``cfg.horizon``.

    ``drop_missing_target=False`` keeps every origin, including the last hour of
    the record whose target lies beyond the data (y is NaN there); the
    forecasting service needs those rows to forecast forward.
    """
    cfg = cfg or FeatureConfig()
    _require_regular_grid(frame)
    h = cfg.horizon
    if not 1 <= h < STEPS_PER_DAY:
        raise ValueError("horizon must be between 1 and 95 steps")

    y = frame[TARGET]
    irr = frame["irradiation"]
    # Extend the grid by h slots so the envelope at T exists even when T is past
    # the end of the record. The envelope only looks at previous days, so values
    # inside the record are unchanged by the extension.
    ext_index = pd.date_range(frame.index[0], frame.index[-1] + h * STEP, freq=STEP)
    env_ext = irradiation_envelope(irr.reindex(ext_index))
    env = env_ext.reindex(frame.index)
    target_time = frame.index + h * STEP

    X = pd.DataFrame(index=frame.index)
    X["ac_now"] = y
    for k in cfg.lags:
        X[f"ac_lag_{k}"] = y.shift(k)
    for w in cfg.roll_windows:
        roll = y.rolling(w, min_periods=w)
        X[f"ac_roll_mean_{w}"] = roll.mean()
        X[f"ac_roll_std_{w}"] = roll.std()
    X["ac_same_slot_yesterday"] = y.shift(STEPS_PER_DAY - h)

    X["irr_now"] = irr
    X["irr_smooth_now"] = frame["irradiation_smooth"]
    X["irr_roll_mean_4"] = irr.rolling(4, min_periods=4).mean()
    X["irr_delta_4"] = irr - irr.shift(4)
    X["irr_same_slot_yesterday"] = irr.shift(STEPS_PER_DAY - h)
    # env at T only uses days before T's date, i.e. times <= T - 24h < t.
    X["irr_envelope_target"] = env_ext.shift(-h).reindex(frame.index)
    X["irr_clearness_now"] = (irr / env.where(env > cfg.daytime_envelope)).clip(0, 1.5)
    X["ambient_now"] = frame["ambient_temp"]
    X["module_now"] = frame["module_temp"]
    X["module_minus_ambient_now"] = frame["module_temp"] - frame["ambient_temp"]

    minutes = target_time.hour * 60 + target_time.minute
    X["hour"] = target_time.hour
    X["tod_sin"] = np.sin(2 * np.pi * minutes / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * minutes / 1440)
    X["day"] = target_time.day
    X["month"] = target_time.month
    X["dayofweek"] = target_time.dayofweek

    if cfg.include_oracle_weather:
        X["oracle_irr_target"] = irr.shift(-h)

    meta = pd.DataFrame(index=frame.index)
    meta["target_time"] = target_time
    meta["target_imputed"] = frame["flag_imputed_target"].shift(-h, fill_value=False).astype(bool)
    meta["target_stat_outlier"] = frame["flag_stat_outlier"].shift(-h, fill_value=False).astype(bool)
    meta["target_daytime"] = X["irr_envelope_target"] > cfg.daytime_envelope

    target = y.shift(-h).rename("target")
    keep = target.notna() if drop_missing_target else pd.Series(True, index=frame.index)
    return SupervisedSet(X=X[keep], y=target[keep], meta=meta[keep], config=cfg)


DEFAULT_SPLIT_DAYS: tuple[tuple[str, int], ...] = (("train", 21), ("val", 4), ("calib", 4))


def split_by_target_day(
    data: SupervisedSet,
    segments: tuple[tuple[str, int], ...] = DEFAULT_SPLIT_DAYS,
    rest: str = "test",
) -> dict[str, SupervisedSet]:
    """Chronological split on whole days of the *target* time.

    Assigning rows by target time means no training row has a target inside a
    later segment. Later segments may use earlier observations as features,
    which is what a real forecaster at time t would have.
    """
    target_day = data.meta["target_time"].dt.normalize()
    days = pd.Index(target_day.unique()).sort_values()
    total = sum(n for _, n in segments)
    if total >= len(days):
        raise ValueError(f"Segments use {total} days but only {len(days)} are available")
    out: dict[str, SupervisedSet] = {}
    start = 0
    bounds = [*segments, (rest, len(days) - total)]
    for name, n in bounds:
        chosen = days[start:start + n]
        mask = target_day.isin(chosen).to_numpy()
        out[name] = SupervisedSet(
            X=data.X[mask], y=data.y[mask], meta=data.meta[mask], config=data.config
        )
        start += n
    return out
