"""C3 - Telemetry anomaly / spoofing detector for the solar plants and the wind turbine.

Works on what the sensors report, before A1 cleaning (A1 would already replace out-of-bound values). Five
checks, all fitted on clean **training days only** (chronological, no test data):

1. ``bounds``       physical ranges (the A1 limits, fixed in advance).
2. ``rate``         |change between consecutive slots| above the training 99.9th percentile (per variable).
3. ``consistency``  cross-sensor residuals above the training 99.9th percentile:
                    solar - AC power vs irradiation (linear, through the origin) and module temperature vs
                    ambient temperature + irradiation (least squares); wind - reported theoretical power vs the
                    power curve learned from wind speed (binned median).
4. ``statistical``  Mahalanobis distance of the slot's standardised readings and residuals from the training
                    distribution, above the training 99.9th percentile.
5. ``replay``       a whole day that (nearly) repeats an earlier day of the record: minimum distance to any
                    earlier day below half the smallest distance between two different training days.

Decisions are per day (the attack unit in C2): a slot check flags a day when its number of flagged slots
exceeds the largest count seen on any training day; ``replay`` is day-level already; ``any`` = at least one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..forecasting.preprocess import PreprocessConfig

_P = PreprocessConfig()
SOLAR_VARS = ("ac_kw", "irradiation", "ambient_temp", "module_temp")
WIND_VARS = ("ac_power_kw", "wind_speed", "wind_dir", "theoretical_power")
WIND_CAPACITY = 3600.0 * 1.05
BOUNDS = {
    "solar": {"ac_kw": (0.0, _P.inverter_ac_max_kw * _P.n_inverters), "irradiation": _P.irradiation_range,
              "ambient_temp": _P.ambient_range, "module_temp": _P.module_range},
    "wind": {"ac_power_kw": (-10.0, WIND_CAPACITY), "wind_speed": (0.0, 40.0), "wind_dir": (0.0, 360.0),
             "theoretical_power": (0.0, WIND_CAPACITY)},
}
STEP = {"solar": "15min", "wind": "10min"}
SLOT_CHECKS = ("bounds", "rate", "consistency", "statistical")
CHECKS = SLOT_CHECKS + ("replay",)
QUANTILE = 0.999
REPLAY_FACTOR = 0.5
DAYLIGHT_IRR = 0.1


# ------------------------------------------------------------------ sensor frames

def _grid(frame: pd.DataFrame, step: str) -> pd.DataFrame:
    idx = pd.date_range(frame.index.min().floor("D"), frame.index.max().ceil("D") - pd.Timedelta(step),
                        freq=step, name="timestamp")
    return frame.reindex(idx)


def solar_sensor_frame(gen: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Plant-level readings on the 15-min grid, uncleaned: AC = mean over reporting inverters x 22 (as A1)."""
    g = gen.assign(timestamp=gen["timestamp"].dt.round("15min")).groupby("timestamp")["AC_POWER"].mean()
    w = weather.assign(timestamp=weather["timestamp"].dt.round("15min")).groupby("timestamp")[
        ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]].mean()
    w.columns = ["irradiation", "ambient_temp", "module_temp"]
    frame = w.join((g * _P.n_inverters).rename("ac_kw"), how="outer")
    return _grid(frame[list(SOLAR_VARS)], STEP["solar"])


def wind_sensor_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Turbine readings on the 10-min grid, uncleaned (columns as returned by ``load_wind_raw``)."""
    f = raw.assign(timestamp=raw["timestamp"].dt.round("10min")).groupby("timestamp")[
        ["ac_power_kw", "wind_speed", "wind_dir", "theoretical_power"]].mean()
    return _grid(f[list(WIND_VARS)], STEP["wind"])


# ------------------------------------------------------------------ detector

@dataclass
class TelemetryDetector:
    kind: str                                   # "solar" | "wind"
    quantile: float = QUANTILE
    replay_factor: float = REPLAY_FACTOR
    params: dict = field(default_factory=dict)

    @property
    def variables(self) -> tuple[str, ...]:
        return SOLAR_VARS if self.kind == "solar" else WIND_VARS

    # -- residual models -------------------------------------------------------
    def _residuals(self, f: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        if self.kind == "solar":
            return pd.DataFrame({
                "res_ac_irr": f["ac_kw"] - p["k_ac"] * f["irradiation"],
                "res_module": f["module_temp"] - (p["mod_coef"][0] + p["mod_coef"][1] * f["ambient_temp"]
                                                  + p["mod_coef"][2] * f["irradiation"]),
            }, index=f.index)
        curve = np.interp(f["wind_speed"].to_numpy(dtype=float), p["curve_x"], p["curve_y"])
        return pd.DataFrame({"res_curve": f["theoretical_power"] - curve}, index=f.index)

    def _fit_models(self, f: pd.DataFrame) -> None:
        if self.kind == "solar":
            day = f[(f["irradiation"] > DAYLIGHT_IRR)].dropna(subset=["ac_kw", "irradiation"])
            self.params["k_ac"] = float((day["ac_kw"] * day["irradiation"]).sum() / (day["irradiation"] ** 2).sum())
            m = f.dropna(subset=["module_temp", "ambient_temp", "irradiation"])
            X = np.column_stack([np.ones(len(m)), m["ambient_temp"], m["irradiation"]])
            self.params["mod_coef"] = np.linalg.lstsq(X, m["module_temp"].to_numpy(), rcond=None)[0].tolist()
        else:
            m = f.dropna(subset=["wind_speed", "theoretical_power"])
            bins = np.floor(m["wind_speed"] / 0.5) * 0.5 + 0.25
            curve = m.groupby(bins)["theoretical_power"].median()
            self.params["curve_x"], self.params["curve_y"] = curve.index.to_numpy(), curve.to_numpy()

    # -- slot checks -----------------------------------------------------------
    def _features(self, f: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([f[list(self.variables)], self._residuals(f)], axis=1)

    def _mahalanobis(self, feats: pd.DataFrame) -> pd.Series:
        z = (feats - self.params["mu"]) / self.params["sd"]
        ok = z.notna().all(axis=1)
        d = np.full(len(z), np.nan)
        zz = z[ok].to_numpy()
        d[ok.to_numpy()] = np.sqrt(np.einsum("ij,jk,ik->i", zz, self.params["prec"], zz))
        return pd.Series(d, index=feats.index)

    def slot_flags(self, f: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = pd.DataFrame(index=f.index)
        bad = pd.Series(False, index=f.index)
        for col, (lo, hi) in BOUNDS[self.kind].items():
            bad |= f[col].notna() & ((f[col] < lo) | (f[col] > hi))
        out["bounds"] = bad
        rate = pd.Series(False, index=f.index)
        for col in self.variables:
            rate |= f[col].diff().abs() > p["rate_thr"][col]
        out["rate"] = rate
        res = self._residuals(f)
        cons = pd.Series(False, index=f.index)
        for col in res:
            cons |= res[col].abs() > p["res_thr"][col]
        out["consistency"] = cons
        out["statistical"] = self._mahalanobis(self._features(f)) > p["maha_thr"]
        return out.fillna(False).astype(bool)

    # -- replay ----------------------------------------------------------------
    def _profiles(self, f: pd.DataFrame) -> dict[pd.Timestamp, np.ndarray]:
        z = (f[list(self.variables)] - self.params["mu"][list(self.variables)]) / \
            self.params["sd"][list(self.variables)]
        return {day: g.to_numpy(dtype=float) for day, g in z.groupby(z.index.normalize())}

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return np.inf
        diff = np.abs(a - b)
        ok = ~np.isnan(diff)
        return float(diff[ok].mean()) if ok.mean() >= 0.5 else np.inf

    # -- fit / report ------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "TelemetryDetector":
        """Fit every threshold on clean training days only."""
        self._fit_models(train)
        q = self.quantile
        self.params["rate_thr"] = {c: float(train[c].diff().abs().quantile(q)) for c in self.variables}
        res = self._residuals(train)
        self.params["res_thr"] = {c: float(res[c].abs().quantile(q)) for c in res}
        feats = self._features(train).dropna()
        self.params["mu"], self.params["sd"] = feats.mean(), feats.std().replace(0, 1.0)
        z = ((feats - self.params["mu"]) / self.params["sd"]).to_numpy()
        self.params["prec"] = np.linalg.pinv(np.cov(z, rowvar=False))
        self.params["maha_thr"] = float(self._mahalanobis(self._features(train)).quantile(q))
        counts = self.slot_flags(train).groupby(train.index.normalize()).sum()
        self.params["day_thr"] = {c: int(counts[c].max()) for c in SLOT_CHECKS}
        prof = list(self._profiles(train).values())
        pair = [self._distance(prof[i], prof[j]) for i in range(len(prof)) for j in range(i)]
        self.params["replay_eps"] = self.replay_factor * float(np.min([d for d in pair if np.isfinite(d)]))
        return self

    def day_report(self, frame: pd.DataFrame, days=None) -> pd.DataFrame:
        """Per-day flags for ``days`` (default: all). Replay compares each day with every earlier day of ``frame``."""
        flags = self.slot_flags(frame)
        counts = flags.groupby(frame.index.normalize()).sum()
        profiles = self._profiles(frame)
        ordered = sorted(profiles)
        days = ordered if days is None else [pd.Timestamp(d).normalize() for d in days]
        rows = []
        for day in days:
            earlier = [profiles[d] for d in ordered if d < day]
            dist = min((self._distance(profiles[day], e) for e in earlier), default=np.inf)
            row = {"day": day, "replay_distance": dist, "replay": bool(dist < self.params["replay_eps"])}
            for c in SLOT_CHECKS:
                row[f"{c}_slots"] = int(counts.loc[day, c]) if day in counts.index else 0
                row[c] = row[f"{c}_slots"] > self.params["day_thr"][c]
            row["any"] = any(row[c] for c in CHECKS)
            rows.append(row)
        return pd.DataFrame(rows)
