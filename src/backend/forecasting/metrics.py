"""Point-forecast error metrics (A3/A5)."""

from __future__ import annotations

import numpy as np


def _arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch {t.shape} vs {p.shape}")
    if t.size == 0:
        raise ValueError("no rows to score")
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError("metrics need finite values")
    return t, p


def mae(y_true, y_pred) -> float:
    t, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def rmse(y_true, y_pred) -> float:
    t, p = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def r2(y_true, y_pred) -> float:
    t, p = _arrays(y_true, y_pred)
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot == 0:
        raise ValueError("R2 undefined for a constant target")
    return 1.0 - float(np.sum((t - p) ** 2)) / ss_tot


def mape(y_true, y_pred, min_true: float) -> float:
    """MAPE in percent over rows with y_true >= min_true (undefined near zero)."""
    t, p = _arrays(y_true, y_pred)
    keep = t >= min_true
    if not keep.any():
        raise ValueError("no rows above min_true for MAPE")
    return float(np.mean(np.abs((t[keep] - p[keep]) / t[keep])) * 100.0)
