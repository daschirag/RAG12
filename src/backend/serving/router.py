"""Deterministic query intent routing and numeric answer helpers for /ask."""

from __future__ import annotations

import re
from typing import Any

from ..config.domain_manifest import DomainManifest
from ..retrieval.metrics import NUMERIC_METRIC_FIELDS

# Superlatives that imply descending (highest / peak) or ascending (lowest) order.
_DESC_PATTERNS = (
    r"\bhighest\b",
    r"\bmaximum\b",
    r"\bmaximal\b",
    r"\bmax\b",
    r"\bmost\b",
    r"\bpeak\b",
    r"\btop\b",
    r"\bgreatest\b",
)
_ASC_PATTERNS = (
    r"\blowest\b",
    r"\bminimum\b",
    r"\bminimal\b",
    r"\bmin\b",
    r"\bleast\b",
    r"\bbottom\b",
    r"\bsmallest\b",
)

# Explicit fault_diagnosis metric fields for numeric /ask routing.
FAULT_DIAGNOSIS_METRICS: frozenset[str] = frozenset(
    {
        "avg_active_power",
        "avg_reactive_power",
        "avg_wind_speed",
    }
)

# Human-readable labels for deterministic numeric answers.
METRIC_LABELS: dict[str, str] = {
    "avg_active_power": "normalized active power",
    "avg_reactive_power": "normalized reactive power",
    "avg_wind_speed": "wind speed (m/s)",
    "avg_ac_power": "AC power (kW)",
    "avg_dc_power": "DC power (kW)",
    "avg_irradiation": "irradiation",
    "avg_ambient_temp": "ambient temperature (C)",
    "avg_module_temp": "module temperature (C)",
    "avg_theoretical_power": "theoretical power (kWh)",
    "avg_wind_direction": "wind direction (degrees)",
}

# Gauge / UI scaling maxima for numeric metric payloads.
METRIC_GAUGE_MAX: dict[str, float] = {
    "avg_wind_speed": 25.0,
    "avg_active_power": 1.0,
    "avg_reactive_power": 1.0,
    "avg_ac_power": 4000.0,
    "avg_dc_power": 4000.0,
    "avg_irradiation": 1.0,
    "avg_ambient_temp": 50.0,
    "avg_module_temp": 50.0,
    "avg_theoretical_power": 4000.0,
    "avg_wind_direction": 360.0,
}

# Natural-language / shorthand aliases → canonical numeric metadata fields.
# Longer aliases first so "wind speed" wins over bare "wind".
_METRIC_ALIASES: tuple[tuple[str, str], ...] = (
    ("avg_wind_speed", "avg_wind_speed"),
    ("avg_active_power", "avg_active_power"),
    ("avg_reactive_power", "avg_reactive_power"),
    ("avg_ac_power", "avg_ac_power"),
    ("avg_dc_power", "avg_dc_power"),
    ("avg_irradiation", "avg_irradiation"),
    ("avg_ambient_temp", "avg_ambient_temp"),
    ("avg_module_temp", "avg_module_temp"),
    ("avg_theoretical_power", "avg_theoretical_power"),
    ("avg_wind_direction", "avg_wind_direction"),
    ("normalized active power", "avg_active_power"),
    ("normalized reactive power", "avg_reactive_power"),
    ("wind_speed", "avg_wind_speed"),
    ("wind speed", "avg_wind_speed"),
    ("active_power", "avg_active_power"),
    ("active power", "avg_active_power"),
    ("reactive_power", "avg_reactive_power"),
    ("reactive power", "avg_reactive_power"),
    ("ac_power", "avg_ac_power"),
    ("ac power", "avg_ac_power"),
    ("dc_power", "avg_dc_power"),
    ("dc power", "avg_dc_power"),
    ("irradiation", "avg_irradiation"),
    ("irradiance", "avg_irradiation"),
    ("ambient_temp", "avg_ambient_temp"),
    ("ambient temperature", "avg_ambient_temp"),
    ("module_temp", "avg_module_temp"),
    ("module temperature", "avg_module_temp"),
    ("theoretical_power", "avg_theoretical_power"),
    ("theoretical power", "avg_theoretical_power"),
    ("wind_direction", "avg_wind_direction"),
    ("wind direction", "avg_wind_direction"),
)

# Prefer domain-relevant metrics when several aliases could match.
_DOMAIN_METRICS: dict[str, frozenset[str]] = {
    "wind": frozenset(
        {
            "avg_wind_speed",
            "avg_active_power",
            "avg_theoretical_power",
            "avg_wind_direction",
        }
    ),
    "solar": frozenset(
        {
            "avg_ac_power",
            "avg_dc_power",
            "avg_irradiation",
            "avg_ambient_temp",
            "avg_module_temp",
        }
    ),
    "fault_diagnosis": FAULT_DIAGNOSIS_METRICS,
}


def _detect_order(query_lower: str) -> str | None:
    """Return ``desc``, ``asc``, or None if no superlative keyword is present."""
    for pattern in _DESC_PATTERNS:
        if re.search(pattern, query_lower):
            return "desc"
    for pattern in _ASC_PATTERNS:
        if re.search(pattern, query_lower):
            return "asc"
    return None


def _detect_metric(query_lower: str, allowed: frozenset[str]) -> str | None:
    """Return the first matching canonical metric field within ``allowed``."""
    for alias, field in _METRIC_ALIASES:
        if field not in allowed:
            continue
        pattern = r"\b" + re.escape(alias).replace(r"\ ", r"[\s_]+") + r"\b"
        if re.search(pattern, query_lower):
            return field
    return None


def classify_query(
    query: str,
    manifest: DomainManifest,
) -> tuple[str, str] | None:
    """Classify a question as numeric ranking when both a superlative and metric match.

    Returns:
        ``(metric_field, order)`` where ``order`` is ``asc`` or ``desc``,
        or ``None`` to fall back to semantic retrieval.
    """
    query_lower = (query or "").strip().lower()
    if not query_lower:
        return None

    order = _detect_order(query_lower)
    if order is None:
        return None

    domain_allowed = _DOMAIN_METRICS.get(manifest.domain_id)
    if domain_allowed is not None:
        allowed = domain_allowed & NUMERIC_METRIC_FIELDS
    else:
        allowed = NUMERIC_METRIC_FIELDS

    if manifest.domain_id == "fault_diagnosis":
        allowed = allowed | (FAULT_DIAGNOSIS_METRICS & NUMERIC_METRIC_FIELDS)

    metric = _detect_metric(query_lower, allowed)
    if metric is None:
        return None

    return metric, order


def _format_metric_value(metric_field: str, value: float) -> str:
    """Format a metric for display (4 dp for unitless normalized powers)."""
    if metric_field in {"avg_active_power", "avg_reactive_power"} and abs(value) <= 1.05:
        return f"{value:.4f}"
    if metric_field in {"avg_irradiation"}:
        return f"{value:.4f}"
    return f"{value:.2f}"


def build_deterministic_numeric_answer(
    metric_field: str,
    order: str,
    top_chunk: dict[str, Any],
) -> str:
    """Build the ground-truth numeric sentence from the top ranked chunk."""
    meta = top_chunk.get("metadata") or {}
    raw = meta.get(metric_field, top_chunk.get("score"))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Top chunk is missing a usable value for metric {metric_field!r}"
        ) from exc

    label = METRIC_LABELS.get(metric_field, metric_field.replace("_", " "))
    superlative = "highest" if order == "desc" else "lowest"
    value_str = _format_metric_value(metric_field, value)

    equipment = meta.get("equipment_id") or "unknown equipment"
    ts_start = meta.get("timestamp_start") or "unknown start"
    ts_end = meta.get("timestamp_end") or "unknown end"

    return (
        f"The {superlative} {label} recorded was {value_str} "
        f"({equipment}, {ts_start} to {ts_end})."
    )


def build_metric_payload(
    metric_field: str,
    top_chunk: dict[str, Any],
) -> dict[str, Any]:
    """Build the numeric ``metric`` object for /ask responses."""
    meta = top_chunk.get("metadata") or {}
    raw = meta.get(metric_field, top_chunk.get("score"))
    value = float(raw)
    return {
        "field": metric_field,
        "value": round(value, 2),
        "max": float(METRIC_GAUGE_MAX.get(metric_field, 1.0)),
    }


def format_ask_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project retrieved chunks into enriched source stubs for the UI."""
    sources: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk.get("chunk_id") is None:
            continue
        meta = chunk.get("metadata") or {}
        sources.append(
            {
                "chunk_id": str(chunk["chunk_id"]),
                "equipment_id": meta.get("equipment_id"),
                "timestamp_start": meta.get("timestamp_start"),
                "timestamp_end": meta.get("timestamp_end"),
            }
        )
    return sources
