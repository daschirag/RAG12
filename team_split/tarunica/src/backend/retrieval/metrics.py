"""Shared helpers for numeric metadata / metric ranking."""

from __future__ import annotations

from typing import Any

# Keep in sync with ingestion.chunkers.NUMERIC_METRIC_FIELDS.
NUMERIC_METRIC_FIELDS: frozenset[str] = frozenset(
    {
        "avg_ac_power",
        "avg_dc_power",
        "avg_irradiation",
        "avg_ambient_temp",
        "avg_module_temp",
        "avg_active_power",
        "avg_wind_speed",
        "avg_theoretical_power",
        "avg_wind_direction",
        "avg_reactive_power",
    }
)


def validate_metric_request(metric_field: str, order: str) -> tuple[str, str]:
    """Validate metric_field / order; return normalized (field, order)."""
    field = (metric_field or "").strip()
    if field not in NUMERIC_METRIC_FIELDS:
        allowed = ", ".join(sorted(NUMERIC_METRIC_FIELDS))
        raise ValueError(
            f"Unsupported metric_field={metric_field!r}. Allowed: {allowed}"
        )

    normalized_order = (order or "desc").strip().lower()
    if normalized_order not in {"asc", "desc"}:
        raise ValueError("order must be 'asc' or 'desc'")
    return field, normalized_order


def flatten_chunk_metadata(
    domain_id: str,
    meta: dict[str, Any] | None,
) -> dict[str, Any]:
    """Flatten chunk metadata for Chroma (scalars only), keeping numeric metrics as float."""
    flat: dict[str, Any] = {"domain_id": domain_id}
    if not isinstance(meta, dict):
        return flat

    for key, value in meta.items():
        if value is None:
            continue
        if key in NUMERIC_METRIC_FIELDS:
            try:
                flat[key] = float(value)
            except (TypeError, ValueError):
                continue
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = str(value)
    return flat


def extract_metric_fields(meta: dict[str, Any] | None) -> dict[str, float]:
    """Pull known numeric metrics out of a metadata dict for Azure top-level fields."""
    out: dict[str, float] = {}
    if not isinstance(meta, dict):
        return out
    for key in NUMERIC_METRIC_FIELDS:
        if key not in meta or meta[key] is None:
            continue
        try:
            out[key] = float(meta[key])
        except (TypeError, ValueError):
            continue
    return out
