"""Tabular chunking and natural-language text builders for GreenRAG.

Mapping note — ``DomainManifest.chunk_size`` / ``chunk_overlap``:
    Manifest fields were originally intended for *token*-based text chunking.
    For tabular SCADA / generation CSVs we reinterpret them as **row counts**:
      - ``chunk_size``  → number of DataFrame rows per chunk window
      - ``chunk_overlap`` → number of overlapping rows between consecutive windows
    Keep this mapping in mind when tuning manifests for ingestion.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from ..config.domain_manifest import DomainManifest


def _safe_mean(series: pd.Series) -> float | None:
    """Return mean of numeric values, or None if empty / all-NaN."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        return None
    return float(numeric.mean())


def _fmt(value: float | None, digits: int = 2) -> str:
    """Format an optional float for prose, or ``n/a``."""
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _sensor_base_name(column: str) -> str:
    """Map ``power_29_avg`` → ``power_29`` for feature_description lookups."""
    base = column
    for suffix in ("_avg", "_max", "_min", "_std"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _unit_from_feature_description(
    feature_description: pd.DataFrame | None,
    column: str,
) -> str | None:
    """Return the catalog unit for a SCADA column, if listed."""
    if feature_description is None or feature_description.empty:
        return None
    if not {"sensor_name", "unit"}.issubset(feature_description.columns):
        return None

    base = _sensor_base_name(column)
    matches = feature_description[
        feature_description["sensor_name"].astype(str) == base
    ]
    if matches.empty:
        return None

    unit = str(matches.iloc[0]["unit"]).strip()
    if not unit or unit.lower() in {"nan", "none", "null"}:
        return None
    return unit


def _is_normalized_metric(
    values: pd.Series,
    unit: str | None = None,
) -> bool:
    """True if values look unitless / normalized (or the catalog unit says so).

    Catalog units can be stale (e.g. labeled ``kW`` while values sit in [0, 1]);
    prefer an explicit 'normal*' unit, otherwise inspect the observed range.
    """
    if unit and "normal" in unit.lower():
        return True

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return False

    vmin = float(numeric.min())
    vmax = float(numeric.max())
    # Roughly bounded in [0, 1] or [-1, 1] — typical for min-max / scaled SCADA.
    return vmin >= -1.05 and vmax <= 1.05


def _format_power_phrase(
    kind: str,
    mean_value: float | None,
    column: str,
    window: pd.DataFrame,
    *,
    digits: int = 2,
) -> str:
    """Build an active/reactive power phrase without asserting a false unit."""
    feature_description = (
        window.attrs.get("feature_description") if hasattr(window, "attrs") else None
    )
    unit = _unit_from_feature_description(
        feature_description if isinstance(feature_description, pd.DataFrame) else None,
        column,
    )
    series = window[column] if column in window.columns else pd.Series(dtype=float)
    if _is_normalized_metric(series, unit):
        # Unitless 0-1 values: allow higher precision for ranking questions.
        return f"normalized {kind} {_fmt(mean_value, digits)}"

    if unit:
        return f"{kind} {_fmt(mean_value, digits)} {unit}"
    return f"{kind} {_fmt(mean_value, digits)}"


def _window_timestamps(window: pd.DataFrame) -> tuple[str | None, str | None]:
    """Extract ISO start/end timestamps from a ``DATE_TIME`` column if present."""
    if "DATE_TIME" not in window.columns or window["DATE_TIME"].isna().all():
        return None, None
    start = window["DATE_TIME"].min()
    end = window["DATE_TIME"].max()
    start_s = None if pd.isna(start) else pd.Timestamp(start).isoformat(sep=" ")
    end_s = None if pd.isna(end) else pd.Timestamp(end).isoformat(sep=" ")
    return start_s, end_s


def _put_metric(metrics: dict[str, float], key: str, value: float | None) -> None:
    """Store a float metric when present (skip None so stores stay clean)."""
    if value is not None:
        metrics[key] = float(value)


def extract_solar_metrics(window: pd.DataFrame) -> dict[str, float]:
    """Numeric aggregates mirrored in ``build_solar_text``."""
    metrics: dict[str, float] = {}
    _put_metric(
        metrics,
        "avg_ac_power",
        _safe_mean(window["AC_POWER"]) if "AC_POWER" in window.columns else None,
    )
    _put_metric(
        metrics,
        "avg_dc_power",
        _safe_mean(window["DC_POWER"]) if "DC_POWER" in window.columns else None,
    )
    _put_metric(
        metrics,
        "avg_irradiation",
        _safe_mean(window["IRRADIATION"]) if "IRRADIATION" in window.columns else None,
    )
    _put_metric(
        metrics,
        "avg_ambient_temp",
        _safe_mean(window["AMBIENT_TEMPERATURE"])
        if "AMBIENT_TEMPERATURE" in window.columns
        else None,
    )
    _put_metric(
        metrics,
        "avg_module_temp",
        _safe_mean(window["MODULE_TEMPERATURE"])
        if "MODULE_TEMPERATURE" in window.columns
        else None,
    )
    return metrics


def extract_wind_metrics(window: pd.DataFrame) -> dict[str, float]:
    """Numeric aggregates mirrored in ``build_wind_text``."""
    metrics: dict[str, float] = {}
    power_col = "LV ActivePower (kW)"
    speed_col = "Wind Speed (m/s)"
    theory_col = "Theoretical_Power_Curve (KWh)"
    direction_cols = [c for c in window.columns if c.startswith("Wind Direction")]
    direction_col = direction_cols[0] if direction_cols else None

    _put_metric(
        metrics,
        "avg_active_power",
        _safe_mean(window[power_col]) if power_col in window.columns else None,
    )
    _put_metric(
        metrics,
        "avg_wind_speed",
        _safe_mean(window[speed_col]) if speed_col in window.columns else None,
    )
    _put_metric(
        metrics,
        "avg_theoretical_power",
        _safe_mean(window[theory_col]) if theory_col in window.columns else None,
    )
    _put_metric(
        metrics,
        "avg_wind_direction",
        _safe_mean(window[direction_col]) if direction_col is not None else None,
    )
    return metrics


def extract_fault_metrics(window: pd.DataFrame) -> dict[str, float]:
    """Numeric aggregates mirrored in ``build_fault_text`` (and chunk metadata).

    Always emit these float fields when the underlying SCADA columns exist —
    they are required for ``search_by_metric`` / superlative /ask routing:
      - ``avg_wind_speed``      ← wind_speed_3_avg
      - ``avg_active_power``    ← power_29_avg (often normalized in [0, 1])
      - ``avg_reactive_power``  ← reactive_power_27_avg (often normalized)
    """
    metrics: dict[str, float] = {}
    _put_metric(
        metrics,
        "avg_wind_speed",
        _safe_mean(window["wind_speed_3_avg"])
        if "wind_speed_3_avg" in window.columns
        else None,
    )
    _put_metric(
        metrics,
        "avg_active_power",
        _safe_mean(window["power_29_avg"])
        if "power_29_avg" in window.columns
        else None,
    )
    _put_metric(
        metrics,
        "avg_reactive_power",
        _safe_mean(window["reactive_power_27_avg"])
        if "reactive_power_27_avg" in window.columns
        else None,
    )
    return metrics


# Domain → metric extractor used when building chunk metadata.
METRIC_EXTRACTORS: dict[str, Callable[[pd.DataFrame], dict[str, float]]] = {
    "solar": extract_solar_metrics,
    "wind": extract_wind_metrics,
    "fault_diagnosis": extract_fault_metrics,
}

# Metric field names promoted into vector-store metadata / Azure filterable fields.
NUMERIC_METRIC_FIELDS: tuple[str, ...] = (
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
)


def build_solar_text(window: pd.DataFrame) -> str:
    """Summarize a solar generation+weather window into a short paragraph."""
    start, end = _window_timestamps(window)
    plant = window["plant_id"].iloc[0] if "plant_id" in window.columns else "?"
    inverter = (
        str(window["SOURCE_KEY"].iloc[0]) if "SOURCE_KEY" in window.columns else "unknown"
    )
    m = extract_solar_metrics(window)
    ac = m.get("avg_ac_power")
    dc = m.get("avg_dc_power")
    irr = m.get("avg_irradiation")
    amb = m.get("avg_ambient_temp")
    mod = m.get("avg_module_temp")

    return (
        f"Between {start or 'unknown'} and {end or 'unknown'}, Plant {plant} "
        f"inverter {inverter} produced an average AC power of {_fmt(ac)} kW "
        f"(DC {_fmt(dc)} kW) under {_fmt(irr, 4)} irradiation (unitless fraction) "
        f"with ambient {_fmt(amb)} °C and module {_fmt(mod)} °C."
    )


def build_wind_text(window: pd.DataFrame) -> str:
    """Summarize a wind SCADA window into a short paragraph."""
    start, end = _window_timestamps(window)
    m = extract_wind_metrics(window)
    power = m.get("avg_active_power")
    speed = m.get("avg_wind_speed")
    theory = m.get("avg_theoretical_power")
    direction = m.get("avg_wind_direction")

    return (
        f"Between {start or 'unknown'} and {end or 'unknown'}, turbine T1 "
        f"delivered an average active power of {_fmt(power)} kW at a mean wind "
        f"speed of {_fmt(speed)} m/s (theoretical curve {_fmt(theory)} kWh) "
        f"with mean wind direction {_fmt(direction)} degrees."
    )


def build_fault_text(window: pd.DataFrame) -> str:
    """Summarize a fault-diagnosis SCADA window, including event hints when known.

    Uses the same ``extract_fault_metrics`` values that ``chunk_dataframe``
    writes into metadata (``avg_wind_speed``, ``avg_active_power``,
    ``avg_reactive_power``) so text and numeric ranking stay aligned.
    """
    start, end = _window_timestamps(window)
    turbine = (
        str(window["turbine_id"].iloc[0]) if "turbine_id" in window.columns else "unknown"
    )
    farm = window.attrs.get("farm", "?") if hasattr(window, "attrs") else "?"

    # Same normalized aggregates stored on the chunk metadata dict.
    m = extract_fault_metrics(window)
    wind = m.get("avg_wind_speed")
    power = m.get("avg_active_power")
    reactive = m.get("avg_reactive_power")

    power_phrase = _format_power_phrase(
        "active power", power, "power_29_avg", window, digits=4
    )
    reactive_phrase = _format_power_phrase(
        "reactive power", reactive, "reactive_power_27_avg", window, digits=4
    )

    status_note = "no status flags highlighted"
    if "status_type_id" in window.columns:
        status_vals = sorted(
            {int(v) for v in window["status_type_id"].dropna().unique()}
        )
        if status_vals:
            status_note = f"status_type_id values observed: {status_vals}"

    event_note = "no matching event_info rows in this window"
    event_info = window.attrs.get("event_info") if hasattr(window, "attrs") else None
    if (
        isinstance(event_info, pd.DataFrame)
        and not event_info.empty
        and start is not None
        and end is not None
    ):
        # event_info uses semicolon CSV columns: asset, event_label, event_description, …
        try:
            events = event_info.copy()
            if "asset" in events.columns:
                events = events[
                    events["asset"].astype(str) == str(turbine)
                ]
            if {"event_start", "event_end"}.issubset(events.columns) and not events.empty:
                w_start = pd.Timestamp(start)
                w_end = pd.Timestamp(end)
                events["event_start"] = pd.to_datetime(
                    events["event_start"], errors="coerce"
                )
                events["event_end"] = pd.to_datetime(
                    events["event_end"], errors="coerce"
                )
                overlap = events[
                    (events["event_start"] <= w_end) & (events["event_end"] >= w_start)
                ]
                if not overlap.empty:
                    labels = (
                        overlap["event_description"]
                        if "event_description" in overlap.columns
                        else overlap.get("event_label", pd.Series(dtype=str))
                    )
                    event_note = "related events: " + "; ".join(
                        str(x) for x in labels.dropna().unique()[:3]
                    )
        except Exception:  # noqa: BLE001 — keep chunk text robust
            event_note = "event_info present but could not be matched for this window"

    return (
        f"Between {start or 'unknown'} and {end or 'unknown'}, Wind Farm {farm} "
        f"turbine {turbine} reported average wind speed {_fmt(wind)} m/s, "
        f"{power_phrase} (sensor power_29_avg), and {reactive_phrase} "
        f"(reactive_power_27_avg). {status_note.capitalize()}. "
        f"Event context: {event_note}."
    )


def chunk_dataframe(
    df: pd.DataFrame,
    manifest: DomainManifest,
    text_builder: Callable[[pd.DataFrame], str],
) -> list[dict[str, Any]]:
    """Slice ``df`` into overlapping row windows and build text chunks.

    ``text_builder`` receives each window (a DataFrame slice) and must return
    a natural-language string. See module docstring for the row-count mapping
    of ``manifest.chunk_size`` / ``chunk_overlap``.

    Metadata includes both provenance fields and float metric aggregates
    (e.g. ``avg_wind_speed``, ``avg_ac_power``) for numeric top-k queries.
    """
    if df.empty:
        return []

    chunk_size = max(1, int(manifest.chunk_size))
    overlap = max(0, int(manifest.chunk_overlap))
    if overlap >= chunk_size:
        # Avoid zero/negative stride; keep at least one new row per window.
        overlap = chunk_size - 1

    step = chunk_size - overlap
    chunks: list[dict[str, Any]] = []
    n = len(df)
    chunk_index = 0
    metric_extractor = METRIC_EXTRACTORS.get(manifest.domain_id)

    for start in range(0, n, step):
        end = min(start + chunk_size, n)
        window = df.iloc[start:end].copy()
        # Preserve farm/event attrs for fault text builders.
        if hasattr(df, "attrs") and df.attrs:
            window.attrs.update(df.attrs)

        text = text_builder(window)
        ts_start, ts_end = _window_timestamps(window)

        equipment_id: str | None = None
        if "SOURCE_KEY" in window.columns:
            equipment_id = str(window["SOURCE_KEY"].iloc[0])
        elif "equipment_id" in window.columns:
            equipment_id = str(window["equipment_id"].iloc[0])
        elif "turbine_id" in window.columns:
            equipment_id = str(window["turbine_id"].iloc[0])

        source_file = None
        if "source_file" in window.columns:
            source_file = str(window["source_file"].iloc[0])

        metadata: dict[str, Any] = {
            "source_file": source_file,
            "row_range": [start, end - 1],
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
            "equipment_id": equipment_id,
        }
        if metric_extractor is not None:
            metadata.update(metric_extractor(window))

        chunks.append(
            {
                "chunk_id": f"{manifest.domain_id}_{chunk_index:06d}",
                "domain_id": manifest.domain_id,
                "text": text,
                "metadata": metadata,
            }
        )
        chunk_index += 1

        if end >= n:
            break

    return chunks
