"""B1 - Fault taxonomy derived from the real event logs (event_info.csv, farms A-C).

The categories are wind-turbine *components*, because that is what the event
descriptions name (e.g. "Hydraulic group", "Pitch failure - defect fan on
pitch motor"). Each description is mapped by transparent keyword rules; an
event can touch several components ("+"-joined descriptions). The primary
component is the one whose keyword appears first in the text.

``default_severity`` is a project-defined assumption (drive-train and
grid-connection components high, auxiliaries lower) used later by the
deterministic severity mapping in B5; it is not a label from the dataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FAULT_ROOT = PROJECT_ROOT / "dataset" / "fault_diagnosis"
FARMS = ("A", "B", "C")


def farm_dir(farm: str) -> Path:
    return FAULT_ROOT / f"Wind Farm {farm}"


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_events(farm: str) -> pd.DataFrame:
    """event_info.csv with a uniform ``asset_id`` column and 0-based ``source_row``."""
    path = farm_dir(farm) / "event_info.csv"
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df = df.rename(columns={"asset": "asset_id"})         # Farm A names the column "asset"
    required = {"asset_id", "event_id", "event_label", "event_start", "event_end", "event_description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["source_row"] = range(len(df))
    df["farm"] = farm
    df["event_start"] = pd.to_datetime(df["event_start"])
    df["event_end"] = pd.to_datetime(df["event_end"])
    # pandas 3 keeps NaN under .where(..., None) for string columns; build the object column explicitly.
    df["event_description"] = [d if isinstance(d, str) and d.strip() else None for d in df["event_description"]]
    return df


def load_all_events() -> pd.DataFrame:
    return pd.concat([load_events(f) for f in FARMS], ignore_index=True)


UNIT_FIXES = {"�C": "degC", "�": "deg", "Celsius": "degC"}


def load_features(farm: str) -> pd.DataFrame:
    """feature_description.csv; the degree sign is stored as U+FFFD in farms A/B, normalised to ASCII."""
    path = farm_dir(farm) / "feature_description.csv"
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df["source_row"] = range(len(df))
    df["description"] = df["description"].fillna("").str.strip()
    df["unit"] = df["unit"].fillna("").map(lambda u: UNIT_FIXES.get(u, u))
    return df


@dataclass(frozen=True)
class Component:
    key: str
    label: str
    event_patterns: tuple[str, ...]       # regexes on the lower-cased event description
    sensor_patterns: tuple[str, ...]      # regexes on the lower-cased sensor description
    default_severity: str                 # project assumption, see module docstring


COMPONENTS: tuple[Component, ...] = (
    Component("transformer", "Transformer", (r"transformer", r"transformator"),
              (r"transformer", r"transformator"), "high"),
    Component("gearbox", "Gearbox and gear-oil system",
              (r"gearbox", r"gear oil", r"oil level", r"oil leakage"),
              (r"gearbox", r"gear oil", r"planetary bearing"), "high"),
    Component("generator_bearing", "Generator bearings", (r"generator bearing",),
              (r"generator bearing",), "high"),
    Component("rotor_main_bearing", "Rotor / main bearing", (r"rotor ?bearing", r"main bearing"),
              (r"rotor bearing", r"axial bearing"), "high"),
    Component("converter", "Power converter", (r"converter", r"umrichter", r"fuse filter"),
              (r"converter", r"inverter", r"igbt"), "high"),
    Component("hydraulic_system", "Hydraulic system", (r"hydraulic", r"accumulator"),
              (r"hydraulic",), "medium"),
    Component("pitch_system", "Pitch system (blade pitch drives, batteries, controllers)",
              (r"pitch", r"axis ?\d", r"dc-link", r"batt", r"beckhoff", r"encoder", r"blade control"),
              (r"pitch", r"axis ?\d", r"battery"), "medium"),
    Component("rotor_brake", "Rotor brake", (r"rotor ?brake", r"rotorbremse", r"brake disc"),
              (r"rotor brake",), "medium"),
    Component("yaw_system", "Yaw system", (r"yaw",), (r"yaw", r"azimuth", r"nacelle direction"), "medium"),
    Component("cooling_system", "Cooling system", (r"cooling", r"cooler"),
              (r"cooling", r"cooler", r"water"), "medium"),
    Component("electrical_auxiliary", "Auxiliary electrical supply and safety chain",
              (r"own consumption", r"\brcd\b", r"24 ?vac", r"versorgung", r"safety chain"),
              (r"internal consumption", r"24 ?vdc", r"power pack", r"electrical cabinet"), "medium"),
    Component("communication_control", "Communication and control",
              (r"communication", r"bk1120", r"harting", r"\bwiring"),     # \b: not "rewiring"
              (r"controller", r"control box", r"board temperature"), "low"),
    Component("blades_hub_structure", "Blades, hub and structural items",
              (r"blade", r"spinner", r"grease collector", r"lightning", r"grounding", r"caps"),
              (r"spinner", r"nose cone", r"hub temperature"), "low"),
)
COMPONENT_BY_KEY = {c.key: c for c in COMPONENTS}
UNSPECIFIED = "unspecified"


def categorize(description: str | None) -> tuple[str, list[str]]:
    """Return (primary component, all components) for an event description."""
    if not isinstance(description, str) or not description.strip():
        return UNSPECIFIED, []
    text = description.lower()
    hits: list[tuple[int, str]] = []
    for comp in COMPONENTS:
        positions = [m.start() for p in comp.event_patterns for m in re.finditer(p, text)]
        if positions:
            hits.append((min(positions), comp.key))
    if not hits:
        return UNSPECIFIED, []
    hits.sort()
    return hits[0][1], [k for _, k in hits]


def sensors_for(component: Component, features: pd.DataFrame) -> pd.DataFrame:
    text = features["description"].str.lower()
    mask = pd.Series(False, index=features.index)
    for p in component.sensor_patterns:
        mask |= text.str.contains(p, regex=True)
    return features[mask]


def taxonomy_table(events: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per anomaly event with its derived categories."""
    events = load_all_events() if events is None else events
    anomalies = events[events["event_label"] == "anomaly"].copy()
    cats = anomalies["event_description"].map(categorize)
    anomalies["primary_component"] = [c[0] for c in cats]
    anomalies["components"] = [c[1] for c in cats]
    return anomalies


# The plan's fault list is for solar inverters; this is how it relates to the real wind data.
PLAN_FAULTS_VS_DATA: tuple[tuple[str, str], ...] = (
    ("inverter overheating",
     "No inverter events (the data is wind turbines). Closest real events: transformer over-temperature "
     "(Farm B 'high temperature in transformer cell' x3, Farm A 'Transformer failure', Farm C main-transformer "
     "overpressure) and two Farm C power-converter events (converter failure / fuse filter; converter torque "
     "oscillations)."),
    ("abnormal voltage",
     "Only Farm C event 70 'Axis 1 DC-link voltage low' - a pitch-axis battery/DC-link issue, not grid or "
     "inverter voltage."),
    ("abnormal current",
     "Only Farm C event 5 'current measurement own consumption' - an auxiliary-supply measurement fault."),
    ("underperformance",
     "Not an event label. The SCADA files carry a status_type (1 = derated operation) from which it could be "
     "derived, but event_info.csv has no underperformance events."),
    ("sensor anomaly",
     "Not an event label. Some descriptions mention defective measurement hardware (pitch encoder, current "
     "measurement), which the taxonomy files under the affected component."),
    ("power-generation deviation",
     "Not an event label; would have to be derived from SCADA (power vs power curve), as done for the forecasting "
     "downtime flag in Track A."),
)
