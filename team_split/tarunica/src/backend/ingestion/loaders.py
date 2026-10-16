"""Dataset loaders for GreenRAG ingestion.

Each loader returns a pandas DataFrame ready for tabular chunking.
Paths are resolved relative to the project root (``dataset/...``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# GreenRAG project root: src/backend/ingestion/loaders.py → ../../../
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "dataset"


def _require_file(path: Path, hint: str) -> Path:
    """Return ``path`` if it exists; otherwise raise a clear FileNotFoundError."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required data file not found: {path}\n"
            f"Hint: {hint}"
        )
    return path


def _require_dir(path: Path, hint: str) -> Path:
    """Return ``path`` if it is a directory; otherwise raise FileNotFoundError."""
    if not path.is_dir():
        raise FileNotFoundError(
            f"Required data directory not found: {path}\n"
            f"Hint: {hint}"
        )
    return path


def _warn_nat_count(label: str, parsed: pd.Series) -> None:
    """Log how many timestamps failed to parse (NaT), instead of dropping silently."""
    nat_count = int(parsed.isna().sum())
    if nat_count:
        logger.warning(
            "%s: %d row(s) could not be parsed as DATE_TIME (NaT); "
            "check for unexpected formats.",
            label,
            nat_count,
        )


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read a CSV as UTF-8; retry with ``utf-8-sig`` if a BOM / mojibake appears.

    Prefer ``encoding='utf-8'``. If column names contain a BOM (``\\ufeff``) or
    classic mojibake (``Â`` before a symbol), re-read with ``utf-8-sig``.
    """
    df = pd.read_csv(path, encoding="utf-8", **kwargs)  # type: ignore[arg-type]
    cols = [str(c) for c in df.columns]
    needs_sig = any(c.startswith("\ufeff") or "Â" in c for c in cols)
    if needs_sig:
        df = pd.read_csv(path, encoding="utf-8-sig", **kwargs)  # type: ignore[arg-type]
    return df


def load_solar() -> pd.DataFrame:
    """Load Plant 1/2 generation + weather CSVs and join on DATE_TIME per plant.

    Returns a single DataFrame with a ``plant_id`` column (1 or 2) and
    weather columns merged onto each generation row.
    """
    solar_dir = _require_dir(
        DATASET_ROOT / "solar",
        "Place Plant_*_Generation_Data.csv and Plant_*_Weather_Sensor_Data.csv under dataset/solar/.",
    )

    frames: list[pd.DataFrame] = []
    for plant_id in (1, 2):
        gen_path = _require_file(
            solar_dir / f"Plant_{plant_id}_Generation_Data.csv",
            f"Expected Plant_{plant_id}_Generation_Data.csv in dataset/solar/.",
        )
        weather_path = _require_file(
            solar_dir / f"Plant_{plant_id}_Weather_Sensor_Data.csv",
            f"Expected Plant_{plant_id}_Weather_Sensor_Data.csv in dataset/solar/.",
        )

        gen = _read_csv(gen_path)
        weather = _read_csv(weather_path)

        required_gen = {"DATE_TIME", "SOURCE_KEY", "DC_POWER", "AC_POWER"}
        required_wx = {"DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"}
        if not required_gen.issubset(gen.columns):
            raise ValueError(
                f"{gen_path.name} is missing columns {required_gen - set(gen.columns)}"
            )
        if not required_wx.issubset(weather.columns):
            raise ValueError(
                f"{weather_path.name} is missing columns {required_wx - set(weather.columns)}"
            )

        gen = gen.copy()
        weather = weather.copy()

        # Plant 1 generation DATE_TIME is day-first (e.g. 15-05-2020 00:00).
        # Relying on pandas auto-detection is unsafe because day/month can be swapped.
        gen_raw = gen["DATE_TIME"].copy()
        gen["DATE_TIME"] = pd.to_datetime(
            gen_raw, format="%d-%m-%Y %H:%M", errors="coerce"
        )
        # Plant 2 generation uses ISO timestamps; fill only the still-unparsed rows.
        still_nat = gen["DATE_TIME"].isna()
        if still_nat.any():
            gen.loc[still_nat, "DATE_TIME"] = pd.to_datetime(
                gen_raw.loc[still_nat],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
        _warn_nat_count(f"Plant {plant_id} generation ({gen_path.name})", gen["DATE_TIME"])

        # Weather files in this dataset use ISO-like timestamps; parse explicitly.
        weather["DATE_TIME"] = pd.to_datetime(
            weather["DATE_TIME"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )
        _warn_nat_count(f"Plant {plant_id} weather ({weather_path.name})", weather["DATE_TIME"])

        # Weather has one sensor row per timestamp; keep weather metrics only.
        weather_cols = [
            "DATE_TIME",
            "AMBIENT_TEMPERATURE",
            "MODULE_TEMPERATURE",
            "IRRADIATION",
        ]
        weather = weather[weather_cols].drop_duplicates(subset=["DATE_TIME"])

        merged = gen.merge(weather, on="DATE_TIME", how="left")
        merged["plant_id"] = plant_id
        merged["source_file"] = f"Plant_{plant_id}_Generation_Data.csv|+Weather"
        frames.append(merged)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["plant_id", "SOURCE_KEY", "DATE_TIME"]
    ).reset_index(drop=True)
    return combined


def load_wind() -> pd.DataFrame:
    """Load the Scada wind turbine file ``dataset/wind/T1.csv`` as-is (plus helpers)."""
    wind_path = _require_file(
        DATASET_ROOT / "wind" / "T1.csv",
        "Place T1.csv under dataset/wind/.",
    )

    df = _read_csv(wind_path)
    if "Date/Time" not in df.columns:
        raise ValueError(f"{wind_path.name} is missing required column 'Date/Time'")

    df = df.copy()
    df["DATE_TIME"] = pd.to_datetime(
        df["Date/Time"], format="%d %m %Y %H:%M", errors="coerce"
    )
    df["source_file"] = "T1.csv"
    df["equipment_id"] = "T1"
    df = df.sort_values("DATE_TIME").reset_index(drop=True)
    return df


def load_fault_diagnosis(
    farm: str = "A",
    sample_frac: float = 0.05,
) -> pd.DataFrame:
    """Load SCADA turbine CSVs for one wind farm, with optional event context.

    Turbine files live under ``dataset/fault_diagnosis/Wind Farm {farm}/datasets/``.
    Files whose names start with ``comma_`` are skipped.

    Dev-mode sampling switch (NOT for final evaluation):
        These SCADA files are large. When ``sample_frac < 1.0`` we keep every
        N-th row where ``N = max(1, round(1 / sample_frac))`` so local
        ingestion stays fast and deterministic. Use ``sample_frac=1.0`` only
        when preparing full evaluation corpora.
    """
    if not 0.0 < sample_frac <= 1.0:
        raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}")

    farm = farm.strip().upper()
    farm_dir = _require_dir(
        DATASET_ROOT / "fault_diagnosis" / f"Wind Farm {farm}",
        f"Expected folder dataset/fault_diagnosis/Wind Farm {farm}/.",
    )
    datasets_dir = _require_dir(
        farm_dir / "datasets",
        f"Expected turbine CSVs under Wind Farm {farm}/datasets/.",
    )

    # Deterministic stride: sample_frac=0.05 → keep every 20th row.
    stride = max(1, int(round(1.0 / sample_frac)))

    csv_files = sorted(
        path
        for path in datasets_dir.glob("*.csv")
        if not path.name.lower().startswith("comma_")
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No turbine CSV files found in {datasets_dir} "
            f"(after excluding comma_* files)."
        )

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        # Filename stem is the turbine / asset id (e.g. "0.csv" → "0").
        turbine_id = path.stem
        try:
            raw = _read_csv(path, sep=";")
        except Exception as exc:  # noqa: BLE001 — surface path in the error
            raise ValueError(f"Failed to read turbine file {path}: {exc}") from exc

        if raw.empty:
            continue

        # Dev-mode downsampling — see docstring above.
        sampled = raw.iloc[::stride].copy()
        sampled["turbine_id"] = turbine_id
        sampled["source_file"] = path.name
        frames.append(sampled)

    if not frames:
        raise ValueError(f"All turbine files under {datasets_dir} were empty.")

    df = pd.concat(frames, ignore_index=True)

    if "time_stamp" in df.columns:
        df["DATE_TIME"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    else:
        df["DATE_TIME"] = pd.NaT

    # Side-car context tables (kept on attrs for chunk text builders).
    event_path = farm_dir / "event_info.csv"
    feature_path = farm_dir / "feature_description.csv"
    event_info = _read_csv(event_path, sep=";") if event_path.is_file() else pd.DataFrame()
    feature_description = (
        _read_csv(feature_path, sep=";") if feature_path.is_file() else pd.DataFrame()
    )

    if event_info.empty:
        print(
            f"Warning: event_info.csv not found or empty under {farm_dir}; "
            "fault text chunks will omit event descriptions."
        )
    if feature_description.empty:
        print(
            f"Warning: feature_description.csv not found or empty under {farm_dir}."
        )

    df.attrs["event_info"] = event_info
    df.attrs["feature_description"] = feature_description
    df.attrs["farm"] = farm
    df.attrs["sample_frac"] = sample_frac
    df.attrs["sample_stride"] = stride

    df = df.sort_values(["turbine_id", "DATE_TIME"]).reset_index(drop=True)
    return df
