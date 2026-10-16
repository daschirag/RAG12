"""GreenRAG ingestion package: loaders, chunkers, and local pipeline."""

from .chunkers import (
    NUMERIC_METRIC_FIELDS,
    build_fault_text,
    build_solar_text,
    build_wind_text,
    chunk_dataframe,
    extract_fault_metrics,
    extract_solar_metrics,
    extract_wind_metrics,
)
from .loaders import load_fault_diagnosis, load_solar, load_wind

__all__ = [
    "NUMERIC_METRIC_FIELDS",
    "build_fault_text",
    "build_solar_text",
    "build_wind_text",
    "chunk_dataframe",
    "extract_fault_metrics",
    "extract_solar_metrics",
    "extract_wind_metrics",
    "load_fault_diagnosis",
    "load_solar",
    "load_wind",
    "run_ingestion",
]


def __getattr__(name: str):
    """Lazy-export ``run_ingestion`` to avoid runpy double-import warnings."""
    if name == "run_ingestion":
        from .pipeline import run_ingestion

        return run_ingestion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
