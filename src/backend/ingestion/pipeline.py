"""Local ingestion pipeline: load → chunk → write JSONL.

Produces ``data/processed/{domain_id}/chunks.jsonl`` for inspection.
Does not call Azure services.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config.domain_manifest import DomainManifest, load_manifest
from .chunkers import (
    build_fault_text,
    build_solar_text,
    build_wind_text,
    chunk_dataframe,
)
from .loaders import load_fault_diagnosis, load_solar, load_wind

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


def _date_range(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return ISO date-range strings from a loaded DataFrame."""
    if "DATE_TIME" not in df.columns or df["DATE_TIME"].isna().all():
        return None, None
    start = df["DATE_TIME"].min()
    end = df["DATE_TIME"].max()
    start_s = None if pd.isna(start) else str(pd.Timestamp(start).date())
    end_s = None if pd.isna(end) else str(pd.Timestamp(end).date())
    return start_s, end_s


def _write_jsonl(chunks: list[dict[str, Any]], output_path: Path) -> None:
    """Write one JSON object per line to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def run_ingestion(domain_id: str, **kwargs: Any) -> None:
    """Load a domain, chunk it, and write local JSONL artifacts.

    Keyword args (fault_diagnosis only):
        farm: str = "A"
        sample_frac: float = 0.05
    """
    manifest: DomainManifest = load_manifest(domain_id)

    if domain_id == "solar":
        df = load_solar()
        text_builder = build_solar_text
    elif domain_id == "wind":
        df = load_wind()
        text_builder = build_wind_text
    elif domain_id == "fault_diagnosis":
        farm = str(kwargs.get("farm", "A"))
        sample_frac = float(kwargs.get("sample_frac", 0.05))
        df = load_fault_diagnosis(farm=farm, sample_frac=sample_frac)
        text_builder = build_fault_text
    else:
        raise ValueError(
            f"Unsupported domain_id '{domain_id}'. "
            "Supported: solar, wind, fault_diagnosis."
        )

    chunks = chunk_dataframe(df, manifest, text_builder)
    output_path = PROCESSED_ROOT / domain_id / "chunks.jsonl"
    _write_jsonl(chunks, output_path)

    start, end = _date_range(df)
    print("Ingestion complete")
    print(f"  domain:     {domain_id}")
    print(f"  chunks:     {len(chunks)}")
    print(f"  date range: {start or 'n/a'} -> {end or 'n/a'}")
    print(f"  output:     {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GreenRAG local ingestion pipeline (CSV → JSONL chunks)."
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=["solar", "wind", "fault_diagnosis"],
        help="Domain to ingest.",
    )
    parser.add_argument(
        "--farm",
        default="A",
        choices=["A", "B", "C"],
        help="Wind farm letter for fault_diagnosis (default: A).",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.05,
        dest="sample_frac",
        help="Dev-mode row fraction for fault_diagnosis (default: 0.05).",
    )
    return parser


def main() -> None:
    """CLI entrypoint: ``python -m src.backend.ingestion.pipeline --domain solar``."""
    args = _build_parser().parse_args()
    kwargs: dict[str, Any] = {}
    if args.domain == "fault_diagnosis":
        kwargs["farm"] = args.farm
        kwargs["sample_frac"] = args.sample_frac
    run_ingestion(args.domain, **kwargs)


if __name__ == "__main__":
    main()
