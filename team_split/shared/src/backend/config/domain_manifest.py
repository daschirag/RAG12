"""Domain manifest schema and loaders for GreenRAG.

Each renewable-energy domain (solar, wind, fault_diagnosis, …) is described
by a YAML file under ``domain_manifests/``. This module validates those files
into a typed ``DomainManifest`` used by the API and (later) by retrieval.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


# Directory that holds one YAML file per domain: {domain_id}.yaml
MANIFESTS_DIR = Path(__file__).resolve().parent / "domain_manifests"


class DomainManifest(BaseModel):
    """Typed configuration for a single GreenRAG knowledge domain."""

    domain_id: str
    display_name: str
    corpus_sources: list[str]
    chunking_strategy: str
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    prompt_template: str
    access_policy: str
    retrieval_top_k: int = Field(default=5)


def load_manifest(domain_id: str) -> DomainManifest:
    """Load and validate ``domain_manifests/{domain_id}.yaml``.

    Raises:
        FileNotFoundError: If no manifest exists for ``domain_id``.
        ValueError: If the YAML cannot be parsed or fails validation.
    """
    path = MANIFESTS_DIR / f"{domain_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Domain manifest not found: {domain_id}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in manifest '{domain_id}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Manifest '{domain_id}' must be a YAML mapping")

    return DomainManifest.model_validate(raw)


def list_domains() -> list[str]:
    """Return sorted domain_ids discovered from ``*.yaml`` filenames."""
    if not MANIFESTS_DIR.is_dir():
        return []

    return sorted(path.stem for path in MANIFESTS_DIR.glob("*.yaml") if path.is_file())
