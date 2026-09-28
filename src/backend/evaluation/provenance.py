"""Run provenance: package versions and git commit, saved next to every result."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PACKAGES = (
    "numpy", "pandas", "scikit-learn", "xgboost", "torch", "statsmodels",
    "matplotlib", "sentence-transformers", "chromadb", "fastapi", "pydantic",
)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    dirty = subprocess.run(
        # Generated outputs under results/ do not make the code state dirty.
        ["git", "status", "--porcelain", "--", ".", ":(exclude)results"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    return out.stdout.strip() + ("-dirty" if dirty else "")


def environment_info(packages: tuple[str, ...] = DEFAULT_PACKAGES) -> dict:
    """Return python/platform/package versions and the current git commit."""
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "packages": versions,
    }
