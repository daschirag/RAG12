"""Run provenance: package versions and base commit, saved next to every result.

Project rule: no git commands are run. The base commit is read directly from
the files under ``.git`` (HEAD, loose refs, packed-refs). Whether the working
tree has uncommitted changes cannot be known without git, and all work is left
uncommitted by rule, so the stamp is ``<commit>-dirty`` and
``git_dirty_checked`` is False.
"""

from __future__ import annotations

import importlib.metadata
import platform
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PACKAGES = (
    "numpy", "pandas", "scikit-learn", "xgboost", "torch", "statsmodels",
    "matplotlib", "sentence-transformers", "chromadb", "fastapi", "pydantic",
)


def read_head_commit(git_dir: Path | None = None) -> str | None:
    """Return the full SHA of HEAD by reading .git files, or None if unavailable."""
    git_dir = git_dir or PROJECT_ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head or None  # detached HEAD holds the SHA itself
    ref = head.split(":", 1)[1].strip()
    try:
        return (git_dir / ref).read_text(encoding="utf-8").strip() or None
    except OSError:
        pass
    try:
        packed = (git_dir / "packed-refs").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in packed.splitlines():
        parts = line.strip().split(" ")
        if len(parts) == 2 and parts[1] == ref:
            return parts[0]
    return None


def environment_info(packages: tuple[str, ...] = DEFAULT_PACKAGES) -> dict:
    """Return python/platform/package versions and the base commit stamp."""
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    sha = read_head_commit()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": f"{sha[:7]}-dirty" if sha else "unknown",
        "git_dirty_checked": False,
        "packages": versions,
    }
