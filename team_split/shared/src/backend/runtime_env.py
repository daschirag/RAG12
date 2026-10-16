"""Process-wide runtime settings shared by the API, experiments and tests.

Model caches are pinned under ``GREENRAG_CACHE_DIR`` (default ``D:\\Rag\\cache``)
so Hugging Face / torch downloads never land in another project's cache or on
the system drive. Call ``configure_cache_dirs()`` before importing
sentence-transformers, transformers or torch hub.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CACHE_DIR = Path(r"D:\Rag\cache")


def configure_cache_dirs() -> Path:
    """Point HF_HOME and TORCH_HOME at the project cache root and return it."""
    root = Path(os.getenv("GREENRAG_CACHE_DIR") or DEFAULT_CACHE_DIR)
    hf_home = root / "huggingface"
    torch_home = root / "torch"
    hf_home.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TORCH_HOME"] = str(torch_home)
    return root
