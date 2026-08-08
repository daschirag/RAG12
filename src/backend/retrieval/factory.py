"""Factory that selects local vs Azure vector store from BACKEND_MODE."""

from __future__ import annotations

import logging
import os

from .base import VectorStore

logger = logging.getLogger(__name__)


def get_vector_store() -> VectorStore:
    """Return the configured ``VectorStore`` implementation.

    Reads ``BACKEND_MODE`` from the environment (default: ``local``).
    """
    mode = (os.getenv("BACKEND_MODE") or "local").strip().lower()

    if mode == "local":
        from .local_store import LocalVectorStore

        logger.info("Vector store backend: local (sentence-transformers + ChromaDB)")
        return LocalVectorStore()

    if mode == "azure":
        from .azure_store import AzureVectorStore

        logger.info("Vector store backend: azure (Azure OpenAI + Azure AI Search)")
        return AzureVectorStore()

    raise RuntimeError(
        f"Unknown BACKEND_MODE={mode!r}. Expected 'local' or 'azure'."
    )
