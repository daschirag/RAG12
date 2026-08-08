"""Abstract vector-store interface shared by local and Azure backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """Pluggable embedding + retrieval backend for GreenRAG chunks."""

    @abstractmethod
    def embed_and_store(self, domain_id: str, chunks: list[dict[str, Any]]) -> None:
        """Embed chunk texts and upsert them into the domain collection/index.

        Each chunk dict follows the ingestion JSONL shape::
            {"chunk_id": str, "text": str, "metadata": dict, ...}
        """

    @abstractmethod
    def search(
        self,
        domain_id: str,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return the top_k nearest chunks for ``query`` within ``domain_id``.

        Each result includes the original chunk fields plus a numeric ``score``
        (higher is more similar, when the backend provides that semantics).
        """

    @abstractmethod
    def search_by_metric(
        self,
        domain_id: str,
        metric_field: str,
        top_k: int,
        order: str = "desc",
    ) -> list[dict[str, Any]]:
        """Return top_k chunks ranked by a numeric metadata field (no embedding).

        ``order`` is ``desc`` (highest first) or ``asc`` (lowest first).
        Each hit's ``score`` is the metric value used for ranking.
        """
