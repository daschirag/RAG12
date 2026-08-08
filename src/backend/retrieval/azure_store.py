"""Azure vector store: Azure OpenAI embeddings + Azure AI Search."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .base import VectorStore
from .metrics import (
    NUMERIC_METRIC_FIELDS,
    extract_metric_fields,
    validate_metric_request,
)

logger = logging.getLogger(__name__)

# text-embedding-3-small default dimensionality
EMBEDDING_DIMENSIONS = 1536

_REQUIRED_ENV = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_API_KEY",
    "AZURE_SEARCH_INDEX",
)


def _require_env(name: str) -> str:
    """Return a non-empty env var or raise a clear RuntimeError."""
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required Azure configuration: {name} is unset or empty. "
            f"Set it in your .env (see .env.example). "
            f"BACKEND_MODE=azure needs: {', '.join(_REQUIRED_ENV)}."
        )
    return value


def _sanitize_index_name(name: str) -> str:
    """Azure AI Search index names: lowercase letters, digits, dashes only."""
    cleaned = re.sub(r"[^a-z0-9-]", "-", name.lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or "greentrag"


class AzureVectorStore(VectorStore):
    """Production backend: Azure OpenAI embeddings + Azure AI Search indexes."""

    def __init__(self) -> None:
        # Validate all keys up front so failures are explicit, not SDK-cryptic.
        self.openai_endpoint = _require_env("AZURE_OPENAI_ENDPOINT")
        self.openai_api_key = _require_env("AZURE_OPENAI_API_KEY")
        self.openai_deployment = _require_env("AZURE_OPENAI_DEPLOYMENT")
        self.search_endpoint = _require_env("AZURE_SEARCH_ENDPOINT")
        self.search_api_key = _require_env("AZURE_SEARCH_API_KEY")
        self.search_index_prefix = _require_env("AZURE_SEARCH_INDEX")

        self._openai = None
        self._index_client = None

    def _get_openai(self):
        if self._openai is None:
            from openai import AzureOpenAI

            self._openai = AzureOpenAI(
                api_key=self.openai_api_key,
                api_version="2024-02-01",
                azure_endpoint=self.openai_endpoint,
            )
        return self._openai

    def _get_index_client(self):
        if self._index_client is None:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents.indexes import SearchIndexClient

            self._index_client = SearchIndexClient(
                endpoint=self.search_endpoint,
                credential=AzureKeyCredential(self.search_api_key),
            )
        return self._index_client

    def _index_name(self, domain_id: str) -> str:
        """One Search index per domain: ``{AZURE_SEARCH_INDEX}-{domain_id}``."""
        return _sanitize_index_name(f"{self.search_index_prefix}-{domain_id}")

    def _ensure_index(self, domain_id: str) -> str:
        """Create the domain index if missing; return its name."""
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        index_name = self._index_name(domain_id)
        client = self._get_index_client()
        existing = {idx.name for idx in client.list_indexes()}
        if index_name in existing:
            return index_name

        fields = [
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="text", type=SearchFieldDataType.String),
            SimpleField(name="metadata", type=SearchFieldDataType.String),
            SimpleField(name="domain_id", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=EMBEDDING_DIMENSIONS,
                vector_search_profile_name="greentrag-vector-profile",
            ),
        ]
        # Filterable / sortable Doubles for numeric top-k (search_by_metric).
        for metric_name in sorted(NUMERIC_METRIC_FIELDS):
            fields.append(
                SimpleField(
                    name=metric_name,
                    type=SearchFieldDataType.Double,
                    filterable=True,
                    sortable=True,
                )
            )
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="greentrag-hnsw")],
            profiles=[
                VectorSearchProfile(
                    name="greentrag-vector-profile",
                    algorithm_configuration_name="greentrag-hnsw",
                )
            ],
        )
        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
        client.create_index(index)
        logger.info("Created Azure AI Search index '%s'.", index_name)
        return index_name

    def _search_client(self, index_name: str):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        return SearchClient(
            endpoint=self.search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(self.search_api_key),
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via the configured Azure OpenAI deployment."""
        if not texts:
            return []
        client = self._get_openai()
        response = client.embeddings.create(
            model=self.openai_deployment,
            input=texts,
        )
        # Preserve input order.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    def embed_and_store(self, domain_id: str, chunks: list[dict[str, Any]]) -> None:
        """Embed and upsert all chunks into the domain Azure AI Search index.

        Uses ``merge_or_upload_documents`` so existing ``chunk_id`` keys are
        refreshed (text, metadata, metric fields, embedding) and new keys are
        inserted — no skip-if-exists path.
        """
        if not chunks:
            logger.info("No chunks to store for domain '%s'.", domain_id)
            return

        valid_chunks = [c for c in chunks if c.get("chunk_id")]
        if not valid_chunks:
            logger.info("No chunks with chunk_id for domain '%s'.", domain_id)
            return

        index_name = self._ensure_index(domain_id)
        search_client = self._search_client(index_name)

        # Batch embeddings to stay within API limits, then merge-or-upload all.
        batch_size = 16
        documents: list[dict[str, Any]] = []
        for start in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[start : start + batch_size]
            texts = [str(c.get("text", "")) for c in batch]
            vectors = self._embed_texts(texts)
            for chunk, vector in zip(batch, vectors, strict=True):
                meta = chunk.get("metadata") or {}
                doc: dict[str, Any] = {
                    "chunk_id": str(chunk["chunk_id"]),
                    "text": str(chunk.get("text", "")),
                    "metadata": json.dumps(meta, ensure_ascii=False),
                    "domain_id": domain_id,
                    "embedding": vector,
                }
                # Promote numeric metrics to top-level filterable/sortable fields.
                doc.update(extract_metric_fields(meta if isinstance(meta, dict) else {}))
                documents.append(doc)

        search_client.merge_or_upload_documents(documents=documents)
        logger.info(
            "Upserted %d chunk(s) into Azure index '%s'.",
            len(documents),
            index_name,
        )

    def search(
        self,
        domain_id: str,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Vector-search the domain index for ``query``."""
        from azure.search.documents.models import VectorizedQuery

        index_name = self._ensure_index(domain_id)
        search_client = self._search_client(index_name)
        query_vector = self._embed_texts([query])[0]

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["chunk_id", "text", "metadata", "domain_id"],
            top=top_k,
        )

        hits: list[dict[str, Any]] = []
        for doc in results:
            raw_meta = doc.get("metadata") or "{}"
            try:
                metadata = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
            except json.JSONDecodeError:
                metadata = {"raw": raw_meta}
            hits.append(
                {
                    "chunk_id": doc.get("chunk_id"),
                    "domain_id": doc.get("domain_id", domain_id),
                    "text": doc.get("text"),
                    "metadata": metadata,
                    "score": float(doc.get("@search.score") or 0.0),
                }
            )
        return hits

    def search_by_metric(
        self,
        domain_id: str,
        metric_field: str,
        top_k: int,
        order: str = "desc",
    ) -> list[dict[str, Any]]:
        """Rank chunks by a numeric Azure Search field (no embedding search)."""
        metric_field, order = validate_metric_request(metric_field, order)
        index_name = self._ensure_index(domain_id)
        search_client = self._search_client(index_name)

        select_fields = [
            "chunk_id",
            "text",
            "metadata",
            "domain_id",
            metric_field,
        ]
        results = search_client.search(
            search_text="*",
            filter=f"{metric_field} ne null",
            order_by=[f"{metric_field} {order}"],
            select=select_fields,
            top=top_k,
        )

        hits: list[dict[str, Any]] = []
        for doc in results:
            raw_meta = doc.get("metadata") or "{}"
            try:
                metadata = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
            except json.JSONDecodeError:
                metadata = {"raw": raw_meta}
            if not isinstance(metadata, dict):
                metadata = {}
            # Ensure the ranking metric is visible in metadata for clients.
            if metric_field in doc and doc[metric_field] is not None:
                metadata[metric_field] = float(doc[metric_field])
            value = float(doc.get(metric_field) or 0.0)
            hits.append(
                {
                    "chunk_id": doc.get("chunk_id"),
                    "domain_id": doc.get("domain_id", domain_id),
                    "text": doc.get("text"),
                    "metadata": metadata,
                    "score": value,
                }
            )
        return hits
