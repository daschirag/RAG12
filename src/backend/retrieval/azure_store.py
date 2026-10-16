"""Azure vector store: Azure AI Search + embeddings from local MiniLM (plan A) or Azure OpenAI.

Plan A (deployed 2026-10-16): AI Search Free holds the vectors; embeddings are computed locally with the same
all-MiniLM-L6-v2 model as the local store, so local and Azure retrieval differ only in the index engine.
Azure OpenAI embeddings are used only when EMBEDDING_PROVIDER resolves to ``azure_openai``.

Index design: ONE index (``AZURE_SEARCH_INDEX``) for all domains, filtered by ``domain_id``. The Free tier allows
3 indexes and there are 4 domains. Document keys are URL-safe base64 of ``domain|chunk_id`` because AI Search keys
may not contain ':' (knowledge-base chunk ids do); the original chunk id is kept in its own field.

The Search admin key comes from AZURE_SEARCH_API_KEY or Key Vault (``search-admin-key``), never from a file.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any, Callable

from ..settings import get_settings
from .base import VectorStore
from .metrics import (
    NUMERIC_METRIC_FIELDS,
    extract_metric_fields,
    validate_metric_request,
)

logger = logging.getLogger(__name__)

OPENAI_DIMENSIONS = 1536            # text-embedding-3-small
LOCAL_DIMENSIONS = 384              # all-MiniLM-L6-v2
LOCAL_MODEL = "all-MiniLM-L6-v2"

_OPENAI_REQUIRED = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT")


def _require_env(name: str) -> str:
    """Return a non-empty env var or raise a clear RuntimeError."""
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required Azure configuration: {name} is unset or empty. "
            f"Set it in your .env (see .env.example).")
    return value


def _sanitize_index_name(name: str) -> str:
    """Azure AI Search index names: lowercase letters, digits, dashes only."""
    cleaned = re.sub(r"[^a-z0-9-]", "-", name.lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or "greenrag"


def doc_key(domain_id: str, chunk_id: str) -> str:
    """AI Search document key (letters, digits, '-', '_', '=' only)."""
    return base64.urlsafe_b64encode(f"{domain_id}|{chunk_id}".encode("utf-8")).decode("ascii")


def _odata(value: str) -> str:
    return value.replace("'", "''")


class AzureVectorStore(VectorStore):
    """BACKEND_MODE=azure vector store. ``embed`` (texts -> vectors) and ``search_client`` are injectable for tests."""

    def __init__(self, embed: Callable[[list[str]], list[list[float]]] | None = None) -> None:
        s = get_settings()
        if not s.azure_search_endpoint:
            raise RuntimeError("Missing required Azure configuration: AZURE_SEARCH_ENDPOINT is unset or empty. "
                               "Set it in your .env (see .env.example).")
        if not s.azure_search_index:
            raise RuntimeError("Missing required Azure configuration: AZURE_SEARCH_INDEX is unset or empty.")
        self.search_endpoint = s.azure_search_endpoint
        self.index_name = _sanitize_index_name(s.azure_search_index)
        self.embedding_provider = s.embeddings
        if self.embedding_provider == "azure_openai":
            self.openai_endpoint = _require_env("AZURE_OPENAI_ENDPOINT")
            self.openai_api_key = _require_env("AZURE_OPENAI_API_KEY")
            self.openai_deployment = _require_env("AZURE_OPENAI_DEPLOYMENT")
            self.dimensions = OPENAI_DIMENSIONS
        else:
            self.dimensions = LOCAL_DIMENSIONS
        self._embed_override = embed
        self._openai = None
        self._index_client = None
        self._client = None
        self._index_ready = False

    # -- clients ---------------------------------------------------------------
    def _credential(self):
        from azure.core.credentials import AzureKeyCredential

        from ..cloud.secrets import get_secret

        return AzureKeyCredential(get_secret("search_api_key"))

    def _get_index_client(self):
        if self._index_client is None:
            from azure.search.documents.indexes import SearchIndexClient

            self._index_client = SearchIndexClient(endpoint=self.search_endpoint, credential=self._credential())
        return self._index_client

    def _search_client(self):
        if self._client is None:
            from azure.search.documents import SearchClient

            self._client = SearchClient(endpoint=self.search_endpoint, index_name=self.index_name,
                                        credential=self._credential())
        return self._client

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._embed_override is not None:
            return [list(map(float, v)) for v in self._embed_override(texts)]
        if self.embedding_provider == "local":
            from .local_store import _get_embedding_model

            return _get_embedding_model(LOCAL_MODEL).encode(texts, show_progress_bar=False).tolist()
        if self._openai is None:
            from openai import AzureOpenAI

            self._openai = AzureOpenAI(api_key=self.openai_api_key, api_version="2024-02-01",
                                       azure_endpoint=self.openai_endpoint)
        response = self._openai.embeddings.create(model=self.openai_deployment, input=texts)
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]

    # -- index -----------------------------------------------------------------
    def index_fields(self) -> list:
        from azure.search.documents.indexes.models import (
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SimpleField,
        )

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="chunk_id", type=SearchFieldDataType.String),
            SearchableField(name="text", type=SearchFieldDataType.String),
            SimpleField(name="metadata", type=SearchFieldDataType.String),
            SimpleField(name="domain_id", type=SearchFieldDataType.String, filterable=True),
            SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable=True, vector_search_dimensions=self.dimensions,
                        vector_search_profile_name="greenrag-vector-profile"),
        ]
        for metric_name in sorted(NUMERIC_METRIC_FIELDS):
            fields.append(SimpleField(name=metric_name, type=SearchFieldDataType.Double, filterable=True,
                                      sortable=True))
        return fields

    def _ensure_index(self) -> str:
        if self._index_ready:
            return self.index_name
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchIndex,
            VectorSearch,
            VectorSearchProfile,
        )

        client = self._get_index_client()
        if self.index_name not in {idx.name for idx in client.list_indexes()}:
            vector_search = VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="greenrag-hnsw")],
                profiles=[VectorSearchProfile(name="greenrag-vector-profile",
                                              algorithm_configuration_name="greenrag-hnsw")])
            client.create_index(SearchIndex(name=self.index_name, fields=self.index_fields(),
                                            vector_search=vector_search))
            logger.info("Created Azure AI Search index '%s' (%d-dim vectors).", self.index_name, self.dimensions)
        self._index_ready = True
        return self.index_name

    # -- VectorStore API -------------------------------------------------------
    def embed_and_store(self, domain_id: str, chunks: list[dict[str, Any]]) -> None:
        """Embed and merge-or-upload chunks into the shared index (existing keys are refreshed)."""
        valid = [c for c in chunks if c.get("chunk_id")]
        if not valid:
            logger.info("No chunks with chunk_id for domain '%s'.", domain_id)
            return
        self._ensure_index()
        client = self._search_client()
        batch_size = 64
        stored = 0
        for start in range(0, len(valid), batch_size):
            batch = valid[start:start + batch_size]
            vectors = self._embed_texts([str(c.get("text", "")) for c in batch])
            docs = []
            for chunk, vector in zip(batch, vectors, strict=True):
                meta = chunk.get("metadata") or {}
                doc: dict[str, Any] = {
                    "id": doc_key(domain_id, str(chunk["chunk_id"])), "chunk_id": str(chunk["chunk_id"]),
                    "text": str(chunk.get("text", "")), "metadata": json.dumps(meta, ensure_ascii=False),
                    "domain_id": domain_id, "embedding": vector}
                doc.update(extract_metric_fields(meta if isinstance(meta, dict) else {}))
                docs.append(doc)
            client.merge_or_upload_documents(documents=docs)
            stored += len(docs)
        logger.info("Upserted %d chunk(s) of '%s' into Azure index '%s'.", stored, domain_id, self.index_name)

    @staticmethod
    def _hit(doc: dict, domain_id: str, score: float) -> dict[str, Any]:
        raw = doc.get("metadata") or "{}"
        try:
            metadata = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            metadata = {"raw": raw}
        return {"chunk_id": doc.get("chunk_id"), "domain_id": doc.get("domain_id", domain_id),
                "text": doc.get("text"), "metadata": metadata if isinstance(metadata, dict) else {}, "score": score}

    def search(self, domain_id: str, query: str, top_k: int) -> list[dict[str, Any]]:
        """Vector search restricted to the domain."""
        from azure.search.documents.models import VectorizedQuery

        self._ensure_index()
        vq = VectorizedQuery(vector=self._embed_texts([query])[0], k_nearest_neighbors=top_k, fields="embedding")
        results = self._search_client().search(
            search_text=None, vector_queries=[vq], filter=f"domain_id eq '{_odata(domain_id)}'",
            select=["chunk_id", "text", "metadata", "domain_id"], top=top_k)
        return [self._hit(doc, domain_id, float(doc.get("@search.score") or 0.0)) for doc in results]

    def search_by_metric(self, domain_id: str, metric_field: str, top_k: int, order: str = "desc") -> list[dict[str, Any]]:
        """Rank the domain's chunks by a numeric field (no embedding search)."""
        metric_field, order = validate_metric_request(metric_field, order)
        self._ensure_index()
        results = self._search_client().search(
            search_text="*", filter=f"domain_id eq '{_odata(domain_id)}' and {metric_field} ne null",
            order_by=[f"{metric_field} {order}"],
            select=["chunk_id", "text", "metadata", "domain_id", metric_field], top=top_k)
        hits = []
        for doc in results:
            hit = self._hit(doc, domain_id, float(doc.get(metric_field) or 0.0))
            if doc.get(metric_field) is not None:
                hit["metadata"][metric_field] = float(doc[metric_field])
            hits.append(hit)
        return hits
