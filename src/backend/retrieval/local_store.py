"""Local vector store: sentence-transformers + ChromaDB (no cloud account)."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from .base import VectorStore
from .metrics import flatten_chunk_metadata, validate_metric_request

logger = logging.getLogger(__name__)

# Project root: src/backend/retrieval/local_store.py → ../../../
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# One PersistentClient per on-disk path — avoid concurrent handle conflicts.
_chroma_clients: dict[str, Any] = {}
_chroma_lock = threading.Lock()
_embedding_models: dict[str, Any] = {}
_model_lock = threading.Lock()

T = TypeVar("T")


def _get_persistent_client(persist_dir: Path) -> Any:
    """Return a process-wide Chroma PersistentClient for ``persist_dir``."""
    key = str(persist_dir.resolve())
    with _chroma_lock:
        client = _chroma_clients.get(key)
        if client is None:
            import chromadb

            persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=key)
            _chroma_clients[key] = client
            logger.info("Opened shared Chroma PersistentClient at %s", key)
        return client


def _get_embedding_model(model_name: str) -> Any:
    """Return a process-wide sentence-transformers model instance."""
    with _model_lock:
        model = _embedding_models.get(model_name)
        if model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading local embedding model: %s", model_name)
            model = SentenceTransformer(model_name)
            _embedding_models[model_name] = model
        return model


def _is_chroma_disk_miss(exc: BaseException) -> bool:
    """True for Chroma's transient concurrent-access disk miss."""
    return "Nothing found on disk" in str(exc)


def _with_chroma_retry(operation: Callable[[], T], *, label: str) -> T:
    """Run ``operation`` with one brief retry on Chroma disk-miss errors."""
    last_exc: BaseException | None = None
    for attempt in range(2):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            if not _is_chroma_disk_miss(exc):
                raise
            last_exc = exc
            if attempt == 0:
                logger.warning(
                    "Chroma transient disk miss during %s; retrying once…",
                    label,
                )
                time.sleep(0.2)
                continue
            break

    raise RuntimeError(
        f"ChromaDB failed during {label} after retry "
        f"('Nothing found on disk' — often caused by concurrent access or a "
        f"corrupted ./chroma_db). Delete ./chroma_db and re-run POST /ingest. "
        f"Original error: {last_exc}"
    ) from last_exc


class LocalVectorStore(VectorStore):
    """Embed with MiniLM and persist vectors in a local ChromaDB folder."""

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else DEFAULT_CHROMA_DIR
        self.model_name = model_name

    def _get_model(self):
        return _get_embedding_model(self.model_name)

    def _get_client(self):
        return _get_persistent_client(self.persist_dir)

    def _collection(self, domain_id: str):
        """Return (or create) one Chroma collection per domain."""
        return self._get_client().get_or_create_collection(
            name=domain_id,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_and_store(self, domain_id: str, chunks: list[dict[str, Any]]) -> None:
        """Embed and upsert chunks into the domain collection.

        - Existing ``chunk_id``s → ``collection.update`` (documents, metadatas,
          embeddings refreshed from the current chunk payload).
        - New ``chunk_id``s → ``collection.add`` after embedding.
        """
        if not chunks:
            logger.info("No chunks to store for domain '%s'.", domain_id)
            return

        valid_chunks = [c for c in chunks if c.get("chunk_id")]
        if not valid_chunks:
            logger.info("No chunks with chunk_id for domain '%s'.", domain_id)
            return

        collection = self._collection(domain_id)

        candidate_ids = [str(c["chunk_id"]) for c in valid_chunks]
        existing: set[str] = set()
        try:
            stored = collection.get(ids=candidate_ids, include=[])
            existing = set(stored.get("ids") or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not look up existing Chroma IDs for '%s': %s",
                domain_id,
                exc,
            )

        update_chunks = [
            c for c in valid_chunks if str(c["chunk_id"]) in existing
        ]
        new_chunks = [
            c for c in valid_chunks if str(c["chunk_id"]) not in existing
        ]

        model = self._get_model()

        if update_chunks:
            update_ids = [str(c["chunk_id"]) for c in update_chunks]
            update_texts = [str(c.get("text", "")) for c in update_chunks]
            update_metas = [
                flatten_chunk_metadata(domain_id, c.get("metadata") or {})
                for c in update_chunks
            ]
            update_embeddings = model.encode(
                update_texts, show_progress_bar=False
            ).tolist()
            collection.update(
                ids=update_ids,
                documents=update_texts,
                embeddings=update_embeddings,
                metadatas=update_metas,
            )
            logger.info(
                "Updated %d existing chunk(s) in local collection '%s'.",
                len(update_chunks),
                domain_id,
            )

        if new_chunks:
            new_ids = [str(c["chunk_id"]) for c in new_chunks]
            new_texts = [str(c.get("text", "")) for c in new_chunks]
            new_metas = [
                flatten_chunk_metadata(domain_id, c.get("metadata") or {})
                for c in new_chunks
            ]
            new_embeddings = model.encode(
                new_texts, show_progress_bar=False
            ).tolist()
            collection.add(
                ids=new_ids,
                documents=new_texts,
                embeddings=new_embeddings,
                metadatas=new_metas,
            )
            logger.info(
                "Added %d new chunk(s) to local collection '%s'.",
                len(new_chunks),
                domain_id,
            )

        if not update_chunks and not new_chunks:
            logger.info("Nothing to upsert for domain '%s'.", domain_id)

    def search(
        self,
        domain_id: str,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Embed the query and return the nearest chunks with similarity scores."""

        def _run() -> list[dict[str, Any]]:
            collection = self._collection(domain_id)
            if collection.count() == 0:
                return []

            model = self._get_model()
            query_embedding = model.encode([query], show_progress_bar=False).tolist()

            result = collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            ids = (result.get("ids") or [[]])[0]
            documents = (result.get("documents") or [[]])[0]
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]

            hits: list[dict[str, Any]] = []
            for chunk_id, text, meta, distance in zip(
                ids, documents, metadatas, distances, strict=False
            ):
                # Cosine distance → similarity in [0, 1] (approx).
                score = 1.0 - float(distance) if distance is not None else 0.0
                hits.append(
                    {
                        "chunk_id": chunk_id,
                        "domain_id": domain_id,
                        "text": text,
                        "metadata": meta or {},
                        "score": score,
                    }
                )
            return hits

        return _with_chroma_retry(_run, label=f"search({domain_id!r})")

    def search_by_metric(
        self,
        domain_id: str,
        metric_field: str,
        top_k: int,
        order: str = "desc",
    ) -> list[dict[str, Any]]:
        """Rank chunks by a numeric metadata field (no embedding search)."""
        metric_field, order = validate_metric_request(metric_field, order)

        def _run() -> list[dict[str, Any]]:
            collection = self._collection(domain_id)
            if collection.count() == 0:
                return []

            stored = collection.get(include=["documents", "metadatas"])
            ids = stored.get("ids") or []
            documents = stored.get("documents") or []
            metadatas = stored.get("metadatas") or []

            ranked: list[tuple[float, dict[str, Any]]] = []
            for chunk_id, text, meta in zip(ids, documents, metadatas, strict=False):
                meta = meta or {}
                raw = meta.get(metric_field)
                if raw is None:
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                ranked.append(
                    (
                        value,
                        {
                            "chunk_id": chunk_id,
                            "domain_id": domain_id,
                            "text": text,
                            "metadata": meta,
                            "score": value,
                        },
                    )
                )

            reverse = order == "desc"
            ranked.sort(key=lambda item: item[0], reverse=reverse)
            return [item[1] for item in ranked[:top_k]]

        return _with_chroma_retry(
            _run, label=f"search_by_metric({domain_id!r}, {metric_field!r})"
        )
