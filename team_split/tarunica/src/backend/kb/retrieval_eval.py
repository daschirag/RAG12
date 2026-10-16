"""B2 - Retrieval evaluation against document-level relevance.

Retrieval returns a ranked list of chunks; relevance is defined per document
(``LabelledQuery.relevant``: doc_id -> grade). Definitions, for the top k
chunks (k = 5):

- Precision@k: distinct relevant documents among the top-k chunks, divided by
  k. Several chunks of the same document count once, so redundant results
  are not rewarded.
- Recall@k: distinct relevant documents found in the top-k chunks, divided by
  min(k, number of relevant documents).
- MRR: 1 / rank of the first chunk that belongs to a relevant document (0 if
  none in the top k).
- nDCG@k: graded gains (2 or 1), counted for a document only at its first
  appearance, discounted by log2(rank + 1), normalised by the ideal ordering.

The index is an in-memory cosine-similarity matrix so the comparison does not
depend on a vector database; ties are broken by chunk order (deterministic).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .queries import LabelledQuery

EmbedFn = Callable[[Sequence[str]], np.ndarray]
K = 5


def metrics_at_k(retrieved_doc_ids: Sequence[str], relevant: dict[str, int], k: int = K) -> dict[str, float]:
    top = list(retrieved_doc_ids)[:k]
    seen: set[str] = set()
    hits, dcg, first = 0, 0.0, 0.0
    for rank, doc_id in enumerate(top, start=1):
        if doc_id in relevant and doc_id not in seen:
            hits += 1
            dcg += relevant[doc_id] / math.log2(rank + 1)
            if first == 0.0:
                first = 1.0 / rank
        seen.add(doc_id)
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(r + 1) for r, g in enumerate(ideal, start=1))
    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / min(k, len(relevant)) if relevant else 0.0,
        "mrr": first,
        "ndcg_at_k": dcg / idcg if idcg > 0 else 0.0,
    }


class CosineIndex:
    """Exact cosine search over chunk embeddings."""

    def __init__(self, chunks: list[dict], embed: EmbedFn) -> None:
        if not chunks:
            raise ValueError("no chunks to index")
        self.chunks = chunks
        self.embed = embed
        self.matrix = self._normalise(np.asarray(embed([c["text"] for c in chunks]), dtype=float))

    @staticmethod
    def _normalise(m: np.ndarray) -> np.ndarray:
        return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)

    def search(self, query: str, k: int = K) -> list[dict]:
        q = self._normalise(np.asarray(self.embed([query]), dtype=float))[0]
        scores = self.matrix @ q
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]     # score desc, then chunk order
        return [{**self.chunks[i], "score": float(scores[i])} for i in order]


def evaluate(index: CosineIndex, queries: list[LabelledQuery], k: int = K) -> pd.DataFrame:
    """Per-query metrics; ``retrieved`` keeps the ranked doc_ids for inspection."""
    rows = []
    for q in queries:
        hits = index.search(q.text, k)
        doc_ids = [h["metadata"]["doc_id"] for h in hits]
        rows.append({"query_id": q.query_id, "query_type": q.query_type, "query_class": q.query_class,
                     **metrics_at_k(doc_ids, q.relevant, k), "retrieved": "|".join(doc_ids)})
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: Sequence[float], n_resamples: int = 10_000, seed: int = 0,
                      ci: float = 0.95) -> tuple[float, float, float]:
    """Mean and percentile CI with queries resampled with replacement."""
    v = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n_resamples, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(v.mean()), float(lo), float(hi)


def minilm_embed() -> EmbedFn:
    """all-MiniLM-L6-v2 (the model the local vector store uses), loaded lazily from the project cache."""
    from ..runtime_env import configure_cache_dirs

    configure_cache_dirs()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts: Sequence[str]) -> np.ndarray:
        return model.encode(list(texts), batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    return embed
