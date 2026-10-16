"""B3 - Keyword (BM25), hybrid (reciprocal rank fusion), reranked and routed retrieval.

All retrievers work on in-memory chunk records in the ingestion format
(``{"chunk_id", "text", "metadata", ...}``) and expose ``search(query, k)``
returning chunk records with a ``score``, best first. Ties are broken by
chunk order, so results are deterministic.

- ``BM25Index``: Okapi BM25 (k1 = 1.5, b = 0.75) over lower-cased word tokens
  with a small English stop-word list; implemented here (no extra dependency).
- ``HybridRetriever``: reciprocal rank fusion, score = sum_i w_i / (k_rrf + rank_i)
  over the top ``candidates`` of each input retriever (Cormack et al. 2009,
  k_rrf = 60).
- ``RerankRetriever``: rescores the top ``candidates`` of a base retriever
  with a cross-encoder (query, passage) scoring function.
- ``RoutedRetriever``: picks a retriever by query intent (conceptual vs
  quantitative).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

STOPWORDS = frozenset(
    "a an and are as at be by can could did do does for from had has have how i in is it its of on or "
    "our should that the their there these this to was we were what when where which who why will with "
    "you your".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _ranked(chunks: list[dict], scores: np.ndarray, k: int) -> list[dict]:
    order = np.lexsort((np.arange(len(scores)), -scores))[:k]   # score desc, then chunk order
    return [{**chunks[i], "score": float(scores[i])} for i in order]


class BM25Index:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("no chunks to index")
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avgdl = float(self.lengths.mean()) or 1.0
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.docs], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int = 5) -> list[dict]:
        return _ranked(self.chunks, self.scores(query), k)


class HybridRetriever:
    def __init__(self, retrievers: Sequence, weights: Sequence[float] | None = None,
                 k_rrf: int = 60, candidates: int = 50) -> None:
        self.retrievers = list(retrievers)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.retrievers)
        if len(self.weights) != len(self.retrievers):
            raise ValueError("one weight per retriever")
        self.k_rrf, self.candidates = k_rrf, candidates

    def search(self, query: str, k: int = 5) -> list[dict]:
        fused: dict[str, float] = {}
        records: dict[str, dict] = {}
        first_seen: dict[str, int] = {}
        for w, retriever in zip(self.weights, self.retrievers):
            for rank, hit in enumerate(retriever.search(query, self.candidates), start=1):
                cid = hit["chunk_id"]
                fused[cid] = fused.get(cid, 0.0) + w / (self.k_rrf + rank)
                records.setdefault(cid, {key: v for key, v in hit.items() if key != "score"})
                first_seen.setdefault(cid, len(first_seen))
        order = sorted(fused, key=lambda c: (-fused[c], first_seen[c]))[:k]
        return [{**records[c], "score": fused[c]} for c in order]


ScoreFn = Callable[[str, Sequence[str]], np.ndarray]    # (query, passages) -> relevance scores


class RerankRetriever:
    def __init__(self, base, score_fn: ScoreFn, candidates: int = 20) -> None:
        self.base, self.score_fn, self.candidates = base, score_fn, candidates

    def search(self, query: str, k: int = 5) -> list[dict]:
        pool = self.base.search(query, self.candidates)
        if not pool:
            return []
        scores = np.asarray(self.score_fn(query, [h["text"] for h in pool]), dtype=float)
        return _ranked(pool, scores, k)


class RoutedRetriever:
    def __init__(self, classify: Callable[[str], str], routes: dict) -> None:
        self.classify, self.routes = classify, routes

    def search(self, query: str, k: int = 5) -> list[dict]:
        return self.routes[self.classify(query)].search(query, k)


class SiblingExpander:
    """Adds the adjacent chunks (chunk_index +/- window) of each retrieved chunk's document.

    Retrieval-time fix for documents split across chunks (e.g. a fault record whose event header and
    logged cause fall into different chunks): the retrieved hits are kept in order, and each hit's
    siblings follow it. Siblings inherit the hit's score and carry ``metadata["expanded_from"]``.
    ``budget`` caps the total number of chunks (None = no cap: all hits plus their siblings).
    """

    def __init__(self, chunks: list[dict], window: int = 1, budget: int | None = None) -> None:
        self.window, self.budget = window, budget
        self.by_pos = {(c["metadata"]["doc_id"], c["metadata"]["chunk_index"]): c for c in chunks}

    def expand(self, hits: list[dict]) -> list[dict]:
        out, seen = [], set()

        def add(item):
            if item["chunk_id"] not in seen and (self.budget is None or len(out) < self.budget):
                seen.add(item["chunk_id"])
                out.append(item)

        for h in hits:
            add(h)
            doc, idx = h["metadata"]["doc_id"], h["metadata"].get("chunk_index")
            if idx is None:
                continue
            for offset in [o for d in range(1, self.window + 1) for o in (-d, d)]:
                sib = self.by_pos.get((doc, idx + offset))
                if sib is not None:
                    add({**sib, "metadata": {**sib["metadata"], "expanded_from": h["chunk_id"]},
                         "score": h.get("score", 0.0)})
        return out


class ExpandingRetriever:
    """Wraps a retriever: search(q, k) returns the base top-k hits with their siblings added."""

    def __init__(self, base, expander: SiblingExpander) -> None:
        self.base, self.expander = base, expander

    def search(self, query: str, k: int = 5) -> list[dict]:
        return self.expander.expand(self.base.search(query, k))


def cross_encoder_scorer(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> ScoreFn:
    """Cross-encoder relevance scores (downloaded once into the project cache)."""
    from ..runtime_env import configure_cache_dirs

    configure_cache_dirs()
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)

    def score(query: str, passages: Sequence[str]) -> np.ndarray:
        return np.asarray(model.predict([(query, p) for p in passages], batch_size=32, show_progress_bar=False))

    return score
