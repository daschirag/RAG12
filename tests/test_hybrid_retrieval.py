"""B3: query classifier, BM25, rank fusion, reranking, routing (no model downloads)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.backend.kb.build import build_documents
from src.backend.kb.chunking import metadata_chunks
from src.backend.kb.queries import build_queries
from src.backend.kb.retrieval_eval import CosineIndex, evaluate
from src.backend.retrieval.hybrid_retriever import (
    BM25Index,
    HybridRetriever,
    RerankRetriever,
    RoutedRetriever,
    tokenize,
)
from src.backend.retrieval.query_classifier import classify_intent
from tests.test_kb_chunking import hash_embed


def _chunks(texts):
    return [{"chunk_id": f"c{i}", "text": t, "metadata": {"doc_id": f"d{i}"}} for i, t in enumerate(texts)]


class FixedRetriever:
    """Returns a fixed ranking (for testing fusion)."""

    def __init__(self, chunks, order):
        self.chunks, self.order = chunks, order

    def search(self, query, k=5):
        return [{**self.chunks[i], "score": 1.0 / (r + 1)} for r, i in enumerate(self.order[:k])]


# ---------------------------------------------------------------- classifier

@pytest.mark.parametrize("query, expected", [
    ("How many gearbox failures were recorded in Wind Farm A?", "quantitative"),
    ("What was the highest AC power on 15 May?", "quantitative"),
    ("What is the average wind speed in December?", "quantitative"),
    ("How long did the transformer event last?", "quantitative"),
    ("What was the temperature of the main bearing before the failure?", "quantitative"),
    ("Why does the pitch battery voltage drop?", "conceptual"),
    ("What should the technician check when the hydraulic pump runs too often?", "conceptual"),
    ("Which standard covers PV performance monitoring?", "conceptual"),
    ("Explain the converter fault on asset 38.", "conceptual"),
    ("", "conceptual"),
])
def test_classifier_independent_examples(query, expected):
    assert classify_intent(query) == expected


def test_classifier_on_labelled_set():
    queries = build_queries(build_documents())
    wrong = [q.query_id for q in queries if classify_intent(q.text) != q.query_class]
    assert wrong == []


# ---------------------------------------------------------------------- BM25

def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the Pitch-axis battery voltage?") == ["pitch", "axis", "battery", "voltage"]


def test_bm25_matches_hand_computed_scores():
    chunks = _chunks(["gearbox oil temperature", "gearbox gearbox bearing", "pitch battery"])
    idx = BM25Index(chunks, k1=1.5, b=0.75)
    n, avgdl = 3, (3 + 3 + 2) / 3
    idf_gear = math.log(1 + (n - 2 + 0.5) / (2 + 0.5))

    def term(tf, dl):
        return idf_gear * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * dl / avgdl))

    s = idx.scores("gearbox")
    assert s[0] == pytest.approx(term(1, 3)) and s[1] == pytest.approx(term(2, 3)) and s[2] == 0
    assert [h["chunk_id"] for h in idx.search("gearbox", 2)] == ["c1", "c0"]
    assert idx.scores("unknownword").sum() == 0


def test_bm25_ties_break_by_chunk_order():
    idx = BM25Index(_chunks(["alpha beta", "alpha beta", "gamma"]))
    assert [h["chunk_id"] for h in idx.search("alpha", 3)] == ["c0", "c1", "c2"]


# --------------------------------------------------------------- fusion etc.

def test_rrf_scores_and_order():
    chunks = _chunks(["a", "b", "c", "d"])
    r1 = FixedRetriever(chunks, [0, 1, 2])
    r2 = FixedRetriever(chunks, [2, 0, 3])
    hits = HybridRetriever([r1, r2], k_rrf=60, candidates=3).search("q", 4)
    expected = {"c0": 1 / 61 + 1 / 62, "c2": 1 / 63 + 1 / 61, "c1": 1 / 62, "c3": 1 / 63}
    assert [h["chunk_id"] for h in hits] == ["c0", "c2", "c1", "c3"]
    for h in hits:
        assert h["score"] == pytest.approx(expected[h["chunk_id"]])
    weighted = HybridRetriever([r1, r2], weights=[0.0, 1.0], candidates=3).search("q", 1)
    assert weighted[0]["chunk_id"] == "c2"
    with pytest.raises(ValueError):
        HybridRetriever([r1, r2], weights=[1.0])


def test_reranker_reorders_only_the_candidate_pool():
    chunks = _chunks(["x", "y", "z", "w"])
    base = FixedRetriever(chunks, [0, 1, 2, 3])
    rer = RerankRetriever(base, lambda q, ps: np.array([{"x": 0, "y": 3, "z": 5, "w": 9}[p] for p in ps]),
                          candidates=3)
    assert [h["chunk_id"] for h in rer.search("q", 3)] == ["c2", "c1", "c0"]    # "w" not in the pool


def test_routed_retriever_dispatch():
    chunks = _chunks(["one", "two"])
    routes = {"quantitative": FixedRetriever(chunks, [1]), "conceptual": FixedRetriever(chunks, [0])}
    r = RoutedRetriever(classify_intent, routes)
    assert r.search("How many?", 1)[0]["chunk_id"] == "c1"
    assert r.search("Why?", 1)[0]["chunk_id"] == "c0"


def _sib_chunks():
    return [{"chunk_id": f"{d}::{i}", "text": f"{d} part {i}", "metadata": {"doc_id": d, "chunk_index": i}}
            for d, n in (("a", 3), ("b", 1), ("c", 2)) for i in range(n)]


def test_sibling_expander_adds_adjacent_chunks_after_each_hit():
    from src.backend.retrieval.hybrid_retriever import SiblingExpander

    chunks = _sib_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    hits = [{**by_id["a::1"], "score": 3.0}, {**by_id["b::0"], "score": 2.0}, {**by_id["c::0"], "score": 1.0}]
    out = SiblingExpander(chunks).expand(hits)
    assert [h["chunk_id"] for h in out] == ["a::1", "a::0", "a::2", "b::0", "c::0", "c::1"]
    assert out[1]["metadata"]["expanded_from"] == "a::1" and out[1]["score"] == 3.0
    assert "expanded_from" not in out[0]["metadata"]
    # The retrieved documents never change; only same-document text is added.
    assert {h["metadata"]["doc_id"] for h in out} == {h["metadata"]["doc_id"] for h in hits}
    capped = SiblingExpander(chunks, budget=3).expand(hits)
    assert [h["chunk_id"] for h in capped] == ["a::1", "a::0", "a::2"]


def test_sibling_expander_does_not_duplicate_retrieved_siblings():
    from src.backend.retrieval.hybrid_retriever import SiblingExpander

    chunks = _sib_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}
    out = SiblingExpander(chunks).expand([{**by_id["a::0"], "score": 2.0}, {**by_id["a::1"], "score": 1.0}])
    assert [h["chunk_id"] for h in out] == ["a::0", "a::1", "a::2"]


def test_all_variants_run_on_the_knowledge_base_with_mocks():
    docs = build_documents()
    chunks = metadata_chunks(docs)
    queries = build_queries(docs)[:12]
    bm25, vec = BM25Index(chunks), CosineIndex(chunks, hash_embed)
    hybrid = HybridRetriever([bm25, vec])
    mock_ce = lambda q, ps: np.array([len(set(q.lower().split()) & set(p.lower().split())) for p in ps], float)
    rerank = RerankRetriever(hybrid, mock_ce)
    for retriever in (bm25, vec, hybrid, rerank):
        res = evaluate(retriever, queries)
        assert len(res) == 12 and res["ndcg_at_k"].between(0, 1).all()
        assert evaluate(retriever, queries).equals(res)                          # deterministic
    # BM25 finds the exact event document for an event query naming farm and asset.
    q = next(q for q in build_queries(docs) if q.query_type == "maintenance")
    top = [h["metadata"]["doc_id"] for h in bm25.search(q.text, 5)]
    assert set(top) & set(q.relevant)
