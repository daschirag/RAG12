"""B2 setup: chunkers, labelled query set and retrieval metrics (mocked embeddings, no model download)."""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np
import pytest

from src.backend.kb.build import build_documents, chunk_documents
from src.backend.kb.chunking import (
    fixed_size_chunks,
    fixed_windows,
    metadata_chunks,
    metadata_header,
    pack_paragraphs,
    semantic_boundaries,
    semantic_chunks,
    split_units,
)
from src.backend.kb.queries import build_queries
from src.backend.kb.retrieval_eval import CosineIndex, bootstrap_mean_ci, evaluate, metrics_at_k
from src.backend.retrieval.metrics import flatten_chunk_metadata


def hash_embed(texts, dim: int = 256) -> np.ndarray:
    """Deterministic bag-of-words hashing embedding (stands in for MiniLM in unit tests)."""
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    return out


@pytest.fixture(scope="module")
def docs():
    return build_documents()


@pytest.fixture(scope="module")
def queries(docs):
    return build_queries(docs)


# ------------------------------------------------------------------ chunkers

def test_fixed_windows_cover_text_without_splitting_words():
    text = " ".join(f"word{i}" for i in range(400))
    windows = fixed_windows(text, size=200, overlap=40)
    assert all(len(w) <= 200 for w in windows)
    assert all(re.fullmatch(r"(word\d+ ?)+", w) for w in windows)          # no word cut in half
    assert windows[0].startswith("word0") and windows[-1].endswith("word399")
    covered = set(" ".join(windows).split())
    assert covered == set(text.split())                                    # nothing lost
    with pytest.raises(ValueError):
        fixed_windows(text, size=100, overlap=100)


def test_pack_paragraphs_matches_b1_chunks(docs):
    b1 = chunk_documents(docs)
    rebuilt = [p for d in docs for p in pack_paragraphs(d.text, 1200)]
    assert [c["text"].split("\n", 1)[1] for c in b1] == rebuilt


def test_split_units_and_semantic_boundaries():
    text = "Pumps fail. Oil leaks appear.\nBattery voltage low. Encoder faulty."
    assert split_units(text) == ["Pumps fail.", "Oil leaks appear.", "Battery voltage low.", "Encoder faulty."]
    emb = np.array([[1, 0], [1, 0.05], [0, 1], [0.05, 1]], dtype=float)
    assert semantic_boundaries(emb) == [2]                                 # topic changes before unit 2
    assert semantic_boundaries(emb[:2]) == []


@pytest.mark.parametrize("name", ["fixed_size", "semantic", "metadata_aware"])
def test_chunkers_respect_documents_and_format(docs, name):
    chunks = {"fixed_size": lambda: fixed_size_chunks(docs),
              "semantic": lambda: semantic_chunks(docs, hash_embed),
              "metadata_aware": lambda: metadata_chunks(docs)}[name]()
    by_doc = {d.doc_id: d for d in docs}
    assert {c["metadata"]["doc_id"] for c in chunks} == set(by_doc)        # every document chunked
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        m = c["metadata"]
        assert m["chunking"] == name and c["domain_id"] == "knowledge_base"
        assert m["synthetic"] == by_doc[m["doc_id"]].synthetic
        body = c["text"].split("\n", 2)[2] if name == "metadata_aware" else c["text"]
        # Every chunk's content comes from its own document (no cross-document chunks).
        assert all(line.strip() in by_doc[m["doc_id"]].text for line in body.split("\n") if line.strip())
        flat = flatten_chunk_metadata("knowledge_base", m)
        assert all(isinstance(v, (str, int, float, bool)) for v in flat.values())


def test_metadata_header_uses_kb_metadata(docs):
    d = next(x for x in docs if x.doc_id == "maint-c-47")
    h = metadata_header(d)
    for part in ("maintenance record", "component rotor brake", "Wind Farm C", "asset 21", "event C-47", "synthetic"):
        assert part in h
    chunk = metadata_chunks([d])[0]
    assert chunk["text"].startswith(h)


def test_chunkers_are_deterministic(docs):
    assert semantic_chunks(docs, hash_embed) == semantic_chunks(docs, hash_embed)
    assert fixed_size_chunks(docs) == fixed_size_chunks(docs)


# ------------------------------------------------------------------ query set

def test_query_set_size_types_and_relevance(docs, queries):
    ids = {d.doc_id for d in docs}
    assert len(queries) >= 50
    assert len({q.query_id for q in queries}) == len(queries)
    assert {q.query_class for q in queries} == {"conceptual", "quantitative"}
    for q in queries:
        assert q.relevant and set(q.relevant) <= ids
        assert set(q.relevant.values()) <= {1, 2} and 2 in q.relevant.values()


def test_event_queries_relevance_is_correct_by_construction(docs, queries):
    by_id = {d.doc_id: d for d in docs}
    for q in (q for q in queries if q.query_type == "event"):
        for doc_id, grade in q.relevant.items():
            d = by_id[doc_id]
            assert f"turbine {d.asset_id}" in q.text and f"Wind Farm {d.farm}" in q.text
            assert (grade == 2) == (d.doc_type == "fault_description")


def test_query_set_is_deterministic(docs, queries):
    assert build_queries(docs) == queries


# ------------------------------------------------------------------ metrics

def test_metrics_known_values():
    rel = {"a": 2, "b": 1}
    m = metrics_at_k(["x", "a", "a", "y", "b"], rel, k=5)
    assert m["precision_at_k"] == pytest.approx(2 / 5)                    # "a" counted once
    assert m["recall_at_k"] == 1.0 and m["mrr"] == 0.5
    dcg = 2 / math.log2(3) + 1 / math.log2(6)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert m["ndcg_at_k"] == pytest.approx(dcg / idcg)
    perfect = metrics_at_k(["a", "b", "x", "y", "z"], rel)
    assert perfect["ndcg_at_k"] == pytest.approx(1.0) and perfect["mrr"] == 1.0
    assert metrics_at_k(["x", "y"], rel) == {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}


def test_recall_denominator_caps_at_k():
    rel = {f"d{i}": 1 for i in range(8)}
    assert metrics_at_k([f"d{i}" for i in range(5)], rel, k=5)["recall_at_k"] == 1.0


def test_cosine_index_and_evaluate_with_mock_embeddings(docs, queries):
    chunks = metadata_chunks(docs)
    index = CosineIndex(chunks, hash_embed)
    hits = index.search("pitch battery voltage low axis", k=5)
    assert len(hits) == 5 and hits[0]["score"] >= hits[-1]["score"]
    res = evaluate(index, queries[:10], k=5)
    assert len(res) == 10
    for m in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"):
        assert res[m].between(0, 1).all()
    assert (res["retrieved"].str.count(r"\|") == 4).all()
    assert evaluate(index, queries[:10]).equals(res)                        # deterministic


def test_bootstrap_ci_contains_mean():
    mean, lo, hi = bootstrap_mean_ci([0.0, 1.0, 0.5, 0.25, 0.75], n_resamples=2000, seed=0)
    assert lo <= mean <= hi and mean == pytest.approx(0.5)
