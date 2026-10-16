"""B2 - Chunking strategies for the knowledge base.

All strategies take KBDocuments and return records in the ingestion format
``{"chunk_id", "domain_id", "text", "metadata"}`` so they are interchangeable
for indexing. No chunk ever spans two documents, and every chunk carries its
``doc_id`` so retrieval can be scored against document-level relevance.

- ``fixed_size_chunks``   (A) character windows with overlap, cut at word
  boundaries; ignores document structure and metadata.
- ``semantic_chunks``     (B) sentence/line units, embedded; a new chunk starts
  where the similarity between neighbouring units drops below the document's
  mean minus one standard deviation (or when the size limit is reached).
- ``metadata_chunks``     (C) paragraph packing inside each document plus a
  header built from the KB metadata (type, component, farm, asset, event,
  real vs synthetic), prepended to every chunk.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

from .schema import KBDocument

KB_DOMAIN = "knowledge_base"
EmbedFn = Callable[[Sequence[str]], np.ndarray]     # list of texts -> (n, d) array


def pack_paragraphs(text: str, max_chars: int) -> list[str]:
    """Pack paragraphs into pieces of at most ``max_chars``; over-long paragraphs split on lines."""
    pieces, current = [], ""
    for para in text.split("\n\n"):
        lines = para.split("\n") if len(para) > max_chars else [para]
        for piece in lines:
            sep = "\n\n" if piece is para else "\n"
            if current and len(current) + len(sep) + len(piece) > max_chars:
                pieces.append(current)
                current = piece
            else:
                current = f"{current}{sep}{piece}" if current else piece
    if current:
        pieces.append(current)
    return pieces


def _record(doc: KBDocument, i: int, n: int, text: str, method: str) -> dict:
    meta = {"doc_id": doc.doc_id, "doc_type": doc.doc_type, "title": doc.title, "domain": doc.domain,
            "synthetic": doc.synthetic, "component": doc.component, "components": ",".join(doc.components),
            "farm": doc.farm, "asset_id": doc.asset_id, "event_id": doc.event_id, "severity": doc.severity,
            "source": doc.provenance.source, "source_file": doc.provenance.source_file,
            "source_rows": ",".join(str(r) for r in doc.provenance.source_rows) or None,
            "doc_sha256": doc.content_sha256, "chunk_index": i, "n_chunks": n, "chunking": method}
    return {"chunk_id": f"{doc.doc_id}::{i}", "domain_id": KB_DOMAIN, "text": text,
            "metadata": {k: v for k, v in meta.items() if v is not None}}


# ------------------------------------------------------------------ (A) fixed size

def fixed_windows(text: str, size: int, overlap: int) -> list[str]:
    """Windows of about ``size`` characters, stepping ``size - overlap``, cut at whitespace."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = text.strip()
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)          # do not split a word
            end = cut if cut > start else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        nxt = end - overlap
        space = text.find(" ", nxt)
        start = space + 1 if 0 <= space < end else nxt
    return [w for w in out if w]


def fixed_size_chunks(docs: list[KBDocument], size: int = 800, overlap: int = 100) -> list[dict]:
    out = []
    for d in docs:
        windows = fixed_windows(d.text, size, overlap)
        out += [_record(d, i, len(windows), w, "fixed_size") for i, w in enumerate(windows)]
    return out


# ---------------------------------------------------------------- (B) semantic

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[-])")


def split_units(text: str) -> list[str]:
    """Sentence-like units: lines (lists, headers) further split into sentences."""
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            units += [s.strip() for s in _SENTENCE.split(line) if s.strip()]
    return units


def semantic_boundaries(emb: np.ndarray) -> list[int]:
    """Indices i where a new chunk starts before unit i (similarity drop between i-1 and i)."""
    if len(emb) < 3:
        return []
    norm = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    sims = np.sum(norm[1:] * norm[:-1], axis=1)
    threshold = sims.mean() - sims.std()
    return [i + 1 for i, s in enumerate(sims) if s < threshold]


def semantic_chunks(docs: list[KBDocument], embed: EmbedFn, max_chars: int = 1000,
                    min_chars: int = 200) -> list[dict]:
    units_per_doc = [split_units(d.text) for d in docs]
    flat = [u for units in units_per_doc for u in units]
    all_emb = np.asarray(embed(flat), dtype=float) if flat else np.zeros((0, 1))
    out, pos = [], 0
    for d, units in zip(docs, units_per_doc):
        emb = all_emb[pos:pos + len(units)]
        pos += len(units)
        breaks = set(semantic_boundaries(emb))
        pieces, current = [], ""
        for i, unit in enumerate(units):
            start_new = current and ((i in breaks and len(current) >= min_chars)
                                     or len(current) + 1 + len(unit) > max_chars)
            if start_new:
                pieces.append(current)
                current = unit
            else:
                current = f"{current}\n{unit}" if current else unit
        if current:
            if pieces and len(current) < min_chars and len(pieces[-1]) + 1 + len(current) <= max_chars:
                pieces[-1] = f"{pieces[-1]}\n{current}"        # merge a short tail
            else:
                pieces.append(current)
        out += [_record(d, i, len(pieces), p, "semantic") for i, p in enumerate(pieces)]
    return out


# ---------------------------------------------------------- (C) metadata-aware

def metadata_header(doc: KBDocument) -> str:
    parts = [doc.doc_type.replace("_", " ")]
    if doc.component:
        parts.append(f"component {doc.component.replace('_', ' ')}")
    if doc.farm:
        parts.append(f"Wind Farm {doc.farm}")
    if doc.asset_id is not None:
        parts.append(f"asset {doc.asset_id}")
    if doc.event_id is not None:
        parts.append(f"event {doc.farm}-{doc.event_id}")
    parts.append(f"domain {doc.domain}")
    parts.append("synthetic" if doc.synthetic else "real data")
    return f"{doc.title}\n[{' | '.join(parts)}]"


def metadata_chunks(docs: list[KBDocument], max_chars: int = 1000) -> list[dict]:
    out = []
    for d in docs:
        header = metadata_header(d)
        pieces = pack_paragraphs(d.text, max_chars)
        out += [_record(d, i, len(pieces), f"{header}\n{p}", "metadata_aware") for i, p in enumerate(pieces)]
    return out


CHUNKERS = ("fixed_size", "semantic", "metadata_aware")
