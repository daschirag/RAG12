"""C3 - Evidence filter between retrieval and generation.

C2 found that GreenRAG's extractive fallback quotes the top-ranked chunk verbatim when the LLM answer fails
its citation check - including a poisoned "Revision 2" procedure. The filter runs document trust
verification (doc_trust.py) on every retrieved chunk **before** anything reaches the LLM or a fallback:

- ``mode="drop"``      untrusted chunks are removed; the retriever over-fetches so ``k`` trusted chunks remain
                       when possible (the production reranker returns at most its candidate pool of 20);
- ``mode="annotate"``  every chunk is kept and marked ``hit["trust"] = {"trusted": ..., "reasons": [...]}``;
                       consumers must honour the mark (GreenRAG's fallback refuses to quote a chunk marked
                       untrusted, see evaluation/rag_systems.py).

``TrustedRetriever`` has the same ``search(query, k)`` interface as the production retriever, so the RAG
systems and the Diagnoser use it unchanged. Wiring it into the served endpoints is C4 (secure retrieval).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .doc_trust import DocumentTrust


@dataclass
class EvidenceFilter:
    trust: DocumentTrust
    mode: str = "drop"                                  # "drop" | "annotate"

    def apply(self, hits: list[dict]) -> tuple[list[dict], list[dict]]:
        """(passed-on hits, rejected hits); each rejected hit carries its trust reasons."""
        kept, rejected = [], []
        for h in hits:
            rep = self.trust.check_hit(h)
            marked = {**h, "trust": {"trusted": rep.trusted, "reasons": rep.reasons}}
            if rep.trusted:
                kept.append(marked)
            else:
                rejected.append(marked)
                if self.mode == "annotate":
                    kept.append(marked)
        return kept, rejected


@dataclass
class TrustedRetriever:
    inner: object
    evidence_filter: EvidenceFilter
    overfetch: int = 10
    last_rejected: list[dict] = field(default_factory=list)

    def search_with_report(self, query: str, k: int = 5) -> tuple[list[dict], list[dict]]:
        """(up to k trusted hits, rejected hits) for one call; safe to use from concurrent requests."""
        kept, rejected = self.evidence_filter.apply(self.inner.search(query, k + self.overfetch))
        return kept[:k], rejected

    def search(self, query: str, k: int = 5) -> list[dict]:
        kept, self.last_rejected = self.search_with_report(query, k)    # last_rejected: single-threaded use only
        return kept
