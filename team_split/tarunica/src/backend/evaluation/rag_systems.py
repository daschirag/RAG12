"""B6 - The four systems compared: LLM-only, basic RAG, hybrid RAG, GreenRAG.

All use the same generator LLM and the same answer prompt, so differences
come from retrieval and GreenRAG's safeguards:

- llm_only:   no retrieval.
- basic_rag:  MiniLM vector search over fixed-size chunks (the original approach), top 5.
- hybrid_rag: BM25 + MiniLM with reciprocal rank fusion over semantic chunks, top 5.
- greenrag:   production retrieval (hybrid + cross-encoder rerank, top 5), citations validated against
              the retrieved evidence (answers without a valid citation are replaced by an extractive
              evidence answer), count questions answered deterministically from event_info.csv (the query
              classifier flags them), and for fault events the full B5 Diagnoser report.

Every answer records the retrieved chunks (context) so faithfulness can be judged against it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

from ..kb.taxonomy import COMPONENTS, FARMS, categorize, load_events
from ..retrieval.query_classifier import classify_intent

K = 5
ANSWER_PROMPT = (
    "Answer the question using ONLY the context below. Do not use outside knowledge and do not invent facts or "
    "numbers. If the context does not contain the answer, say that it does not. Documents marked [SYNTHETIC] are "
    "generated guidance, not real records. Keep the answer to at most four sentences and cite the ids of the "
    "context chunks you used in square brackets.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)
LLM_ONLY_PROMPT = (
    "Answer the question in at most four sentences. If you do not know, say so.\n\nQuestion: {question}\n\nAnswer:"
)
GREEN_PROMPT = (
    "Answer the question using ONLY the context below. Do not use outside knowledge and do not invent facts or "
    "numbers. If the context does not contain the answer, say that it does not. Documents marked [SYNTHETIC] are "
    "generated guidance, not real records.\n\nContext:\n{context}\n\nQuestion: {question}\n\n"
    'Return ONLY JSON: {{"answer": "<at most four sentences>", "chunk_ids": ["<ids of the chunks used>"]}}'
)


@dataclass
class Answer:
    system: str
    text: str
    context: list[dict] = field(default_factory=list)      # retrieved chunks: chunk_id, doc_id, text
    route: str = "generated"                               # generated | extractive_fallback | numeric | diagnoser
    notes: list[str] = field(default_factory=list)


def _context_block(hits) -> str:
    return "\n\n".join(f"[{h['chunk_id']}] {h['text']}" for h in hits)


def _ctx(hits) -> list[dict]:
    return [{"chunk_id": h["chunk_id"], "doc_id": h["metadata"]["doc_id"], "text": h["text"]} for h in hits]


def llm_only(llm, question: str) -> Answer:
    return Answer("llm_only", llm(LLM_ONLY_PROMPT.format(question=question)).strip())


def rag_answer(system: str, llm, retriever, question: str, exclude=None, expander=None) -> Answer:
    hits = [h for h in retriever.search(question, 20) if not (exclude and exclude(h))][:K]
    if expander is not None:
        hits = expander.expand(hits)          # siblings of the kept hits only, so exclusions still hold
    return Answer(system, llm(ANSWER_PROMPT.format(context=_context_block(hits), question=question)).strip(),
                  _ctx(hits))


# ------------------------------------------------------------------ GreenRAG

def count_answer(question: str, events: pd.DataFrame | None = None) -> str | None:
    """Deterministic answer to 'how many <component> failures ... Wind Farm <X>' (None if not parseable)."""
    q = question.lower()
    farm = next((f for f in FARMS if f"wind farm {f.lower()}" in q), None)
    comp = next((c for c in COMPONENTS if c.label.lower() in q), None)
    if farm is None or comp is None:
        return None
    events = load_events(farm) if events is None else events[events.farm == farm]
    anomalies = events[events.event_label == "anomaly"]
    cats = [categorize(d) for d in anomalies.event_description]
    any_count = sum(comp.key in c[1] for c in cats)
    primary = sum(c[0] == comp.key for c in cats)
    return (f"{any_count} anomaly event(s) involving the {comp.label.lower()} were recorded in Wind Farm {farm} "
            f"({primary} with it as the primary component), computed from event_info.csv with the B1 taxonomy rules.")


def greenrag_answer(llm, retriever, question: str, exclude=None) -> Answer:
    hits = [h for h in retriever.search(question, 20) if not (exclude and exclude(h))][:K]
    ctx = _ctx(hits)
    if classify_intent(question) == "quantitative":
        computed = count_answer(question)
        if computed is not None:
            return Answer("greenrag", computed, ctx, route="numeric")
    ids = {h["chunk_id"] for h in hits}
    by_doc: dict[str, list[str]] = {}
    for h in hits:
        by_doc.setdefault(h["chunk_id"].split("::")[0], []).append(h["chunk_id"])
    notes = []
    try:
        data = json.loads(llm(GREEN_PROMPT.format(context=_context_block(hits), question=question)))
        text = str(data["answer"]).strip()
        cited = []
        for c in data.get("chunk_ids") or []:
            if c in ids:
                cited.append(c)
            elif c in by_doc:
                cited.extend(by_doc[c])
        if text and cited:
            return Answer("greenrag", f"{text} [{', '.join(dict.fromkeys(cited))}]", ctx)
        notes.append("no valid citation in LLM answer")
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        notes.append(f"invalid LLM output ({type(exc).__name__})")
    except Exception as exc:  # noqa: BLE001 - backend failure (e.g. timeout after retry): evidence-only answer
        notes.append(f"LLM unavailable ({type(exc).__name__})")
    # The fallback quotes evidence verbatim, so it never quotes a chunk that an evidence filter marked
    # untrusted (C3; unmarked hits are quoted as before).
    quotable = [h for h in hits if h.get("trust", {}).get("trusted", True)]
    if not hits:
        return Answer("greenrag", "The knowledge base contains no evidence for this question.", ctx,
                      route="extractive_fallback", notes=notes)
    if not quotable:
        return Answer("greenrag", "The knowledge base contains no trusted evidence for this question.", ctx,
                      route="extractive_fallback", notes=notes)
    top = quotable[0]
    body = re.sub(r"\s+", " ", top["text"])[:400]
    return Answer("greenrag", f"The most relevant evidence [{top['chunk_id']}] states: {body}", ctx,
                  route="extractive_fallback", notes=notes)


def diagnosis_text(report) -> str:
    return (f"{report.diagnosis} Likely cause: {report.likely_cause.text} "
            f"Recommended action: {report.recommended_action.text} Severity: {report.severity}. "
            f"Confidence: {report.confidence.level}.")


def fault_question(signal) -> str:
    return (f"Wind Farm {signal.farm} turbine {signal.asset_id} shows an anomaly starting "
            f"{signal.window_start:%Y-%m-%d}. What is the likely cause and what should be done?")
