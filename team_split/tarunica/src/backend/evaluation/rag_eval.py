"""B6 - RAG answer evaluation: claim-level faithfulness, hallucination, relevance, numeric exact match.

Claim-level, not citation-level: the whole answer text is split into sentences
("claims") and every claim is judged against evidence, so an answer whose
citations are valid can still be caught making unsupported statements in its
prose (the failure found in B5).

- faithfulness(answer, retrieved context): share of claims supported by the
  system's OWN retrieved context (undefined for LLM-only: no context).
- hallucination: a claim is unsupported if neither the reference documents
  (relevant by construction) nor the system's retrieved context support it.
  Per answer: any unsupported claim -> hallucinated. Comparable across all
  systems, including LLM-only.
- numeric check (no judge): every number in the answer must occur in the
  reference text; count questions are scored by exact match against counts
  computed from event_info.csv.
- answer relevance: judge score 1-5 mapped to 0-1.
- context relevance: from relevance labels (share of retrieved documents that
  are relevant by construction), not from the judge.

The judge is a local LLM. It is not ground truth: it has its own error and
biases (including self-preference when it grades output of the same model).
Its accuracy is measured on a calibration set with known labels.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass

LLM = Callable[[str], str]
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")
_NUM = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?(?![\w])")
MIN_CLAIM_WORDS = 4


_CITATION = re.compile(r"\[[^\]]*::\d+[^\]]*\]")


def strip_citations(text: str) -> str:
    """Remove bracketed chunk-id citations such as [fault-a-73::0, trouble-wind-gearbox::1]."""
    return re.sub(r"\s{2,}", " ", _CITATION.sub("", text or "")).strip()


def split_claims(text: str) -> list[str]:
    """Sentences with at least MIN_CLAIM_WORDS words (list items and lines count as sentences), citations removed."""
    claims = []
    for line in strip_citations(text).split("\n"):
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line).strip()
        for s in _SENT.split(line):
            s = s.strip()
            if len(re.findall(r"\w+", s)) >= MIN_CLAIM_WORDS:
                claims.append(s)
    return claims


def numbers_in(text: str) -> list[str]:
    return [n.replace(",", ".") for n in _NUM.findall(text or "")]


def unsupported_numbers(answer: str, reference: str) -> list[str]:
    """Numbers in the answer that do not occur (as numbers) in the reference text."""
    ref = set(numbers_in(reference))
    ref |= {r.rstrip("0").rstrip(".") for r in ref if "." in r}
    out = []
    for n in numbers_in(strip_citations(answer)):
        norm = n.rstrip("0").rstrip(".") if "." in n else n
        if n not in ref and norm not in ref:
            out.append(n)
    return out


def extract_count(answer: str) -> int | None:
    """First integer in the answer, or a small number word."""
    words = {"no": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
    m = re.search(r"\b(\d+)\b", strip_citations(answer))
    if m:
        return int(m.group(1))
    for w in re.findall(r"[a-z]+", (answer or "").lower()):
        if w in words:
            return words[w]
    return None


# --------------------------------------------------------------------- judge

SUPPORT_PROMPT = (
    "You check whether a claim is supported by a context.\n"
    "Context:\n\"\"\"\n{context}\n\"\"\"\n\n"
    "Claim: \"{claim}\"\n\n"
    "Is the claim fully supported by the context (every fact in the claim is stated or directly implied by the "
    "context)? General statements that add facts not in the context are NOT supported. Statements that the "
    "information is unavailable are supported only if the context indeed does not contain it.\n"
    'Answer ONLY with JSON: {{"verdict": "supported"}} or {{"verdict": "not_supported"}}'
)
RELEVANCE_PROMPT = (
    "Rate how well the answer addresses the question, from 1 (does not address it at all) to 5 (directly and "
    "completely addresses it). Judge relevance only, not correctness.\n"
    "Question: {question}\nAnswer: {answer}\n"
    'Answer ONLY with JSON: {{"score": <1-5>}}'
)


# v2 (B6 follow-up): the v1 prompt showed only the retrieved text, so the judge could not verify true statements
# about the retrieval itself ("no troubleshooting procedure was retrieved") or restatements of the input
# ("anomaly signal for turbine 0"). v2 adds the question and the list of retrieved documents and says how such
# statements are checked. The standard for facts about equipment and faults is unchanged.
SUPPORT_PROMPT_V2 = (
    "You check whether a claim in a system's answer is supported.\n"
    "Question given to the system:\n\"\"\"\n{question}\n\"\"\"\n"
    "Documents the system retrieved (ids and titles):\n\"\"\"\n{manifest}\n\"\"\"\n"
    "Text of the retrieved documents:\n\"\"\"\n{context}\n\"\"\"\n\n"
    "Claim: \"{claim}\"\n\n"
    "Rules:\n"
    "1. Facts about equipment, faults, causes, actions or numbers are supported only if the document text states or "
    "directly implies them. Added facts or speculation are NOT supported.\n"
    "2. Statements about which documents were or were not retrieved, or that the retrieved text does or does not "
    "state something, are supported only if they are true according to the list and text above.\n"
    "3. Restating information given in the question (turbine, wind farm, date, that an anomaly was signalled) is "
    "supported.\n"
    'Answer ONLY with JSON: {{"verdict": "supported"}} or {{"verdict": "not_supported"}}'
)


def judge_claim_v2(judge: LLM, claim: str, context: str, question: str, manifest: str,
                   max_context: int = 6000) -> bool | None:
    return parse_verdict(judge(SUPPORT_PROMPT_V2.format(question=question or "(none)",
                                                         manifest=manifest or "(none)",
                                                         context=context[:max_context] or "(none)", claim=claim)))


def process_calibration_items(records: list[dict]) -> list[dict]:
    """Process/input statements with known truth, built from real answer records (question + retrieved docs).

    For each record: true and false versions of 'X was retrieved', 'no troubleshooting procedure was retrieved'
    and 'anomaly signal for turbine N'. Each item: claim, question, manifest, context, supported, kind.
    """
    items = []
    all_docs = sorted({c["doc_id"] for r in records for c in r["context"]})
    for r in records:
        docs = list(dict.fromkeys(c["doc_id"] for c in r["context"]))
        if not docs:
            continue
        manifest = "\n".join(docs)
        context = "\n\n".join(c["text"] for c in r["context"])
        q = r["question"]
        turbine = re.search(r"turbine (\d+)", q)
        absent = next((d for d in all_docs if d not in docs and d.startswith("fault-")), None)
        base = {"question": q, "manifest": manifest, "context": context}
        items.append({**base, "claim": f"The record {docs[0]} was retrieved.", "supported": True,
                      "kind": "process_true"})
        if absent:
            items.append({**base, "claim": f"The record {absent} was retrieved.", "supported": False,
                          "kind": "process_false"})
        has_trouble = any(d.startswith("trouble-") for d in docs)
        items.append({**base, "claim": "No troubleshooting procedure was retrieved.", "supported": not has_trouble,
                      "kind": "process_true" if not has_trouble else "process_false"})
        if turbine:
            n = int(turbine.group(1))
            items.append({**base, "claim": f"There is an anomaly signal for turbine {n}.", "supported": True,
                          "kind": "input_true"})
            items.append({**base, "claim": f"There is an anomaly signal for turbine {n + 100}.", "supported": False,
                          "kind": "input_false"})
    return items


def calibrate_judge_v2(judge: LLM, items: list[CalibrationItem], process_items: list[dict]) -> dict:
    """v2 judge on the original calibration items (no question/manifest) plus process/input items."""
    rows = [(i.supported, i.kind, judge_claim_v2(judge, i.claim, i.context, "", "")) for i in items]
    rows += [(p["supported"], p["kind"], judge_claim_v2(judge, p["claim"], p["context"], p["question"], p["manifest"]))
             for p in process_items]
    out = {"n": len(rows), "unparsed": sum(v is None for _, _, v in rows), "by_kind": {}}
    for kind in sorted({k for _, k, _ in rows}):
        sel = [(s, v) for s, k, v in rows if k == kind]
        out["by_kind"][kind] = {"n": len(sel), "accuracy": sum((v is True) == s for s, v in sel) / len(sel)}
    pos = [(s, v) for s, _, v in rows if s]
    neg = [(s, v) for s, _, v in rows if not s]
    out["true_positive_rate"] = sum(v is True for _, v in pos) / len(pos)
    out["true_negative_rate"] = sum(v is False for _, v in neg) / len(neg)
    out["balanced_accuracy"] = (out["true_positive_rate"] + out["true_negative_rate"]) / 2
    return out


def verify_template_claim(claim: str, question: str, context: list[dict], components: dict[str, str | None],
                          labels: dict[str, str]) -> bool | None:
    """Deterministic check of the statements GreenRAG's fallback writes from templates (faultdiag.diagnosis).

    Those statements describe the retrieval (which records were retrieved, whether a cause text or a procedure
    was among them) or restate the input, and are filled in by code from the actual evidence, so their truth is
    checkable against the retrieved evidence without a judge. Returns None for any claim that is not one of the
    templates (those are left to the LLM judge). ``labels`` maps component keys to their labels.
    """
    text = re.sub(r"^(Likely cause|Recommended action):\s*", "", claim.strip())
    docs = [c["doc_id"] for c in context]
    comp_labels = {labels.get(components.get(d) or "", "").lower() for d in docs} - {""}
    m = re.match(r"Anomaly signal for Wind Farm (\w) turbine (\d+)(?:; evidence points to the (.+?))?\.?$", text)
    if m:
        ok = f"Wind Farm {m.group(1)} turbine {m.group(2)} " in question + " "
        return ok and (m.group(3) is None or m.group(3).lower() in comp_labels)
    m = re.match(r'Logged description for this incident: "(.*)"', text, flags=re.S)
    if m:
        return any(m.group(1) in c["text"] for c in context)
    m = re.match(r"The record of this incident \((.+?)\) was retrieved, but the retrieved text does not state", text)
    if m:
        return m.group(1) in docs and not any("original wording" in c["text"] for c in context
                                              if c["doc_id"] == m.group(1))
    m = re.match(r"Not established for this incident; the closest evidence concerns (.+?) \((.+?)\)", text)
    if m:
        label = labels.get(components.get(m.group(2)) or "", "an unrelated topic").lower()
        return m.group(2) in docs and m.group(1).lower() == label
    m = re.match(r"Follow the checks in (.+?) \(synthetic procedure\)", text)
    if m:
        return m.group(1) in docs
    if re.match(r"Inspect the turbine; no troubleshooting procedure was retrieved", text):
        return not any(d.startswith("trouble-") for d in docs)
    if "no evidence was retrieved" in text or text.startswith("Not established: no evidence retrieved"):
        return not docs
    return None


def parse_verdict(raw: str) -> bool | None:
    try:
        v = json.loads(raw).get("verdict", "")
    except (json.JSONDecodeError, AttributeError):
        v = raw or ""
    v = str(v).lower()
    if "not" in v and "support" in v:
        return False
    if "support" in v:
        return True
    return None


def parse_score(raw: str) -> float | None:
    try:
        s = json.loads(raw).get("score")
    except (json.JSONDecodeError, AttributeError):
        m = re.search(r"[1-5]", raw or "")
        s = m.group(0) if m else None
    try:
        s = float(s)
    except (TypeError, ValueError):
        return None
    return (min(max(s, 1), 5) - 1) / 4


def judge_claim(judge: LLM, claim: str, context: str, max_context: int = 6000) -> bool | None:
    return parse_verdict(judge(SUPPORT_PROMPT.format(context=context[:max_context], claim=claim)))


def judge_relevance(judge: LLM, question: str, answer: str) -> float | None:
    return parse_score(judge(RELEVANCE_PROMPT.format(question=question, answer=answer[:3000])))


# ----------------------------------------------------------- judge calibration

@dataclass(frozen=True)
class CalibrationItem:
    claim: str
    context: str
    supported: bool
    kind: str


def calibration_set(docs, n_each: int = 20, seed: int = 0) -> list[CalibrationItem]:
    """Claims with known support: sentences from the context (supported), from a document about another
    component (unsupported), and context sentences with one number changed (unsupported)."""
    rng = random.Random(seed)
    usable = [d for d in docs if d.component and len(split_claims(d.text)) >= 3]
    rng.shuffle(usable)
    items: list[CalibrationItem] = []
    by_comp: dict[str, list] = {}
    for d in usable:
        by_comp.setdefault(d.component, []).append(d)
    for d in usable:
        sents = [s for s in split_claims(d.text) if "SYNTHETIC" not in s and len(s) < 300]
        if len(sents) < 2:
            continue
        context = d.text[:2500]
        kinds = [k for k in ("supported", "other_component", "changed_number")
                 if sum(i.kind == k for i in items) < n_each]
        if not kinds:
            break
        kind = kinds[len(items) % len(kinds)]
        if kind == "supported":
            items.append(CalibrationItem(rng.choice(sents), context, True, kind))
        elif kind == "other_component":
            others = [o for c, ds in by_comp.items() if c != d.component for o in ds]
            other = rng.choice(others)
            osents = [s for s in split_claims(other.text) if "SYNTHETIC" not in s and len(s) < 300]
            if osents:
                items.append(CalibrationItem(rng.choice(osents), context, False, kind))
        else:
            numeric = [s for s in sents if numbers_in(s)]
            if numeric:
                s = rng.choice(numeric)
                n = numbers_in(s)[0]
                changed = str(int(float(n)) + 7) if n.replace("-", "").isdigit() else str(round(float(n) * 3.1, 1))
                items.append(CalibrationItem(s.replace(n, changed, 1), context, False, kind))
    return items


def calibrate_judge(judge: LLM, items: list[CalibrationItem]) -> dict:
    rows = [(i, judge_claim(judge, i.claim, i.context)) for i in items]
    tp = sum(1 for i, v in rows if i.supported and v is True)
    tn = sum(1 for i, v in rows if not i.supported and v is False)
    pos = sum(i.supported for i in items)
    neg = len(items) - pos
    unparsed = sum(v is None for _, v in rows)
    by_kind = {}
    for kind in {i.kind for i in items}:
        sel = [(i, v) for i, v in rows if i.kind == kind]
        by_kind[kind] = sum((v is True) == i.supported for i, v in sel) / len(sel)
    tpr, tnr = tp / pos if pos else 0.0, tn / neg if neg else 0.0
    return {"n": len(items), "accuracy": (tp + tn) / len(items), "balanced_accuracy": (tpr + tnr) / 2,
            "true_positive_rate": tpr, "true_negative_rate": tnr, "unparsed": unparsed, "by_kind": by_kind}
