"""C3 - Document trust verification for knowledge-base evidence.

B1 stores ``content_sha256`` in every KBDocument. That hash is *self-declared*: whoever writes a document
can compute it, so on its own it only detects accidental corruption. Trust therefore comes from a
**registry** - the doc_id -> sha256 map of the KB as built by ``src.backend.kb.build`` - signed with an HMAC
key held outside the KB (``GREENRAG_TRUST_KEY``). Checks per document (and per retrieved chunk):

- ``registered``     the doc_id is in the signed registry (added documents fail);
- ``registry_hash``  the current text hashes to the registered value (edited documents fail);
- ``self_hash``      the declared content_sha256 matches the text (reported, not sufficient on its own);
- ``provenance``     real fault records must quote the logged description of their event in event_info.csv;
- ``stuffing``       the most frequent content word is not an outlier share of the text (keyword stuffing);
- ``injection``      no prompt-injection signal (src/backend/security/injection_detector.py);
- ``chunk_in_doc``   (chunks) the chunk text occurs in its document.

The last four are content checks that still work when the registry itself is compromised.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..kb.schema import KBDocument, text_hash
from .injection_detector import injection_signals, is_injection

KEY_ENV = "GREENRAG_TRUST_KEY"
CONTENT_CHECKS = ("provenance", "stuffing", "injection")
REGISTRY_CHECKS = ("registered", "registry_hash")
STOPWORDS = frozenset(
    "the a an and or of to in on for with is are be by as at from this that it its not no if was were has have "
    "which what when than then there their these those into per can may should must only also any all each".split())


class RegistryError(ValueError):
    """The registry signature does not verify: nothing in it can be trusted."""


# ------------------------------------------------------------------ registry

def _canonical(docs: dict[str, str]) -> bytes:
    return json.dumps(docs, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign(docs: dict[str, str], key: bytes) -> str:
    return hmac.new(key, _canonical(docs), hashlib.sha256).hexdigest()


def build_registry(docs: list[KBDocument], key: bytes | None) -> dict:
    """Registry of the trusted KB build; hashes are recomputed from the text, never copied from the field."""
    entries = {d.doc_id: text_hash(d.text) for d in docs}
    return {"version": 1, "docs": entries, "signature": sign(entries, key) if key else None}


def verify_registry(registry: dict, key: bytes | None) -> dict[str, str]:
    """Return the doc_id -> hash map, raising RegistryError when a key is given and the signature fails."""
    docs = registry["docs"]
    if key is not None and not (registry.get("signature")
                                and hmac.compare_digest(registry["signature"], sign(docs, key))):
        raise RegistryError("trust registry signature does not verify")
    return docs


def env_key() -> bytes | None:
    value = os.getenv(KEY_ENV)
    return value.encode("utf-8") if value else None


def save_registry(registry: dict, path: Path) -> None:
    path.write_text(json.dumps(registry, indent=1, sort_keys=True), encoding="utf-8")


def load_registry(path: Path, key: bytes | None) -> dict[str, str]:
    return verify_registry(json.loads(path.read_text(encoding="utf-8")), key)


# ------------------------------------------------------------------ content checks

def stuffing_share(text: str) -> float:
    """Share of the most frequent content word among content words (0 for short texts)."""
    words = [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOPWORDS]
    if len(words) < 50:
        return 0.0
    return Counter(words).most_common(1)[0][1] / len(words)


def calibrate_stuffing(clean_docs: list[KBDocument], margin: float = 1.5) -> float:
    """Threshold from the clean KB only: ``margin`` x the largest share observed there."""
    return margin * max(stuffing_share(d.text) for d in clean_docs)


def logged_descriptions(events) -> dict[tuple[str, int], str]:
    """(farm, event_id) -> logged description, from the event_info.csv frames (``kb.taxonomy.load_all_events``)."""
    return {(str(e.farm), int(e.event_id)): str(e.event_description) for e in events.itertuples()
            if isinstance(e.event_description, str)}


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


@dataclass
class TrustReport:
    doc_id: str
    trusted: bool
    checks: dict[str, bool]                     # check -> passed
    reasons: list[str] = field(default_factory=list)


@dataclass
class DocumentTrust:
    registry: dict[str, str] | None             # verified doc_id -> sha256; None = registry unavailable
    store: dict[str, KBDocument]                # documents as currently served
    descriptions: dict[tuple[str, int], str]
    stuffing_threshold: float
    required: tuple[str, ...] = REGISTRY_CHECKS + CONTENT_CHECKS

    def check_document(self, doc: KBDocument) -> TrustReport:
        reg = self.registry or {}
        checks = {
            "registered": doc.doc_id in reg,
            "registry_hash": reg.get(doc.doc_id) == text_hash(doc.text),
            "self_hash": doc.content_sha256 == text_hash(doc.text),
            "provenance": self._provenance_ok(doc),
            "stuffing": stuffing_share(doc.text) <= self.stuffing_threshold,
            "injection": not is_injection(doc.text),
        }
        failed = [c for c in self.required if not checks[c]]
        reasons = [f"{c} failed" for c in failed]
        if not checks["injection"]:
            reasons.append("injection signals: " + ", ".join(injection_signals(doc.text)))
        return TrustReport(doc.doc_id, not failed, checks, reasons)

    def check_hit(self, hit: dict) -> TrustReport:
        doc_id = hit["metadata"]["doc_id"]
        doc = self.store.get(doc_id)
        if doc is None:
            return TrustReport(doc_id, False, {"known_document": False}, ["document not in the store"])
        rep = self.check_document(doc)
        in_doc = _squash(hit["text"]) in _squash(doc.text)
        rep.checks["chunk_in_doc"] = in_doc
        chunk_injection = is_injection(hit["text"])
        rep.checks["chunk_injection"] = not chunk_injection
        if not in_doc:
            rep.trusted, rep.reasons = False, rep.reasons + ["chunk text not found in its document"]
        if chunk_injection:
            rep.trusted, rep.reasons = False, rep.reasons + ["injection signals in the chunk"]
        return rep

    def _provenance_ok(self, doc: KBDocument) -> bool:
        """Real fault records must quote their event's logged description exactly; other documents pass."""
        if doc.synthetic or doc.doc_type != "fault_description" or doc.farm is None or doc.event_id is None:
            return True
        logged = self.descriptions.get((doc.farm, int(doc.event_id)))
        if logged is None:
            return False                              # claims an event that the dataset does not have
        return f'(original wording): "{logged}"' in doc.text
