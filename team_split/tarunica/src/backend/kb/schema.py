"""B1 - Knowledge-base document schema, provenance and validation.

Every document says where it came from:

- real-data documents (``synthetic = False``) point to a dataset file and the
  0-based data rows (header excluded) they were derived from;
- generated documents (``synthetic = True``) name the generator module and are
  labelled synthetic in their title and text as well as in metadata.

``content_sha256`` lets later phases (C3 document trust verification) detect
edits to a stored document.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DocType = Literal[
    "equipment_manual",
    "maintenance_record",
    "fault_description",
    "troubleshooting_procedure",
    "standard_summary",
    "scada_knowledge",
]
DOC_TYPES: tuple[str, ...] = DocType.__args__  # type: ignore[attr-defined]
SYNTHETIC_MARKER = "[SYNTHETIC]"


class Provenance(BaseModel):
    source: Literal["dataset", "generated"]
    source_file: str | None = Field(None, description="Repo-relative dataset path (real-data documents).")
    source_rows: list[int] = Field(default_factory=list, description="0-based data rows in source_file.")
    generator: str | None = Field(None, description="Module that generated the text (synthetic documents).")
    derived_by: str = Field(..., description="Module that assembled this document.")
    notes: str = ""


class KBDocument(BaseModel):
    doc_id: str = Field(..., min_length=3, pattern=r"^[a-z0-9][a-z0-9_.:-]*$")
    doc_type: DocType
    title: str = Field(..., min_length=3)
    text: str = Field(..., min_length=20)
    domain: Literal["wind", "solar", "general"]
    synthetic: bool
    component: str | None = None                   # primary taxonomy category, if any
    components: list[str] = Field(default_factory=list)
    farm: str | None = None                        # "A" | "B" | "C" for wind-farm documents
    asset_id: int | None = None
    event_id: int | None = None
    severity: str | None = None                    # project-defined default for the component
    tags: list[str] = Field(default_factory=list)
    provenance: Provenance
    content_sha256: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> "KBDocument":
        p = self.provenance
        if self.synthetic:
            if p.source != "generated" or not p.generator:
                raise ValueError(f"{self.doc_id}: synthetic documents need provenance.source='generated' and a generator")
            if SYNTHETIC_MARKER not in self.title or SYNTHETIC_MARKER not in self.text:
                raise ValueError(f"{self.doc_id}: synthetic documents must carry {SYNTHETIC_MARKER} in title and text")
        else:
            if p.source != "dataset" or not p.source_file or not p.source_rows:
                raise ValueError(f"{self.doc_id}: real-data documents need a source_file and source_rows")
            if SYNTHETIC_MARKER in self.title or SYNTHETIC_MARKER in self.text:
                raise ValueError(f"{self.doc_id}: real-data documents must not carry the synthetic marker")
        if self.component and self.component not in self.components:
            raise ValueError(f"{self.doc_id}: primary component must be listed in components")
        return self

    def with_hash(self) -> "KBDocument":
        return self.model_copy(update={"content_sha256": text_hash(self.text)})


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class KBValidationError(ValueError):
    """Raised with every problem found when a knowledge base fails validation."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def validate_kb(docs: list[KBDocument], required_types: tuple[str, ...] = DOC_TYPES) -> None:
    """Collection-level checks (per-document checks run in the pydantic model)."""
    problems: list[str] = []
    seen: set[str] = set()
    for d in docs:
        if d.doc_id in seen:
            problems.append(f"duplicate doc_id {d.doc_id}")
        seen.add(d.doc_id)
        if d.content_sha256 != text_hash(d.text):
            problems.append(f"{d.doc_id}: content_sha256 does not match text")
    present = {d.doc_type for d in docs}
    for t in required_types:
        if t not in present:
            problems.append(f"no documents of type {t}")
    if problems:
        raise KBValidationError(problems)
