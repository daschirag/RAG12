"""C1: docs/threat_model.md is complete, internally consistent and grounded in files that exist."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "threat_model.md"
LAYERS = ("IoT / SCADA", "Ingestion", "Cloud", "Database", "Vector DB", "Retriever", "LLM", "Decision engine")
HEADER = ["ID", "Attack", "Vulnerability (in this system)", "Impact", "Detection", "Mitigation", "Status"]
STATUSES = ("existing", "partial", "planned", "out of scope")


def sections() -> dict[str, list[list[str]]]:
    """Layer heading -> table rows (cells), for the numbered layer sections."""
    out, current = {}, None
    for line in DOC.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## \d+\. (.+)$", line)
        if m:
            current = m.group(1)
            out[current] = []
        elif line.startswith("## "):
            current = None
        elif current and line.startswith("|"):
            out[current].append([c.strip() for c in line.strip().strip("|").split("|")])
    return out


def rows():
    for layer, table in sections().items():
        for r in table[2:]:
            yield layer, r


def test_every_layer_has_a_table_with_the_required_columns():
    s = sections()
    for layer in LAYERS:
        match = [k for k in s if k.startswith(layer)]
        assert match, f"missing section for {layer}"
        table = s[match[0]]
        assert table[0] == HEADER, f"{layer}: header {table[0]}"
        assert len(table) >= 3, f"{layer}: no threats"


def test_every_threat_is_complete_and_uniquely_identified():
    ids = []
    for layer, r in rows():
        assert len(r) == len(HEADER), f"{layer}: row has {len(r)} cells: {r[0]}"
        assert all(len(c) > 2 for c in r), f"{r[0]} has an empty cell"
        assert re.match(r"^[A-Z]{2}-\d+ \([HML]\)$", r[0]), r[0]
        assert r[6].startswith(STATUSES), f"{r[0]} status {r[6]!r}"
        ids.append(r[0].split()[0])
    assert len(ids) == len(set(ids))


def test_file_references_exist():
    """Every backticked repo path mentioned in the tables points to a real file or folder."""
    missing = []
    for _, r in rows():
        for path in re.findall(r"`([\w./-]+\.(?:py|json|jsonl|md|yaml|txt))`", " ".join(r)):
            if not (ROOT / path).exists():
                missing.append(path)
    assert missing == []


def test_existing_and_partial_controls_name_a_file():
    for _, r in rows():
        if r[6].startswith(("existing", "partial")):
            assert re.search(r"`[\w./\\:-]+`", r[4] + r[5] + r[2]), f"{r[0]} claims a control without a file"


def test_planned_items_name_a_phase():
    for _, r in rows():
        if r[6].startswith("planned"):
            assert re.search(r"\b[CI]\d\b", r[6]), f"{r[0]} planned without a phase"


def test_summary_counts_match_the_tables():
    counts = Counter(next(s for s in STATUSES if r[6].startswith(s)) for _, r in rows())
    text = DOC.read_text(encoding="utf-8")
    m = re.search(r"Summary: (\d+) threats across (\d+) layers; (\d+) existing, (\d+) partial, (\d+) planned, "
                  r"(\d+) out of scope", text)
    assert m, "summary line missing"
    total, layers, ex, pa, pl, oos = map(int, m.groups())
    assert (total, layers) == (sum(counts.values()), len(LAYERS))
    assert (ex, pa, pl, oos) == (counts["existing"], counts["partial"], counts["planned"], counts["out of scope"])


@pytest.mark.parametrize("claim, path", [
    ("auth.py is empty", "src/backend/security/auth.py"),
])
def test_stated_facts_still_hold(claim, path):
    assert (ROOT / path).read_text(encoding="utf-8").strip() == "", claim


def test_joblib_load_claim_holds():
    assert "joblib.load" in (ROOT / "src/backend/forecasting/service.py").read_text(encoding="utf-8")


def test_local_only_scope_is_stated():
    text = DOC.read_text(encoding="utf-8")
    assert "local mode only" in text.lower() and "out of scope" in text
