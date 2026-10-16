"""B1 knowledge base: taxonomy rules, provenance, synthetic labelling, validation, ingestion format."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.backend import main
from src.backend.config import list_domains, load_manifest
from src.backend.kb.build import build_documents, chunk_documents, load_documents, write_kb
from src.backend.kb.schema import (
    DOC_TYPES,
    SYNTHETIC_MARKER,
    KBDocument,
    KBValidationError,
    Provenance,
    text_hash,
    validate_kb,
)
from src.backend.kb.taxonomy import (
    PROJECT_ROOT,
    UNSPECIFIED,
    categorize,
    load_all_events,
    load_events,
    load_features,
    taxonomy_table,
)
from src.backend.retrieval.metrics import flatten_chunk_metadata


@pytest.fixture(scope="module")
def docs() -> list[KBDocument]:
    return build_documents()


# ----------------------------------------------------------------- taxonomy

@pytest.mark.parametrize("description, primary, all_components", [
    ("Hydraulic group", "hydraulic_system", ["hydraulic_system"]),
    ("high temperature in transformer cell", "transformer", ["transformer"]),
    ("Turbine is in standstill since 01.08 due to rotorbearing damage.", "rotor_main_bearing", ["rotor_main_bearing"]),
    ("23020 : Axis 3 not ready-to-operate", "pitch_system", ["pitch_system"]),
    ("15004 : Safety chain relay open + 93005 : Gear oil cooler bypass valve", "electrical_auxiliary",
     ["electrical_auxiliary", "gearbox", "cooling_system"]),
    ("WEC in failure with pitch battery issues - rewiring", "pitch_system", ["pitch_system"]),   # not "wiring"
    ("P20_Grounding role brake disc + P20_cover-lightning-main-cabinet-hub", "blades_hub_structure",
     ["blades_hub_structure", "rotor_brake"]),
    (None, UNSPECIFIED, []),
    ("something unrelated", UNSPECIFIED, []),
])
def test_categorize_rules(description, primary, all_components):
    assert categorize(description) == (primary, all_components)


def test_event_files_loaded_with_uniform_columns():
    events = load_all_events()
    assert len(events) == 95                                              # 22 + 15 + 58 datasets
    assert events.groupby("farm")["event_label"].value_counts().to_dict() == {
        ("A", "anomaly"): 12, ("A", "normal"): 10, ("B", "normal"): 9, ("B", "anomaly"): 6,
        ("C", "normal"): 31, ("C", "anomaly"): 27}
    assert "asset" not in events.columns and events["asset_id"].notna().all()   # Farm A column renamed


def test_every_anomaly_event_is_categorised():
    table = taxonomy_table()
    assert len(table) == 45
    assert (table["primary_component"] != UNSPECIFIED).all()


def test_units_are_ascii_normalised():
    for farm in "ABC":
        units = load_features(farm)["unit"]
        assert not units.str.contains("�").any()
        assert all(u.isascii() for u in units)


# --------------------------------------------------------- documents and provenance

def test_all_doc_types_and_counts(docs):
    types = pd.Series([d.doc_type for d in docs]).value_counts().to_dict()
    assert set(types) == set(DOC_TYPES)
    assert types["fault_description"] == 45 and types["maintenance_record"] == 45
    validate_kb(docs)


def test_synthetic_labelling(docs):
    for d in docs:
        generated_type = d.doc_type in {"equipment_manual", "maintenance_record", "troubleshooting_procedure",
                                        "standard_summary"}
        assert d.synthetic is generated_type, d.doc_id
        if d.synthetic:
            assert d.provenance.source == "generated" and d.provenance.generator
            assert SYNTHETIC_MARKER in d.title and d.text.startswith(SYNTHETIC_MARKER)
        else:
            assert SYNTHETIC_MARKER not in d.text


def test_real_documents_point_to_their_source_rows(docs):
    for d in docs:
        if d.synthetic:
            continue
        path = PROJECT_ROOT / d.provenance.source_file
        assert path.is_file(), d.doc_id
        n_rows = len(pd.read_csv(path, sep=";", encoding="utf-8"))
        assert all(0 <= r < n_rows for r in d.provenance.source_rows), d.doc_id
    # A fault document's row really is that event, with the same description.
    d = next(x for x in docs if x.doc_id == "fault-c-47")
    events = load_events("C")
    row = events.iloc[d.provenance.source_rows[0]]
    assert row["event_id"] == 47 and row["asset_id"] == d.asset_id
    assert row["event_description"] in d.text


def test_maintenance_records_reference_real_events_but_are_synthetic(docs):
    real_ids = {(d.farm, d.event_id) for d in docs if d.doc_type == "fault_description"}
    maint = [d for d in docs if d.doc_type == "maintenance_record"]
    assert {(d.farm, d.event_id) for d in maint} == real_ids
    assert all("invented" in d.text for d in maint)


def test_hashes_detect_tampering(docs):
    tampered = [d if d.doc_id != "fault-a-68" else d.model_copy(update={"text": d.text + " edited"}) for d in docs]
    with pytest.raises(KBValidationError, match="content_sha256"):
        validate_kb(tampered)


def test_validation_rejects_bad_documents(docs):
    prov = Provenance(source="generated", generator="x", derived_by="y")
    with pytest.raises(ValueError, match="SYNTHETIC"):
        KBDocument(doc_id="bad-1", doc_type="equipment_manual", title="No marker title", text="x" * 30,
                   domain="wind", synthetic=True, provenance=prov)
    with pytest.raises(ValueError, match="source_file"):
        KBDocument(doc_id="bad-2", doc_type="fault_description", title="Real doc", text="x" * 30, domain="wind",
                   synthetic=False, provenance=Provenance(source="dataset", derived_by="y"))
    with pytest.raises(KBValidationError, match="duplicate doc_id"):
        validate_kb(docs + [docs[0]])
    with pytest.raises(KBValidationError, match="no documents of type"):
        validate_kb([d for d in docs if d.doc_type != "standard_summary"])


def test_build_is_deterministic(docs):
    again = build_documents()
    assert [d.model_dump() for d in again] == [d.model_dump() for d in docs]


# --------------------------------------------------------- storage and ingestion

def test_write_and_reload_roundtrip(docs, tmp_path):
    doc_path, chunk_path = write_kb(docs, tmp_path)
    reloaded = load_documents(doc_path)
    assert [d.model_dump() for d in reloaded] == [d.model_dump() for d in docs]
    assert all(d.content_sha256 == text_hash(d.text) for d in reloaded)
    chunks = [json.loads(line) for line in chunk_path.open(encoding="utf-8")]
    assert {c["metadata"]["doc_id"] for c in chunks} == {d.doc_id for d in docs}


def test_chunks_follow_ingestion_contract(docs):
    chunks = chunk_documents(docs)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert set(c) == {"chunk_id", "domain_id", "text", "metadata"} and c["domain_id"] == "knowledge_base"
        assert c["text"].startswith(c["metadata"]["title"])
        flat = flatten_chunk_metadata("knowledge_base", c["metadata"])      # what the vector stores receive
        assert all(isinstance(v, (str, int, float, bool)) for v in flat.values())
        assert flat["synthetic"] in (True, False)
    # Long documents are split, and every piece keeps the provenance fields.
    multi = [c for c in chunks if c["metadata"]["n_chunks"] > 1]
    assert multi and all("source" in c["metadata"] for c in multi)


def test_knowledge_base_domain_is_ingestible(docs, tmp_path, monkeypatch, fake_store, fake_generator):
    assert "knowledge_base" in list_domains()
    manifest = load_manifest("knowledge_base")
    assert "{context}" in manifest.prompt_template and "{question}" in manifest.prompt_template
    write_kb(docs, tmp_path / "knowledge_base")
    monkeypatch.setattr(main, "PROCESSED_ROOT", tmp_path)
    monkeypatch.setattr(main, "get_vector_store", lambda: fake_store)
    monkeypatch.setattr(main, "get_generator", lambda: fake_generator)
    client = TestClient(main.app)
    resp = client.post("/ingest/knowledge_base")
    assert resp.status_code == 200 and resp.json()["chunks_stored"] == len(chunk_documents(docs))
    hits = client.get("/query/knowledge_base", params={"q": "pitch battery voltage low", "top_k": 3}).json()["results"]
    assert any("pitch" in h["metadata"].get("component", "") for h in hits)
