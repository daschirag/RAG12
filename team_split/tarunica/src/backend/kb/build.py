"""B1 - Build, validate and store the GreenRAG-Sec knowledge base.

    python -m src.backend.kb.build

Outputs:
- data/processed/knowledge_base/documents.jsonl  one KBDocument per line (gitignored)
- data/processed/knowledge_base/chunks.jsonl     ingestion-format chunks for POST /ingest/knowledge_base
- results/kb/kb_summary.md, results/kb/fault_taxonomy.csv, results/kb/kb_build.json

The chunk file follows the existing ingestion contract
(``{"chunk_id", "domain_id", "text", "metadata"}``, see ingestion/pipeline.py),
so the existing /ingest endpoint and vector stores work unchanged. B1 uses a
simple paragraph-packing chunker; B2 compares chunking strategies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..evaluation.provenance import environment_info
from ..ingestion.pipeline import _write_jsonl
from .chunking import pack_paragraphs
from .content import MAINTENANCE_TEMPLATES, SOLAR_INVERTER_GUIDE, STANDARD_OVERVIEWS, WIND_COMPONENT_GUIDE
from .schema import SYNTHETIC_MARKER, KBDocument, Provenance, validate_kb
from .taxonomy import (
    COMPONENT_BY_KEY,
    COMPONENTS,
    FARMS,
    PLAN_FAULTS_VS_DATA,
    categorize,
    farm_dir,
    load_events,
    load_features,
    rel,
    sensors_for,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
KB_DOMAIN = "knowledge_base"
KB_DIR = PROJECT_ROOT / "data" / "processed" / KB_DOMAIN
RESULTS_DIR = PROJECT_ROOT / "results" / "kb"
MAX_CHUNK_CHARS = 1200
THIS = "src/backend/kb/build.py"
CONTENT = "src/backend/kb/content.py"
SYNTHETIC_PREAMBLE = (f"{SYNTHETIC_MARKER} Generated for the GreenRAG-Sec research prototype. This is not a real "
                      "{what}; its content was written for this project and must not be used for real operations.")


def _synthetic(doc_id, doc_type, title, what, body, domain, **extra) -> KBDocument:
    text = SYNTHETIC_PREAMBLE.format(what=what) + "\n\n" + body.strip()
    prov = Provenance(source="generated", generator=CONTENT, derived_by=THIS, notes=extra.pop("notes", ""))
    return KBDocument(doc_id=doc_id, doc_type=doc_type, title=f"{SYNTHETIC_MARKER} {title}", text=text,
                      domain=domain, synthetic=True, provenance=prov, **extra).with_hash()


def _real(doc_id, doc_type, title, body, source_file: Path, rows: list[int], **extra) -> KBDocument:
    prov = Provenance(source="dataset", source_file=rel(source_file), source_rows=sorted(int(r) for r in rows),
                      derived_by=THIS, notes=extra.pop("notes", ""))
    return KBDocument(doc_id=doc_id, doc_type=doc_type, title=title, text=body.strip(), domain="wind",
                      synthetic=False, provenance=prov, **extra).with_hash()


def _bullets(items) -> str:
    return "\n".join(f"- {i}" for i in items)


def _steps(items) -> str:
    return "\n".join(f"{n}. {i}" for n, i in enumerate(items, start=1))


def _duration(start: pd.Timestamp, end: pd.Timestamp) -> str:
    hours = (end - start) / pd.Timedelta("1h")
    return f"{hours / 24:.1f} days"


# ------------------------------------------------------------ real-data documents

def fault_description_docs(events: pd.DataFrame) -> list[KBDocument]:
    docs = []
    for _, e in events[events["event_label"] == "anomaly"].iterrows():
        primary, comps = categorize(e["event_description"])
        comp = COMPONENT_BY_KEY.get(primary)
        others = [COMPONENT_BY_KEY[c].label for c in comps[1:]]
        body = (
            f"Fault event {e.farm}-{e.event_id}: Wind Farm {e.farm}, turbine asset {e.asset_id}.\n"
            f"Label: anomaly (a period leading up to a logged failure).\n"
            f"Event window: {e.event_start:%Y-%m-%d %H:%M} to {e.event_end:%Y-%m-%d %H:%M} "
            f"({_duration(e.event_start, e.event_end)}). Timestamps are anonymised in the dataset.\n"
            f"Logged root-cause description (original wording): \"{e.event_description}\"\n\n"
            f"Component (keyword rules in src/backend/kb/taxonomy.py): {comp.label if comp else 'unspecified'}"
            + (f"; also mentions: {', '.join(others)}" if others else "") + ".\n"
            f"Default severity for this component (project assumption): {comp.default_severity if comp else 'unknown'}."
        )
        docs.append(_real(
            f"fault-{e.farm.lower()}-{e.event_id}", "fault_description",
            f"Fault event {e.farm}-{e.event_id}: {comp.label if comp else 'unspecified'} (asset {e.asset_id})",
            body, farm_dir(e.farm) / "event_info.csv", [e.source_row],
            component=primary if comp else None, components=comps, farm=e.farm, asset_id=int(e.asset_id),
            event_id=int(e.event_id), severity=comp.default_severity if comp else None,
            tags=["event", "anomaly"]))
    return docs


def farm_overview_docs(events: pd.DataFrame) -> list[KBDocument]:
    docs = []
    for farm in FARMS:
        ev = events[events["farm"] == farm]
        anomalies, normals = ev[ev["event_label"] == "anomaly"], ev[ev["event_label"] == "normal"]
        normal_lines = [f"- asset {r.asset_id}: {r.event_start:%Y-%m-%d} to {r.event_end:%Y-%m-%d} "
                        f"({_duration(r.event_start, r.event_end)}), event {r.event_id}"
                        for r in normals.sort_values(["asset_id", "event_start"]).itertuples()]
        counts = anomalies["event_description"].map(lambda d: categorize(d)[0]).value_counts()
        body = (
            f"Historical SCADA event overview for Wind Farm {farm}.\n"
            f"Events in event_info.csv: {len(ev)} ({len(anomalies)} anomaly, {len(normals)} normal) on "
            f"{ev['asset_id'].nunique()} turbines (asset ids {', '.join(str(a) for a in sorted(ev['asset_id'].unique()))}).\n"
            f"Event windows span {ev['event_start'].min():%Y-%m-%d} to {ev['event_end'].max():%Y-%m-%d} "
            "(anonymised timestamps). SCADA values are 10-minute statistics.\n\n"
            "Anomaly events by component: "
            + ", ".join(f"{COMPONENT_BY_KEY[k].label if k in COMPONENT_BY_KEY else k} {v}" for k, v in counts.items())
            + ".\n\nNormal-behaviour reference windows (no fault):\n" + "\n".join(normal_lines) + "\n\n"
            "Turbine status codes in the SCADA files (per the dataset README): 0 normal operation, 1 derated "
            "operation, 2 idling, 3 service, 4 downtime, 5 other; 0 and 2 count as normal operation."
        )
        docs.append(_real(f"scada-overview-{farm.lower()}", "scada_knowledge", f"Wind Farm {farm}: event overview",
                          body, farm_dir(farm) / "event_info.csv", list(ev["source_row"]), farm=farm,
                          tags=["overview", "normal_behaviour"]))
    return docs


def component_history_docs(events: pd.DataFrame) -> list[KBDocument]:
    docs = []
    anomalies = events[events["event_label"] == "anomaly"].copy()
    anomalies["comps"] = anomalies["event_description"].map(lambda d: categorize(d)[1])
    for comp in COMPONENTS:
        for farm in FARMS:
            sel = anomalies[(anomalies["farm"] == farm) & anomalies["comps"].map(lambda c: comp.key in c)]
            if sel.empty:
                continue
            days = (sel["event_end"] - sel["event_start"]) / pd.Timedelta("1D")
            lines = [f"- event {r.event_id} (asset {r.asset_id}, {r.event_start:%Y-%m-%d}, "
                     f"{_duration(r.event_start, r.event_end)}): \"{r.event_description}\""
                     for r in sel.sort_values("event_start").itertuples()]
            body = (f"Historical {comp.label.lower()} events in Wind Farm {farm}: {len(sel)} anomaly event(s) on "
                    f"{sel['asset_id'].nunique()} turbine(s). Event window length: min {days.min():.1f}, "
                    f"median {days.median():.1f}, max {days.max():.1f} days.\n" + "\n".join(lines))
            docs.append(_real(f"history-{comp.key}-{farm.lower()}", "scada_knowledge",
                              f"{comp.label}: event history, Wind Farm {farm}", body,
                              farm_dir(farm) / "event_info.csv", list(sel["source_row"]), component=comp.key,
                              components=[comp.key], farm=farm, severity=comp.default_severity,
                              tags=["history"]))
    return docs


def sensor_catalogue_docs() -> list[KBDocument]:
    docs = []
    for farm in FARMS:
        feats = load_features(farm)
        path = farm_dir(farm) / "feature_description.csv"
        used: set[int] = set()
        for comp in COMPONENTS:
            sel = sensors_for(comp, feats)
            if sel.empty:
                continue
            used |= set(sel["source_row"])
            lines = [f"- {r.description} [{r.unit or 'no unit'}] ({r.sensor_name}; {r.statistics_type})"
                     for r in sel.itertuples()]
            body = (f"SCADA signals related to the {comp.label.lower()} available in Wind Farm {farm} "
                    f"({len(sel)} sensors, 10-minute statistics):\n" + "\n".join(lines))
            docs.append(_real(f"sensors-{comp.key}-{farm.lower()}", "scada_knowledge",
                              f"{comp.label}: SCADA signals, Wind Farm {farm}", body, path, list(sel["source_row"]),
                              component=comp.key, components=[comp.key], farm=farm, tags=["sensors"]))
        rest = feats[~feats["source_row"].isin(used)]
        if not rest.empty:
            lines = [f"- {r.description} [{r.unit or 'no unit'}] ({r.sensor_name})" for r in rest.itertuples()]
            docs.append(_real(f"sensors-other-{farm.lower()}", "scada_knowledge",
                              f"Other SCADA signals (power, wind, environment), Wind Farm {farm}",
                              f"Other SCADA signals in Wind Farm {farm} ({len(rest)} sensors):\n" + "\n".join(lines),
                              path, list(rest["source_row"]), farm=farm, tags=["sensors"]))
    return docs


# ------------------------------------------------------------ synthetic documents

def _signals_line(comp_key: str) -> str:
    comp = COMPONENT_BY_KEY[comp_key]
    parts = []
    for farm in FARMS:
        sel = sensors_for(comp, load_features(farm))
        if not sel.empty:
            names = sorted(set(sel["description"]))[:6]
            parts.append(f"Farm {farm}: {'; '.join(names)}" + (" ..." if sel["description"].nunique() > 6 else ""))
    return ("Related SCADA signals in this dataset (real sensor names from feature_description.csv): "
            + (" | ".join(parts) if parts else "none identified") + ".")


def manual_docs() -> list[KBDocument]:
    docs = []
    for key, g in WIND_COMPONENT_GUIDE.items():
        comp = COMPONENT_BY_KEY[key]
        body = (f"Equipment manual section: {comp.label} (wind turbine).\n\nPurpose: {g['purpose']}\n\n"
                f"Typical warning signs:\n{_bullets(g['symptoms'])}\n\nCommon causes:\n{_bullets(g['causes'])}\n\n"
                f"{_signals_line(key)}\n\nSafety: {g['safety']}")
        docs.append(_synthetic(f"manual-wind-{key}", "equipment_manual", f"Manual: {comp.label}", "equipment manual",
                               body, "wind", component=key, components=[key], severity=comp.default_severity,
                               tags=["manual"]))
    for key, g in SOLAR_INVERTER_GUIDE.items():
        body = (f"Equipment manual section: PV inverter - {g['title']}.\n\nTypical warning signs:\n"
                f"{_bullets(g['symptoms'])}\n\nCommon causes:\n{_bullets(g['causes'])}\n\n"
                "Note: the solar dataset in this project has no inverter fault labels; this section exists because "
                "the project plan lists this fault type.")
        docs.append(_synthetic(f"manual-solar-{key}", "equipment_manual", f"Manual: PV inverter - {g['title']}",
                               "equipment manual", body, "solar", component=key, components=[key],
                               tags=["manual", "solar_inverter"]))
    return docs


def troubleshooting_docs() -> list[KBDocument]:
    docs = []
    for key, g in WIND_COMPONENT_GUIDE.items():
        comp = COMPONENT_BY_KEY[key]
        body = (f"Troubleshooting procedure: {comp.label} (wind turbine).\n\nChecks, in order:\n{_steps(g['checks'])}"
                f"\n\nCorrective actions:\n{_bullets(g['actions'])}\n\nSafety: {g['safety']}")
        docs.append(_synthetic(f"trouble-wind-{key}", "troubleshooting_procedure", f"Troubleshooting: {comp.label}",
                               "troubleshooting procedure", body, "wind", component=key, components=[key],
                               severity=comp.default_severity, tags=["troubleshooting"]))
    for key, g in SOLAR_INVERTER_GUIDE.items():
        body = (f"Troubleshooting procedure: PV inverter - {g['title']}.\n\nChecks, in order:\n{_steps(g['checks'])}"
                f"\n\nCorrective actions:\n{_bullets(g['actions'])}")
        docs.append(_synthetic(f"trouble-solar-{key}", "troubleshooting_procedure",
                               f"Troubleshooting: PV inverter - {g['title']}", "troubleshooting procedure", body,
                               "solar", component=key, components=[key], tags=["troubleshooting", "solar_inverter"]))
    return docs


def maintenance_docs(events: pd.DataFrame) -> list[KBDocument]:
    docs = []
    for _, e in events[events["event_label"] == "anomaly"].iterrows():
        primary, comps = categorize(e["event_description"])
        finding, action = MAINTENANCE_TEMPLATES[primary]
        comp = COMPONENT_BY_KEY[primary]
        visit = e["event_end"] + pd.Timedelta("1D")
        body = (f"Maintenance record SYN-WO-{e.farm}{e.event_id:03d} (synthetic work order).\n"
                f"Turbine: Wind Farm {e.farm}, asset {e.asset_id}. Visit date: {visit:%Y-%m-%d}.\n"
                f"Related real event: {e.farm}-{e.event_id} (window {e.event_start:%Y-%m-%d} to "
                f"{e.event_end:%Y-%m-%d}), logged as \"{e.event_description}\".\n"
                f"Component: {comp.label}.\n"
                f"Findings (invented for this synthetic record): {finding}\n"
                f"Actions (invented for this synthetic record): {action}")
        docs.append(_synthetic(f"maint-{e.farm.lower()}-{e.event_id}", "maintenance_record",
                               f"Maintenance record for event {e.farm}-{e.event_id} ({comp.label})",
                               "maintenance record", body, "wind", component=primary, components=comps, farm=e.farm,
                               asset_id=int(e.asset_id), event_id=int(e.event_id), severity=comp.default_severity,
                               tags=["maintenance"],
                               notes=f"event reference taken from {rel(farm_dir(e.farm) / 'event_info.csv')} "
                                     f"row {e.source_row}; findings and actions are invented"))
    return docs


def standard_docs() -> list[KBDocument]:
    docs = []
    for key, s in STANDARD_OVERVIEWS.items():
        body = (f"{s['name']} - {s['topic']}.\n\n{s['overview']}\n\nThis is an original short overview written for "
                "this project, not text from the standard. Consult the official standard for any requirement.")
        docs.append(_synthetic(f"std-{key}", "standard_summary", f"Standard overview: {s['name']} ({s['topic']})",
                               "standard or a summary of its content", body, s["domain"],
                               tags=["standard", s["name"]]))
    return docs


# --------------------------------------------------------------------- assembly

def build_documents() -> list[KBDocument]:
    events = pd.concat([load_events(f) for f in FARMS], ignore_index=True)
    docs = (fault_description_docs(events) + farm_overview_docs(events) + component_history_docs(events)
            + sensor_catalogue_docs() + manual_docs() + troubleshooting_docs() + maintenance_docs(events)
            + standard_docs())
    validate_kb(docs)
    return docs


def chunk_documents(docs: list[KBDocument], max_chars: int = MAX_CHUNK_CHARS) -> list[dict]:
    """Paragraph-packing chunker; returns records in the ingestion JSONL format.

    ``max_chars`` limits the body of a chunk; the document title is prefixed on top.
    """
    chunks = []
    for d in docs:
        pieces = pack_paragraphs(d.text, max_chars)
        for i, text in enumerate(pieces):
            meta = {"doc_id": d.doc_id, "doc_type": d.doc_type, "title": d.title, "domain": d.domain,
                    "synthetic": d.synthetic, "component": d.component, "components": ",".join(d.components),
                    "farm": d.farm, "asset_id": d.asset_id, "event_id": d.event_id, "severity": d.severity,
                    "source": d.provenance.source, "source_file": d.provenance.source_file,
                    "source_rows": ",".join(str(r) for r in d.provenance.source_rows) or None,
                    "doc_sha256": d.content_sha256, "chunk_index": i, "n_chunks": len(pieces)}
            # Every chunk starts with its document title so it stays interpretable on its own.
            chunks.append({"chunk_id": f"{d.doc_id}::{i}", "domain_id": KB_DOMAIN, "text": f"{d.title}\n{text}",
                           "metadata": {k: v for k, v in meta.items() if v is not None}})
    return chunks


def write_kb(docs: list[KBDocument], out_dir: Path = KB_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_path = out_dir / "documents.jsonl"
    with doc_path.open("w", encoding="utf-8") as handle:
        for d in docs:
            handle.write(d.model_dump_json() + "\n")
    chunk_path = out_dir / "chunks.jsonl"
    _write_jsonl(chunk_documents(docs), chunk_path)
    return doc_path, chunk_path


def load_documents(path: Path = KB_DIR / "documents.jsonl") -> list[KBDocument]:
    with path.open(encoding="utf-8") as handle:
        return [KBDocument.model_validate_json(line) for line in handle if line.strip()]


def write_report(docs: list[KBDocument], chunks_path: Path, results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    events = pd.concat([load_events(f) for f in FARMS], ignore_index=True)
    anomalies = events[events["event_label"] == "anomaly"].copy()
    cats = anomalies["event_description"].map(categorize)
    anomalies["primary_component"] = [c[0] for c in cats]
    anomalies["all_components"] = [";".join(c[1]) for c in cats]
    anomalies[["farm", "asset_id", "event_id", "event_start", "event_end", "primary_component", "all_components",
               "event_description", "source_row"]].to_csv(results_dir / "fault_taxonomy.csv", index=False,
                                                         encoding="utf-8")
    n_chunks = sum(1 for _ in chunks_path.open(encoding="utf-8"))
    table = pd.DataFrame([{"doc_type": d.doc_type, "synthetic": d.synthetic, "domain": d.domain} for d in docs])
    by_type = table.groupby(["doc_type", "synthetic"]).size().unstack(fill_value=0)
    comp_farm = anomalies.groupby(["primary_component", "farm"]).size().unstack(fill_value=0)
    env = environment_info()
    L = ["# B1 knowledge base build", "",
         f"Generated by `python -m src.backend.kb.build` at base commit {env['git_commit']}.",
         f"Documents: {len(docs)}; chunks: {n_chunks} (paragraph packing, max {MAX_CHUNK_CHARS} chars).", "",
         "## Documents by type", "", "| doc_type | real (from dataset) | synthetic |", "|---|---|---|"]
    for t, r in by_type.iterrows():
        L.append(f"| {t} | {int(r.get(False, 0))} | {int(r.get(True, 0))} |")
    L += ["", "## Fault taxonomy derived from event_info.csv (anomaly events, primary component)", "",
          "| component | default severity (assumption) | " + " | ".join(f"Farm {f}" for f in comp_farm.columns)
          + " | total |", "|---|---|" + "---|" * (len(comp_farm.columns) + 1)]
    for key, r in comp_farm.sort_values(list(comp_farm.columns), ascending=False).iterrows():
        comp = COMPONENT_BY_KEY.get(key)
        L.append(f"| {comp.label if comp else key} | {comp.default_severity if comp else '-'} | "
                 + " | ".join(str(int(v)) for v in r) + f" | {int(r.sum())} |")
    counts = events.groupby(["farm", "event_label"]).size().unstack(fill_value=0)
    L += ["", "Event counts per farm (event_info.csv): "
          + "; ".join(f"Farm {f}: {int(r['anomaly'])} anomaly, {int(r['normal'])} normal" for f, r in counts.iterrows())
          + f". Total anomaly events: {int(counts['anomaly'].sum())} (the dataset README states 44).", "",
          "## Plan fault list (solar inverter) vs real data (wind turbines)", "",
          "| plan fault type | what the real data contains |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in PLAN_FAULTS_VS_DATA]
    (results_dir / "kb_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (results_dir / "kb_build.json").write_text(json.dumps(
        {"documents": len(docs), "chunks": n_chunks, "environment": env,
         "doc_sha256": {d.doc_id: d.content_sha256 for d in docs}}, indent=2), encoding="utf-8")


def main() -> None:
    docs = build_documents()
    doc_path, chunk_path = write_kb(docs)
    write_report(docs, chunk_path)
    synthetic = sum(d.synthetic for d in docs)
    print(f"knowledge base: {len(docs)} documents ({len(docs) - synthetic} real, {synthetic} synthetic) -> {doc_path}")
    print(f"chunks -> {chunk_path}")


if __name__ == "__main__":
    main()
