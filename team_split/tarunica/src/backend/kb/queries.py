"""B2 - Labelled query set with relevance known by construction.

Each query is generated from specific knowledge-base documents, so the relevant
documents are known without manual judging. Relevance is graded:
2 = the document the question is about, 1 = a closely related document
(e.g. the maintenance record for a fault, or the manual next to a
troubleshooting procedure).

``query_class`` labels each query for the B3 query classifier:
"quantitative" asks for counts or specific values, "conceptual" for
explanations, procedures or where information lives.

Query wording is template-based and deliberately avoids copying document titles;
event queries use the turbine, farm and month rather than the logged text,
except the "symptom" type, which quotes part of the logged description (as an
operator pasting an alarm text would).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .content import SOLAR_INVERTER_GUIDE, STANDARD_OVERVIEWS, WIND_COMPONENT_GUIDE
from .schema import KBDocument
from .taxonomy import COMPONENT_BY_KEY


@dataclass(frozen=True)
class LabelledQuery:
    query_id: str
    text: str
    query_type: str
    query_class: str                             # "conceptual" | "quantitative"
    relevant: dict[str, int] = field(default_factory=dict)   # doc_id -> grade (1 or 2)


def _by_type(docs, doc_type):
    return [d for d in docs if d.doc_type == doc_type]


def build_queries(docs: list[KBDocument]) -> list[LabelledQuery]:
    ids = {d.doc_id: d for d in docs}
    faults = sorted(_by_type(docs, "fault_description"), key=lambda d: (d.farm, d.event_id))
    queries: list[LabelledQuery] = []

    # 1. Event by turbine and month (12): relevant = every fault of that asset starting in that month.
    starts = {d.doc_id: pd.Timestamp(d.text.split("Event window: ")[1][:10]) for d in faults}
    for d in faults[::4][:12]:
        month = starts[d.doc_id].strftime("%B %Y")
        same = [f for f in faults if f.farm == d.farm and f.asset_id == d.asset_id
                and starts[f.doc_id].strftime("%B %Y") == month]
        rel = {f.doc_id: 2 for f in same} | {f"maint-{f.farm.lower()}-{f.event_id}": 1 for f in same}
        queries.append(LabelledQuery(f"event-{d.doc_id}",
                                     f"What fault affected turbine {d.asset_id} in Wind Farm {d.farm} in {month}?",
                                     "event", "conceptual", rel))

    # 2. Symptom text pasted from the log (6): events with a unique description in Farm B/C.
    desc_counts = pd.Series([d.text.split("(original wording): ")[1].split("\n")[0].lower() for d in faults]).value_counts()
    picked = 0
    for d in faults:
        desc = d.text.split("(original wording): ")[1].split("\n")[0].strip('"')
        if d.farm == "A" or desc_counts[f'"{desc.lower()}"'] > 1 or len(desc.split()) < 5:
            continue
        snippet = " ".join(desc.split()[:8])
        queries.append(LabelledQuery(f"symptom-{d.doc_id}", f"The log says: {snippet}. What is this fault about?",
                                     "symptom", "conceptual",
                                     {d.doc_id: 2, f"maint-{d.farm.lower()}-{d.event_id}": 1}))
        picked += 1
        if picked == 6:
            break

    # 3. Component troubleshooting from a symptom (13).
    for key, g in WIND_COMPONENT_GUIDE.items():
        symptom = g["symptoms"][0]
        queries.append(LabelledQuery(f"trouble-{key}",
                                     f"A wind turbine shows {symptom}. What should the technician check first?",
                                     "troubleshooting", "conceptual",
                                     {f"trouble-wind-{key}": 2, f"manual-wind-{key}": 1}))

    # 4. Available SCADA signals (8).
    sensor_docs = [d for d in docs if d.doc_id.startswith("sensors-") and d.component][:: 3][:8]
    for d in sensor_docs:
        label = COMPONENT_BY_KEY[d.component].label.lower()
        queries.append(LabelledQuery(f"signals-{d.doc_id}",
                                     f"Which measurements can be used to monitor the {label} in Wind Farm {d.farm}?",
                                     "signals", "conceptual", {d.doc_id: 2}))

    # 5. Counts of past failures (6, quantitative).
    history = sorted(_by_type(docs, "scada_knowledge"), key=lambda d: d.doc_id)
    history = [d for d in history if d.doc_id.startswith("history-")][::3][:6]
    for d in history:
        label = COMPONENT_BY_KEY[d.component].label.lower()
        queries.append(LabelledQuery(f"count-{d.doc_id}",
                                     f"How many {label} failures were recorded in Wind Farm {d.farm}?",
                                     "count", "quantitative",
                                     {d.doc_id: 2, f"scada-overview-{d.farm.lower()}": 1}))

    # 6. Solar inverter questions (6).
    for key, g in SOLAR_INVERTER_GUIDE.items():
        queries.append(LabelledQuery(f"solar-{key}",
                                     f"In a PV plant we see {g['symptoms'][0]}. What could cause this?",
                                     "solar", "conceptual",
                                     {f"manual-solar-{key}": 2, f"trouble-solar-{key}": 1}))

    # 7. Standards (7).
    for key, s in STANDARD_OVERVIEWS.items():
        queries.append(LabelledQuery(f"std-{key}", f"Which standard deals with {s['topic'].lower()}?",
                                     "standard", "conceptual", {f"std-{key}": 2}))

    # 8. Maintenance follow-up (6).
    for d in faults[1::7][:6]:
        queries.append(LabelledQuery(f"maint-{d.doc_id}",
                                     f"What work was done at the maintenance visit after event {d.farm}-{d.event_id}?",
                                     "maintenance", "conceptual",
                                     {f"maint-{d.farm.lower()}-{d.event_id}": 2, d.doc_id: 1}))

    missing = sorted({doc_id for q in queries for doc_id in q.relevant if doc_id not in ids})
    if missing:
        raise ValueError(f"query relevance points to unknown documents: {missing}")
    return queries
