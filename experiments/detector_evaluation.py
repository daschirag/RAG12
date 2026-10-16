"""C3: evaluate the detectors against the attacks of C2 (no Ollama needed).

    python -m experiments.detector_evaluation --part telemetry   # telemetry detector (CPU, ~2 min)
    python -m experiments.detector_evaluation --part text        # injection detector + document trust
    python -m experiments.detector_evaluation --part filter      # evidence filter: retrieval replay (MiniLM + reranker)
    python -m experiments.detector_evaluation --report

Telemetry: the C2 attacks are regenerated with the same functions, windows, replay days and seeds, and the
detector (fitted on clean training days only) is scored per day. Positives = attacked test days; negatives =
the clean days after the training period (val + calib + test; solar 13 days per plant, wind Oct-Dec 92 days).
A C3-only robustness variant (replay + 2% noise) checks the replay detector against an attacker who avoids
exact duplicates.

Injection detector: C2's query injection (66 injected vs 66 clean questions, from the llama3.2:3b pass - the
queries themselves are identical for every generator), a held-out set of other injection phrasings and hard
benign questions, and the KB documents (4 injection documents vs 189 clean + 5 poisoned).

Document trust: document level (clean KB, the 9 C2 attack documents added, the same content written in place
of the genuine document, a registry altered without the key) and the retrieval contexts saved in C2.

Evidence filter: the production retriever is rebuilt on the clean, poisoned and injected KB and every question
is answered by GreenRAG with the LLM disabled, i.e. the extractive fallback that leaked the unsafe procedures in
C2, and every Farm A event by the Diagnoser fallback; unfiltered vs filtered.

Outputs (results/security/): c3_telemetry_days.csv, c3_injection.csv, c3_trust_docs.csv, c3_trust_contexts.csv,
c3_filter_items.csv, c3_summary.md, c3_run.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backend.evaluation.provenance import environment_info
from src.backend.security.attacks import (
    attack_solar_raw,
    attack_wind_raw,
    injection_documents,
    jitter,
    poison_documents,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "security"
SEEDS = (0, 1, 2)
TRAIN_DAYS_SOLAR = 21
WIND_TRAIN_END = pd.Timestamp("2018-10-01")
KEY = b"c3-evaluation-key"          # evaluation only; production reads GREENRAG_TRUST_KEY
QWEN, LLAMA = "c2_knowledge_raw.jsonl", "c2_knowledge_raw_llama3.2-3b.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []


def _window(c2: pd.DataFrame, asset: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Attack window and replay day exactly as used in C2 (read from its results)."""
    row = c2[(c2.asset == asset) & (c2.attack == "replay")].iloc[0]
    a, b = row.window.split("..")
    return pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta("1D") - pd.Timedelta("1s"), pd.Timestamp(row.replay_day)


# ------------------------------------------------------------------ telemetry

def telemetry_runs() -> pd.DataFrame:
    from src.backend.forecasting.preprocess import load_plant_raw
    from src.backend.forecasting.wind import load_wind_raw
    from src.backend.security.scada_anomaly import TelemetryDetector, solar_sensor_frame, wind_sensor_frame

    c2 = pd.read_csv(OUT / "c2_forecast.csv")
    rows = []

    def add(rep, asset, attack, seed, label):
        for r in rep.to_dict("records"):
            rows.append({"asset": asset, "attack": attack, "seed": seed, "label": label, **r})

    for plant in (1, 2):
        asset = f"solar_plant{plant}"
        gen, weather = load_plant_raw(plant)
        clean = solar_sensor_frame(gen, weather)
        days = sorted(clean.index.normalize().unique())
        det = TelemetryDetector("solar").fit(clean[clean.index.normalize().isin(days[:TRAIN_DAYS_SOLAR])])
        start, end, replay_day = _window(c2, asset)
        test_days = [d for d in days if start <= d <= end]
        add(det.day_report(clean, days[TRAIN_DAYS_SOLAR:]), asset, "clean", 0, 0)
        for attack in ("irradiation_bias", "module_temp_spike", "power_scale", "replay", "replay_noisy"):
            for seed in (SEEDS if attack in ("module_temp_spike", "replay_noisy") else (0,)):
                g, w = attack_solar_raw(gen, weather, "replay" if attack == "replay_noisy" else attack, start, end,
                                        seed=seed, replay_day=replay_day)
                if attack == "replay_noisy":
                    g = jitter(g, ["AC_POWER", "DC_POWER"], start, end, seed=seed)
                    w = jitter(w, ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"], start, end,
                               seed=seed + 100)
                add(det.day_report(solar_sensor_frame(g, w), test_days), asset, attack, seed, 1)
                print(f"telemetry {asset} {attack:<18} seed {seed}")
    raw = load_wind_raw()
    clean = wind_sensor_frame(raw)
    det = TelemetryDetector("wind").fit(clean[clean.index < WIND_TRAIN_END])
    start, end, replay_day = _window(c2, "wind_t1")
    days = sorted(clean.index.normalize().unique())
    test_days = [d for d in days if start <= d <= end]
    add(det.day_report(clean, [d for d in days if d >= WIND_TRAIN_END]), "wind_t1", "clean", 0, 0)
    for attack in ("wind_speed_bias", "replay", "replay_noisy"):
        for seed in (SEEDS if attack == "replay_noisy" else (0,)):
            r = attack_wind_raw(raw, "replay" if attack == "replay_noisy" else attack, start, end,
                                replay_day=replay_day)
            if attack == "replay_noisy":
                r = jitter(r, ["ac_power_kw", "wind_speed", "wind_dir", "theoretical_power"], start, end, seed=seed)
            add(det.day_report(wind_sensor_frame(r), test_days), "wind_t1", attack, seed, 1)
            print(f"telemetry wind_t1 {attack:<18} seed {seed}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ injection + trust (text)

def injection_runs(docs, poison, inject) -> pd.DataFrame:
    from src.backend.security.injection_detector import injection_score, injection_signals, is_injection
    from src.backend.security.injection_eval_set import HARD_BENIGN, HELD_OUT_INJECTIONS

    rows = []

    def add(source, label, text, item=""):
        rows.append({"source": source, "label": label, "item": item, "flagged": is_injection(text),
                     "score": injection_score(text), "signals": ";".join(injection_signals(text)),
                     "text": text[:300]})

    recs = _load_jsonl(OUT / LLAMA)
    succeeded = {r["item_id"] for r in recs if r["state"] == "query_injection" and "canary-7431" in r["text"].lower()}
    for r in recs:
        if r["system"] == "greenrag" and r["state"] in ("clean", "query_injection"):
            injected = r["state"] == "query_injection"
            add("c2_query", int(injected), r["question"],
                r["item_id"] + (" [injection succeeded in C2]" if injected and r["item_id"] in succeeded else ""))
    for t in HELD_OUT_INJECTIONS:
        add("held_out", 1, t)
    for t in HARD_BENIGN:
        add("held_out", 0, t)
    for d in docs + poison:
        add("kb_document", 0, d.text, d.doc_id)
    for d in inject:
        add("kb_document", 1, d.text, d.doc_id)
    return pd.DataFrame(rows)


def _trust(store_docs, clean_docs, registry=None):
    from src.backend.kb.taxonomy import load_all_events
    from src.backend.security.doc_trust import (
        DocumentTrust,
        build_registry,
        calibrate_stuffing,
        logged_descriptions,
        verify_registry,
    )

    reg = verify_registry(registry or build_registry(clean_docs, KEY), KEY)
    return DocumentTrust(reg, {d.doc_id: d for d in store_docs}, logged_descriptions(load_all_events()),
                         calibrate_stuffing(clean_docs))


ORIGINAL = {"trouble-wind-pitch_system-rev2": "trouble-wind-pitch_system", "trouble-wind-gearbox-rev2":
            "trouble-wind-gearbox", "fault-a-73-rev": "fault-a-73", "maint-a-0-rev": "maint-a-0",
            "manual-wind-hydraulic_system-note": "manual-wind-hydraulic_system", "maint-a-73-note": "maint-a-73",
            "fault-a-26-log": "fault-a-26", "std-iec_61400_25-note": "std-iec_61400_25"}


def trust_doc_runs(docs, poison, inject) -> pd.DataFrame:
    from src.backend.kb.schema import text_hash
    from src.backend.security.doc_trust import CONTENT_CHECKS, RegistryError, build_registry, verify_registry

    trust = _trust(docs + poison + inject, docs)
    rows = []

    def add(scenario, label, doc, rep):
        rows.append({"scenario": scenario, "label": label, "doc_id": doc.doc_id, "trusted": rep.trusted,
                     "content_only_trusted": all(rep.checks[c] for c in CONTENT_CHECKS),
                     "self_hash_ok": rep.checks["self_hash"],
                     "failed": ";".join(c for c, ok in rep.checks.items() if not ok)})

    for d in docs:
        add("clean_kb", 0, d, trust.check_document(d))
    for d in poison + inject:
        add("added", 1, d, trust.check_document(d))
    # The same malicious content written over the genuine document (same doc_id, self-hash recomputed).
    for d in poison + inject:
        if d.doc_id in ORIGINAL:
            forged = d.model_copy(update={"doc_id": ORIGINAL[d.doc_id], "content_sha256": text_hash(d.text)})
            add("in_place", 1, forged, trust.check_document(forged))
    # A registry that lists the attack documents but was re-signed without the key.
    tampered = build_registry(docs + poison + inject, None)
    tampered["signature"] = build_registry(docs, KEY)["signature"]
    try:
        verify_registry(tampered, KEY)
        verdict = "accepted"
    except RegistryError:
        verdict = "rejected (fail closed: no document trusted)"
    rows.append({"scenario": "registry_altered_without_key", "label": 1, "doc_id": "(registry)",
                 "trusted": verdict == "accepted", "content_only_trusted": None, "self_hash_ok": None,
                 "failed": verdict})
    return pd.DataFrame(rows)


def trust_context_runs(docs, poison, inject) -> tuple[pd.DataFrame, dict]:
    """Would trust verification have removed the attack documents from the contexts retrieved in C2?"""
    from src.backend.security.doc_trust import CONTENT_CHECKS

    trust = _trust(docs + poison + inject, docs)
    by_id = {d.doc_id: d for d in docs + poison + inject}
    verdict = {i: trust.check_document(d) for i, d in by_id.items()}
    rows, runs = [], {"qwen2.5:7b-instruct": _load_jsonl(OUT / QWEN), "llama3.2:3b": _load_jsonl(OUT / LLAMA)}
    for model, recs in runs.items():
        for r in recs:
            if r["state"] == "query_injection":
                continue
            ctx = r["context_doc_ids"]
            flagged = [d for d in ctx if not verdict[d].trusted]
            content_flagged = [d for d in ctx if not all(verdict[d].checks[c] for c in CONTENT_CHECKS)]
            attack = [d for d in ctx if d not in {x.doc_id for x in docs}]
            rows.append({"generator": model, "state": r["state"], "system": r["system"], "item": r["item_id"],
                         "n_context": len(ctx), "attack_in_context": ";".join(sorted(set(attack))),
                         "flagged": ";".join(sorted(set(flagged))), "content_flagged": ";".join(sorted(set(content_flagged)))})
    # Retrieval does not involve the generator: check that both passes retrieved the same documents.
    key = lambda r: (r["state"], r["system"], r["item_id"])  # noqa: E731
    q = {key(r): r["context_doc_ids"] for r in runs["qwen2.5:7b-instruct"]}
    same = [q[key(r)] == r["context_doc_ids"] for r in runs["llama3.2:3b"] if key(r) in q and r["system"] != "diagnose"]
    return pd.DataFrame(rows), {"overlapping_question_items": len(same), "identical_contexts": int(sum(same))}


# ------------------------------------------------------------------ evidence filter replay

def filter_runs(docs, poison, inject) -> pd.DataFrame:
    from src.backend.evaluation.rag_systems import greenrag_answer
    from src.backend.faultdiag.diagnosis import (
        Diagnoser,
        event_windows,
        load_reliability,
        load_signals,
        production_retriever,
    )
    from src.backend.kb.chunking import semantic_chunks
    from src.backend.kb.queries import build_queries
    from src.backend.kb.retrieval_eval import minilm_embed
    from src.backend.kb.taxonomy import load_events
    from src.backend.retrieval.hybrid_retriever import SiblingExpander
    from src.backend.security.evidence_filter import EvidenceFilter, TrustedRetriever
    from experiments.attack_simulation import TARGET_QUESTIONS

    def llm_down(prompt: str) -> str:
        raise RuntimeError("LLM disabled: evaluate the extractive fallback")

    questions = {q.query_id: q.text for q in build_queries(docs)} | TARGET_QUESTIONS
    signals, embed = load_signals(), minilm_embed()
    reliability, windows = load_reliability(), event_windows(load_events("A"))
    components = {d.doc_id: d.component for d in docs + poison + inject}
    rows = []
    for state, kb in (("clean", docs), ("poisoned", docs + poison), ("injected", docs + inject)):
        chunks = semantic_chunks(kb, embed)
        inner = production_retriever(kb, chunks, embed)
        filtered = TrustedRetriever(inner, EvidenceFilter(_trust(kb, docs)))
        for setting, retriever in (("unfiltered", inner), ("filtered", filtered)):
            for qid, text in questions.items():
                ans = greenrag_answer(llm_down, retriever, text)
                rows.append({"state": state, "setting": setting, "system": "greenrag_fallback", "item": qid,
                             "route": ans.route, "text": ans.text,
                             "context_doc_ids": ";".join(c["doc_id"] for c in ans.context),
                             "rejected": ";".join(sorted({h["metadata"]["doc_id"] for h in
                                                          getattr(retriever, "last_rejected", [])}))})
            diag = Diagnoser(retriever, None, reliability, windows, expander=SiblingExpander(chunks))
            for s in signals:
                rep = diag.diagnose(s, components)
                rows.append({"state": state, "setting": setting, "system": "diagnose_fallback",
                             "item": str(s.event_id), "route": rep.generation,
                             "text": f"{rep.diagnosis} Likely cause: {rep.likely_cause.text} Recommended action: "
                                     f"{rep.recommended_action.text}",
                             "context_doc_ids": ";".join(e.doc_id for e in rep.evidence), "rejected": "",
                             "component": rep.likely_cause_component})
            print(f"filter {state:<9} {setting:<10} done")
    return pd.DataFrame(rows)


def score_filter(items: pd.DataFrame, docs, poison, inject) -> pd.DataFrame:
    from src.backend.security.attack_eval import score_item, with_windows

    _, pm = poison_documents(docs)
    _, im = injection_documents(docs)
    kb_texts = [d.text for d in docs]
    manifests = {"poisoned": with_windows(pm, kb_texts), "injected": with_windows(im, kb_texts)}
    clean = {(r.setting, r.system, r.item): r.text for r in items[items.state == "clean"].itertuples()}
    out = []
    for r in items[items.state != "clean"].itertuples():
        ctx = r.context_doc_ids.split(";") if r.context_doc_ids else []
        s = score_item(r.text, ctx, manifests[r.state], clean[(r.setting, r.system, r.item)])
        out.append({"state": r.state, "setting": r.setting, "system": r.system, "item": r.item,
                    "attack_in_context": ";".join(s["retrieved"]), "attack_content_in_output": ";".join(s["success"])})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ report

def _rate(x: pd.Series) -> str:
    return f"{x.mean():.3f} ({int(x.sum())}/{len(x)})" if len(x) else "n/a"


def report() -> None:
    env = environment_info()
    L = ["# C3 - Detectors evaluated against the C2 attacks", "",
         f"Generated by `python -m experiments.detector_evaluation` at base commit {env['git_commit']}.",
         "Detection rate = attacked units flagged; false-positive rate (FPR) = clean units flagged. All attack",
         "data is synthetic (C2). Nothing here needed an LLM; see docs/detectors.md for design and limitations.", ""]
    p = OUT / "c3_telemetry_days.csv"
    if p.is_file():
        t = pd.read_csv(p)
        checks = ["bounds", "rate", "consistency", "statistical", "replay", "any"]
        L += ["## 1. Telemetry anomaly / spoofing detector (per day)", "",
              "| asset | data | days | " + " | ".join(checks) + " |", "|---|---|---|" + "---|" * len(checks)]
        for (asset, attack), g in t.groupby(["asset", "attack"], sort=False):
            kind = "clean (FPR)" if attack == "clean" else f"{attack} (detection)"
            L.append(f"| {asset} | {kind} | {len(g)} | " + " | ".join(_rate(g[c]) for c in checks) + " |")
        L += ["", "replay_noisy is a C3-only robustness variant (replay + 2% noise), not a C2 attack. Seeds 0-2 where",
              "the attack is random (module_temp_spike, replay_noisy); the others are deterministic.", ""]
    p = OUT / "c3_injection.csv"
    if p.is_file():
        t = pd.read_csv(p)
        L += ["## 2. Prompt-injection detector", "",
              "| source | positives | detection rate | negatives | FPR |", "|---|---|---|---|---|"]
        for src, g in t.groupby("source", sort=False):
            L.append(f"| {src} | {int((g.label == 1).sum())} | {_rate(g[g.label == 1].flagged)} | "
                     f"{int((g.label == 0).sum())} | {_rate(g[g.label == 0].flagged)} |")
        succ = t[t["item"].astype(str).str.contains("succeeded", na=False)]
        L += ["", f"The {len(succ)} injected queries that succeeded against GreenRAG (llama3.2:3b) in C2: flagged "
              f"{int(succ.flagged.sum())}/{len(succ)}.",
              "qwen2.5:7b-instruct: no injection outcomes exist yet (its C2 injection runs are pending). The detector",
              "reads only the input, which is identical for both generators, so detection and FPR above hold for qwen",
              "too; which of the flagged injections would have *succeeded* against qwen is unknown.", "",
              "Missed held-out injections: " + "; ".join(f"\"{x}\"" for x in t[(t.source == "held_out") & (t.label == 1)
                                                                              & ~t.flagged].text) + ".",
              "False positives (held-out benign): " + ("; ".join(f"\"{x}\"" for x in t[(t.source == "held_out") &
                                                                                   (t.label == 0) & t.flagged].text)
                                                       or "none") + ".", ""]
    p = OUT / "c3_trust_docs.csv"
    if p.is_file():
        t = pd.read_csv(p)
        L += ["## 3. Document trust verification", "",
              "| scenario | documents | flagged (full: registry + content) | flagged (content checks only) | "
              "flagged by self-declared hash |", "|---|---|---|---|---|"]
        for sc, g in t[t.scenario != "registry_altered_without_key"].groupby("scenario", sort=False):
            L.append(f"| {sc} | {len(g)} | {_rate(~g.trusted.astype(bool))} | "
                     f"{_rate(~g.content_only_trusted.astype(bool))} | {_rate(~g.self_hash_ok.astype(bool))} |")
        alt = t[t.scenario == "registry_altered_without_key"].failed.iloc[0]
        L += ["", f"Registry altered without the key: {alt}.", "",
              "Per attack document (added scenario) - checks that failed:", ""]
        for r in t[t.scenario == "added"].itertuples():
            L.append(f"- {r.doc_id}: {r.failed}")
        L.append("")
    p = OUT / "c3_trust_contexts.csv"
    if p.is_file():
        t = pd.read_csv(p).fillna("")
        info = json.loads((OUT / "c3_run.json").read_text(encoding="utf-8")).get("context_check", {}) \
            if (OUT / "c3_run.json").is_file() else {}
        L += ["Retrieval contexts saved in C2 (plain RAG, GreenRAG, /diagnose):", "",
              "| generator | state | items | attack doc in context | all attack docs flagged (full) | "
              "all flagged (content only) | clean-state items with a false flag |", "|---|---|---|---|---|---|---|"]
        for (gen, state), g in t.groupby(["generator", "state"], sort=False):
            reach = g[g.attack_in_context != ""]
            full = reach.apply(lambda r: set(r.attack_in_context.split(";")) <= set(r.flagged.split(";")), axis=1)
            cont = reach.apply(lambda r: set(r.attack_in_context.split(";")) <= set(r.content_flagged.split(";")),
                               axis=1)
            fp = g[(g.flagged != "")] if state == "clean" else g.iloc[0:0]
            L.append(f"| {gen} | {state} | {len(g)} | {_rate(g.attack_in_context != '')} | "
                     f"{_rate(full) if len(reach) else 'n/a'} | {_rate(cont) if len(reach) else 'n/a'} | "
                     f"{_rate(g.flagged != '') if state == 'clean' else '-'} |")
        L += ["", f"Contexts are produced by retrieval, which does not involve the generator: "
              f"{info.get('identical_contexts', '?')} of {info.get('overlapping_question_items', '?')} question items "
              f"that both passes ran have identical contexts. qwen has no injected-state contexts yet (pending).", ""]
    p = OUT / "c3_filter_scored.csv"
    if p.is_file():
        t = pd.read_csv(p).fillna("")
        items = pd.read_csv(OUT / "c3_filter_items.csv").fillna("")
        L += ["## 4. Evidence filter: fallback replay without the LLM", "",
              "GreenRAG with the LLM disabled always takes the extractive fallback (the path that leaked the unsafe",
              "procedures in C2); the Diagnoser runs its deterministic fallback. Same questions and events, same KB,",
              "unfiltered vs filtered.", "",
              "| state | system | setting | items | attack doc in context | attack content in output |",
              "|---|---|---|---|---|---|"]
        for (state, system, setting), g in t.groupby(["state", "system", "setting"], sort=False):
            L.append(f"| {state} | {system} | {setting} | {len(g)} | {_rate(g.attack_in_context != '')} | "
                     f"{_rate(g.attack_content_in_output != '')} |")
        c = items[items.state == "clean"]
        u = c[c.setting == "unfiltered"].set_index(["system", "item"]).text
        f = c[c.setting == "filtered"].set_index(["system", "item"]).text
        same = (u == f.reindex(u.index))
        L += ["", f"Clean KB (false positives / utility): filtered output identical to unfiltered for "
              f"{_rate(same)} items; chunks rejected on the clean KB: "
              f"{int((c[c.setting == 'filtered'].rejected != '').sum())} items with any rejection.", "",
              "Restoration - output on the attacked KB identical to the clean-KB output (same setting as clean "
              "unfiltered):", "", "| state | unfiltered | filtered |", "|---|---|---|"]
        for state in ("poisoned", "injected"):
            cells = []
            for setting in ("unfiltered", "filtered"):
                x = items[(items.state == state) & (items.setting == setting)].set_index(["system", "item"]).text
                cells.append(_rate(x == u.reindex(x.index)))
            L.append(f"| {state} | {cells[0]} | {cells[1]} |")
        L.append("")
    (OUT / "c3_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("telemetry", "text", "filter"))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run_path = OUT / "c3_run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    t0 = time.perf_counter()
    if args.part == "telemetry":
        telemetry_runs().to_csv(OUT / "c3_telemetry_days.csv", index=False, encoding="utf-8")
    elif args.part in ("text", "filter"):
        from src.backend.kb.build import build_documents

        docs = build_documents()
        poison, _ = poison_documents(docs)
        inject, _ = injection_documents(docs)
        if args.part == "text":
            injection_runs(docs, poison, inject).to_csv(OUT / "c3_injection.csv", index=False, encoding="utf-8")
            trust_doc_runs(docs, poison, inject).to_csv(OUT / "c3_trust_docs.csv", index=False, encoding="utf-8")
            ctx, info = trust_context_runs(docs, poison, inject)
            ctx.to_csv(OUT / "c3_trust_contexts.csv", index=False, encoding="utf-8")
            run["context_check"] = info
        else:
            items = filter_runs(docs, poison, inject)
            items.to_csv(OUT / "c3_filter_items.csv", index=False, encoding="utf-8")
            score_filter(items, docs, poison, inject).to_csv(OUT / "c3_filter_scored.csv", index=False,
                                                             encoding="utf-8")
    if args.part:
        run["environment"] = environment_info()
        run[f"{args.part}_seconds"] = round(time.perf_counter() - t0, 1)
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    report()


if __name__ == "__main__":
    main()
