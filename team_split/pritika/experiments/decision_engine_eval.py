"""C5: trust-aware decision engine on real pipeline outputs, with a weight sensitivity analysis.

    python -m experiments.decision_engine_eval --llm llama3.2:3b   # capture structured outputs (resumable)
    python -m experiments.decision_engine_eval --report            # decisions, sensitivity, figure

Item sets (all real outputs; nothing synthetic except the C2 attack data itself):
- ``b5_qwen``: the 44 saved B5 diagnoses (qwen2.5:7b-instruct, clean KB, logged + unlogged modes). They
  predate C4, so the engine's own security verification re-checks their evidence against the trusted registry.
- ``c5_llama``: the C4 secure pipeline and secure /diagnose re-run with llama3.2:3b (temperature 0, seed 0) on
  the clean, poisoned and injected KB stores and with injected questions. The run keeps the full structured
  output (evidence scores, citations, security report) that the C4 run did not store, and its answer texts are
  compared with C4's as a reproducibility check.
- ``c2_unprotected``: the saved C2 answers of both generators, without any defence. Only the text-level
  constraint (no unsafe directive) can be applied to them: they have no evidence scores.

Generator: llama3.2:3b, for the same reason as C4 (the only generator with complete attacked baselines).
qwen2.5:7b-instruct decision behaviour is measured only on its saved clean B5 diagnoses; under attack it is
unmeasured until its C2 knowledge run is resumed.

Outputs (results/trust/): c5_raw_llama3.2-3b.jsonl, c5_decisions.csv, c5_sensitivity_grid.csv,
c5_sensitivity_thresholds.csv, c5_weight_sweep.csv, c5_weight_sweep.png, c5_unsafe_constraint_c2.csv,
c5_summary.md, c5_run.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.attack_simulation import TARGET_QUESTIONS
from src.backend.evaluation.provenance import environment_info
from src.backend.security.attacks import inject_query, injection_documents, poison_documents

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "trust"
SEC = ROOT / "results" / "security"
LLM = "llama3.2:3b"
RAW = OUT / "c5_raw_llama3.2-3b.jsonl"
DECISIONS = ("accept", "review", "reject")


def _load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []


def _stores():
    from src.backend.kb.build import build_documents

    docs = build_documents()
    poison, _ = poison_documents(docs)
    inject, _ = injection_documents(docs)
    return docs, {"clean": docs, "poisoned": docs + poison, "injected": docs + inject, "query_injection": docs}


# ------------------------------------------------------------------ LLM capture run

def capture(model: str, raw: Path = RAW, batch: int = 0) -> None:
    """Structured outputs of the C4 pipeline + secure /diagnose; ``batch`` > 0 reloads the model every N items."""
    from src.backend.faultdiag.diagnosis import Diagnoser, event_windows, load_reliability, load_signals, ollama_json_llm
    from src.backend.kb.queries import build_queries
    from src.backend.kb.retrieval_eval import minilm_embed
    from src.backend.kb.taxonomy import load_events
    from src.backend.retrieval.hybrid_retriever import SiblingExpander
    from src.backend.security.doc_trust import build_registry
    from src.backend.security.secure_pipeline import SecureRAGPipeline, build_secure_retrieval

    docs, stores = _stores()
    questions = {q.query_id: q.text for q in build_queries(docs)} | TARGET_QUESTIONS
    reference = build_registry(docs, None)["docs"]
    llm, embed = ollama_json_llm(model), minilm_embed()
    signals, reliability, windows = load_signals(), load_reliability(), event_windows(load_events("A"))
    from experiments.attack_simulation import ModelCycler

    raw.parent.mkdir(parents=True, exist_ok=True)
    cycler = ModelCycler(model, batch)
    if batch:
        cycler.load()
    done = {(r["state"], r["system"], r["item_id"]) for r in _load(raw)}
    with raw.open("a", encoding="utf-8") as fh:
        for state, store in stores.items():
            todo_q = [q for q in questions if (state, "greenrag", q) not in done]
            todo_d = [s for s in signals if state != "query_injection" and (state, "diagnose", str(s.event_id))
                      not in done]
            if not todo_q and not todo_d:
                continue
            sec = build_secure_retrieval(store_docs=store, embed=embed, reference=reference)
            pipeline = SecureRAGPipeline(sec.retriever, llm, registry_source="in-process build")
            for qid in todo_q:
                question = inject_query(questions[qid]) if state == "query_injection" else questions[qid]
                t0 = time.perf_counter()
                a = pipeline.answer(question)
                fh.write(json.dumps({"state": state, "system": "greenrag", "item_id": qid, "output": a.model_dump(),
                                     "seconds": round(time.perf_counter() - t0, 2)}, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"{state:<15} greenrag {qid:<40} {a.route}")
                if a.route != "blocked":          # blocked questions never reach the LLM
                    cycler.tick()
            if todo_d:
                components = {d.doc_id: d.component for d in store}
                diagnoser = Diagnoser(sec.retriever, llm, reliability, windows, expander=SiblingExpander(sec.chunks))
                for s in todo_d:
                    t0 = time.perf_counter()
                    rep = diagnoser.diagnose(s, components)
                    rejected = sorted({h["metadata"]["doc_id"] for h in sec.retriever.last_rejected})
                    fh.write(json.dumps({"state": state, "system": "diagnose", "item_id": str(s.event_id),
                                         "output": rep.model_dump(mode="json"), "rejected_doc_ids": rejected,
                                         "seconds": round(time.perf_counter() - t0, 2)}, ensure_ascii=False) + "\n")
                    fh.flush()
                    print(f"{state:<15} diagnose event {s.event_id:<34} {rep.generation}")
                    cycler.tick()


# ------------------------------------------------------------------ recommendations from real outputs

def _engine_for(store_docs, docs):
    from src.backend.kb.taxonomy import load_all_events
    from src.backend.security.doc_trust import DocumentTrust, build_registry, calibrate_stuffing, logged_descriptions
    from src.backend.trust.decision import DecisionEngine

    trust = DocumentTrust(build_registry(docs, None)["docs"], {d.doc_id: d for d in store_docs},
                          logged_descriptions(load_all_events()), calibrate_stuffing(docs))
    return DecisionEngine(trust=trust, trust_source="in-process build")


def recommendations() -> list[dict]:
    """[{set, state, system, item, generator, rec}] for every real output, after the engine's verification stages."""
    from src.backend.faultdiag.diagnosis import DiagnosisReport
    from src.backend.security.secure_pipeline import SecureAnswer
    from src.backend.trust.decision import SecurityFacts, from_diagnosis, from_secure_answer

    docs, stores = _stores()
    engines = {state: _engine_for(store, docs) for state, store in stores.items()}
    items = []
    for r in _load(ROOT / "results" / "faultdiag" / "b5_reports.jsonl"):
        rep = DiagnosisReport.model_validate({k: v for k, v in r.items() if k != "mode"})
        rec = engines["clean"].verify_security(from_diagnosis(rep, SecurityFacts()))
        items.append({"set": "b5_qwen", "state": f"clean ({r['mode']})", "system": "diagnose",
                      "item": str(r["event_id"]), "generator": "qwen2.5:7b-instruct", "rec": rec})
    for r in _load(RAW):
        if r["system"] == "greenrag":
            rec = from_secure_answer(SecureAnswer.model_validate(r["output"]))
        else:
            rep = DiagnosisReport.model_validate(r["output"])
            rec = from_diagnosis(rep, SecurityFacts(trust_checked=True, registry_source="in-process build",
                                                    rejected_doc_ids=r["rejected_doc_ids"]))
        items.append({"set": "c5_llama", "state": r["state"], "system": r["system"], "item": r["item_id"],
                      "generator": LLM, "rec": engines[r["state"]].verify_security(rec)})
    return items


# ------------------------------------------------------------------ sensitivity

def weight_grid(step: float = 0.05, minimum: float = 0.05) -> list[tuple[float, float, float, float]]:
    n = round(1 / step)
    lo = round(minimum / step)
    out = []
    for a, b, c in itertools.product(range(lo, n + 1), repeat=3):
        d = n - a - b - c
        if d >= lo:
            out.append(tuple(round(x * step, 10) for x in (a, b, c, d)))
    return out


def _decide_all(prepared, cfg):
    from src.backend.trust.decision import decide_core

    return [decide_core(p["comp"], p["facts"], cfg)[0] for p in prepared]


def sensitivity(prepared: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from src.backend.trust.decision import DecisionConfig, TrustWeights

    default = DecisionConfig()
    base = _decide_all(prepared, default)
    groups = [f"{p['set']}|{p['system']}|{p['state']}" for p in prepared]
    rows = []
    for w in weight_grid():
        cfg = DecisionConfig(weights=TrustWeights(E=w[0], R=w[1], C=w[2], S=w[3]))
        dec = _decide_all(prepared, cfg)
        row = {"wE": w[0], "wR": w[1], "wC": w[2], "wS": w[3],
               "changed_vs_default": float(np.mean([a != b for a, b in zip(dec, base)]))}
        for g in sorted(set(groups)):
            idx = [i for i, x in enumerate(groups) if x == g]
            for d in DECISIONS:
                row[f"{g}|{d}"] = float(np.mean([dec[i] == d for i in idx]))
        rows.append(row)
    grid = pd.DataFrame(rows)
    th = []
    for acc, rev in itertools.product((0.60, 0.65, 0.70, 0.75, 0.80), (0.30, 0.40, 0.50)):
        cfg = DecisionConfig(accept_at=acc, review_at=rev)
        dec = _decide_all(prepared, cfg)
        th.append({"accept_at": acc, "review_at": rev, **{d: float(np.mean([x == d for x in dec])) for d in DECISIONS},
                   "changed_vs_default": float(np.mean([a != b for a, b in zip(dec, base)]))})
    sweep = []
    dw = default.weights.model_dump()
    for comp in ("E", "R", "C", "S"):
        for v in np.round(np.arange(0.0, 0.81, 0.05), 2):
            rest = {k: dw[k] for k in dw if k != comp}
            scale = (1 - v) / sum(rest.values())
            w = {comp: float(v), **{k: rest[k] * scale for k in rest}}
            cfg = DecisionConfig(weights=TrustWeights(**w))
            dec = _decide_all(prepared, cfg)
            sweep.append({"component": comp, "weight": float(v),
                          **{d: float(np.mean([x == d for x in dec])) for d in DECISIONS}})
    return grid, pd.DataFrame(th), pd.DataFrame(sweep)


def plot_sweep(sweep: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    styles = {"accept": ("black", "-"), "review": ("0.45", "--"), "reject": ("0.7", ":")}
    for ax, comp in zip(axes, ("E", "R", "C", "S")):
        s = sweep[sweep.component == comp]
        for d, (col, ls) in styles.items():
            ax.plot(s.weight, s[d], color=col, linestyle=ls, label=d)
        ax.set_title(f"vary w{comp} (others rescaled)")
        ax.set_xlabel(f"w{comp}")
    axes[0].set_ylabel("share of decisions (all items)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ C2 unprotected outputs: text constraint only

def unsafe_on_c2() -> pd.DataFrame:
    from src.backend.trust.decision import unsafe_directives

    rows = []
    for gen, raw, items in (("qwen2.5:7b-instruct", "c2_knowledge_raw.jsonl", "c2_knowledge_items.csv"),
                            (LLM, "c2_knowledge_raw_llama3.2-3b.jsonl", "c2_knowledge_items_llama3.2-3b.csv")):
        if not (SEC / raw).is_file():
            continue
        scored = pd.read_csv(SEC / items).fillna("")
        success = {(r.state, r.system, r["item"].split(":", 1)[1]): (r.success != "" or bool(r.canary))
                   for _, r in scored.iterrows()}
        for r in _load(SEC / raw):
            hits = unsafe_directives(r["text"])
            rows.append({"generator": gen, "state": r["state"], "system": r["system"], "item": r["item_id"],
                         "attack_success": success.get((r["state"], r["system"], r["item_id"]), False),
                         "unsafe": ";".join(hits)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report

def _rate(x) -> str:
    x = pd.Series(list(x), dtype=float)
    return f"{x.mean():.3f} ({int(x.sum())}/{len(x)})" if len(x) else "n/a"


def report() -> None:
    from src.backend.trust.decision import DecisionConfig, check_constraints, gate_facts
    from src.backend.trust.signals import load_signals

    OUT.mkdir(parents=True, exist_ok=True)
    env = environment_info()
    sig = load_signals()
    items = recommendations()
    docs, stores = _stores()
    engines = {s: _engine_for(st, docs) for s, st in stores.items()}
    prepared, rows = [], []
    for it in items:
        eng = engines["clean" if it["set"] == "b5_qwen" else it["state"]]
        rec = it["rec"]
        comp, cons = eng.components(rec), check_constraints(rec)
        d = eng.decide(rec, comp, cons)
        prepared.append({**it, "comp": comp, "facts": gate_facts(rec, cons)})
        rows.append({"set": it["set"], "generator": it["generator"], "state": it["state"], "system": it["system"],
                     "item": it["item"], "route": rec.route, "decision": d.decision, "trust_score": d.trust_score,
                     **{k: round(v, 4) for k, v in comp.items()},
                     "failed_constraints": ";".join(c.name for c in cons if not c.passed),
                     "reasons": " | ".join(d.reasons)})
    dec = pd.DataFrame(rows)
    dec.to_csv(OUT / "c5_decisions.csv", index=False, encoding="utf-8")
    grid, th, sweep = sensitivity(prepared)
    grid.to_csv(OUT / "c5_sensitivity_grid.csv", index=False, encoding="utf-8")
    th.to_csv(OUT / "c5_sensitivity_thresholds.csv", index=False, encoding="utf-8")
    sweep.to_csv(OUT / "c5_weight_sweep.csv", index=False, encoding="utf-8")
    plot_sweep(sweep, OUT / "c5_weight_sweep.png")
    unsafe = unsafe_on_c2()
    unsafe.to_csv(OUT / "c5_unsafe_constraint_c2.csv", index=False, encoding="utf-8")

    cfg = DecisionConfig()
    L = ["# C5 - Trust-aware decision engine on real pipeline outputs", "",
         f"Generated by `python -m experiments.decision_engine_eval` at base commit {env['git_commit']}.",
         f"Default weights {cfg.weights.model_dump()}, accept at T >= {cfg.accept_at} with floors {cfg.floors} and no",
         f"soft constraint failure or KB tampering, reject below T = {cfg.review_at} or on any hard failure.", "",
         "## Measured inputs (loaded from saved results)", "",
         f"- C (diagnosis): B4 detector AUC {sig.detector_auc:.3f}, 95% CI [{sig.detector_ci_low:.3f}, "
         f"{sig.detector_ci_high:.3f}] -> C = max(0, (ci_low - 0.5) / 0.5) = "
         f"{max(0.0, (sig.detector_ci_low - 0.5) / 0.5):.3f}.",
         f"- C (knowledge answers): lower 95% bound of GreenRAG B6 faithfulness = "
         f"{sig.answer_faithfulness_low.value:.3f} (n = {sig.answer_faithfulness_low.n}).",
         f"- C (forecasts, 90% CQR-Mondrian coverage CI low): " + ", ".join(
             f"{a} {low:.3f} -> C = {max(0.0, 1 - max(0.0, nom - low) / (1 - nom)):.3f}"
             for a, (nom, low, _) in sig.forecast_coverage_low.items()) + ".",
         f"- E fallback factor: B6 answer relevance fallback / generated = {sig.fallback_relevance_ratio.value:.3f}.",
         "- S: check strength = 1 - (1 - TPR) / (1 - FPR), C3 rates:", "",
         "| check | TPR | FPR | strength |", "|---|---|---|---|",
         *[f"| {name} | {c.tpr.value:.3f} (n = {c.tpr.n}) | {c.fpr.value:.3f} (n = {c.fpr.n}) | {c.strength:.3f} |"
           for name, c in [("injection detector (held-out)", sig.injection_check),
                           ("document trust, signed/in-process registry", sig.trust_check_full),
                           ("document trust, content checks only", sig.trust_check_content_only),
                           *[(f"telemetry {a}", c) for a, c in sig.telemetry_check.items()]]], "",
         "## Decisions with the default configuration", "",
         "| set | generator | system | state | items | accept | review | reject | mean T | mean E | mean R | mean C | mean S |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (s, g, sy, st), grp in dec.groupby(["set", "generator", "system", "state"], sort=False):
        L.append(f"| {s} | {g} | {sy} | {st} | {len(grp)} | " + " | ".join(
            _rate(grp.decision == d) for d in DECISIONS) + f" | {grp.trust_score.mean():.3f} | "
            + " | ".join(f"{grp[c].mean():.3f}" for c in "ERCS") + " |")
    top = dec[dec.decision != "accept"].reasons.str.split(" \\| ").explode().str.replace(r"\(.*", "", regex=True) \
        .str.replace(r"[0-9.]+", "#", regex=True).str.strip().value_counts().head(10)
    L += ["", "Most frequent reasons for not accepting (numbers masked):", ""]
    L += [f"- {n} x {r}" for r, n in top.items()]
    llama_diag = dec[(dec.set == "c5_llama") & (dec.system == "diagnose")]
    L += ["", f"Diagnoses reaching accept: {int((dec[dec.system == 'diagnose'].decision == 'accept').sum())} of "
          f"{int((dec.system == 'diagnose').sum())} (B5 qwen + C5 llama). Diagnoses rejected by a hard constraint: "
          f"{int(llama_diag.failed_constraints.str.contains('high_severity|no_unsafe|claims_cite').sum())} (llama).", ""]
    # reproducibility of the capture run against C4
    c4 = {(r["state"], r["system"], r["item_id"]): r["text"] for r in _load(SEC / "c4_raw_llama3.2-3b.jsonl")}
    same = []
    for r in _load(RAW):
        o = r["output"]
        text = o["answer"] if r["system"] == "greenrag" else (
            f"{o['diagnosis']} Likely cause: {o['likely_cause']['text']} Recommended action: "
            f"{o['recommended_action']['text']}")
        key = (r["state"], r["system"], r["item_id"])
        if key in c4:
            same.append(text == c4[key])
    L += [f"Reproducibility: the C5 capture run reproduces the C4 answer text for {_rate(same)} items "
          "(same pipeline, generator, temperature 0, seed 0).", ""]
    # sensitivity
    L += ["## Sensitivity to the weights", "",
          f"All {len(grid)} weight vectors on a 0.05 grid with every weight >= 0.05 (sum 1); decisions recomputed for "
          f"all {len(prepared)} items with the default thresholds and floors.", "",
          f"- Items whose decision differs from the default: mean {grid.changed_vs_default.mean():.3f}, "
          f"max {grid.changed_vs_default.max():.3f} (weights at the max: "
          + ", ".join(f"w{k}={grid.loc[grid.changed_vs_default.idxmax(), f'w{k}']:.2f}" for k in "ERCS") + ").", ""]
    L += ["| group | default accept / review / reject | accept share across the grid (min - max) | "
          "reject share across the grid (min - max) |", "|---|---|---|---|"]
    for g in sorted({f"{p['set']}|{p['system']}|{p['state']}" for p in prepared}):
        sub = [p for p in prepared if f"{p['set']}|{p['system']}|{p['state']}" == g]
        base = _decide_all(sub, cfg)
        L.append(f"| {g.replace('|', ' / ')} | " + " / ".join(f"{np.mean([b == d for b in base]):.2f}" for d in DECISIONS)
                 + f" | {grid[f'{g}|accept'].min():.2f} - {grid[f'{g}|accept'].max():.2f} | "
                   f"{grid[f'{g}|reject'].min():.2f} - {grid[f'{g}|reject'].max():.2f} |")
    L += ["", "Thresholds (default weights):", "", "| accept_at | review_at | accept | review | reject | changed vs default |",
          "|---|---|---|---|---|---|"]
    for _, r in th.iterrows():
        L.append(f"| {r.accept_at:.2f} | {r.review_at:.2f} | {r.accept:.3f} | {r.review:.3f} | {r.reject:.3f} | "
                 f"{r.changed_vs_default:.3f} |")
    L += ["", "One-at-a-time sweep (others rescaled): results/trust/c5_weight_sweep.png, c5_weight_sweep.csv.", ""]
    # C2 unprotected: text constraint
    if not unsafe.empty:
        L += ["## Defence in depth: the unsafe-directive constraint on C2's unprotected outputs", "",
              "| generator | state | outputs | successful attacks | successful attacks with an unsafe directive | "
              "clean-state outputs flagged (false positives) |", "|---|---|---|---|---|---|"]
        for (g, st), grp in unsafe.groupby(["generator", "state"], sort=False):
            succ = grp[grp.attack_success]
            L.append(f"| {g} | {st} | {len(grp)} | {len(succ)} | "
                     f"{_rate(succ.unsafe != '') if len(succ) else 'n/a'} | "
                     f"{_rate(grp.unsafe != '') if st == 'clean' else '-'} |")
        L.append("")
    (OUT / "c5_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", help="capture structured outputs with this Ollama model (llama3.2:3b)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run_path = OUT / "c5_run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    if args.llm:
        t0 = time.perf_counter()
        capture(args.llm)
        run.update({"generator": args.llm, "temperature": 0, "llm_seed": 0,
                    "capture_seconds_this_session": round(time.perf_counter() - t0, 1)})
    run["environment"] = environment_info()
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    report()


if __name__ == "__main__":
    main()
