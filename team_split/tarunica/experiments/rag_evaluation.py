"""B6: evaluate LLM-only vs basic RAG vs hybrid RAG vs GreenRAG (resumable).

    python -m experiments.rag_evaluation            # generation, judge calibration, judging, report
    python -m experiments.rag_evaluation --report   # rebuild the report from saved files only

Items: the 64 labelled KB queries (B2/B3) and the 22 Wind Farm A events in logged and unlogged mode (B5).
Generator LLM: qwen2.5:7b-instruct (temperature 0, seed 0). Judge: the local model with the higher
balanced accuracy on a calibration set with known labels (qwen2.5:7b-instruct vs llama3.2:3b).
See src/backend/evaluation/rag_eval.py for metric definitions.

Outputs (results/rag/): answers.jsonl, judgements.jsonl, judge_calibration.json, b6_summary.md,
b6_per_answer.csv, b6_run.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backend.evaluation.provenance import environment_info
from src.backend.evaluation.rag_eval import (
    calibrate_judge,
    calibration_set,
    extract_count,
    judge_claim,
    judge_relevance,
    split_claims,
    unsupported_numbers,
)
from src.backend.evaluation.rag_systems import (
    Answer,
    count_answer,
    diagnosis_text,
    fault_question,
    greenrag_answer,
    llm_only,
    rag_answer,
)
from src.backend.faultdiag.diagnosis import (
    Diagnoser,
    event_windows,
    load_reliability,
    load_signals,
    ollama_json_llm,
    ollama_llm,
)
from src.backend.kb.build import build_documents
from src.backend.kb.chunking import fixed_size_chunks, semantic_chunks
from src.backend.kb.queries import build_queries
from src.backend.kb.retrieval_eval import CosineIndex, bootstrap_mean_ci, minilm_embed
from src.backend.kb.taxonomy import categorize, load_events
from src.backend.retrieval.hybrid_retriever import BM25Index, HybridRetriever, RerankRetriever, cross_encoder_scorer

OUT = Path(__file__).resolve().parents[1] / "results" / "rag"
SYSTEMS = ("llm_only", "basic_rag", "hybrid_rag", "greenrag")
GENERATOR = "qwen2.5:7b-instruct"
JUDGE_CANDIDATES = ("qwen2.5:7b-instruct", "llama3.2:3b")


def ollama_text(model: str):
    """Plain-text generation with the same per-call timeout and retry as the JSON client."""
    return ollama_llm(model, json_mode=False)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


# ------------------------------------------------------------------ items

def build_items(docs):
    items = []
    by_id = {d.doc_id: d for d in docs}
    for q in build_queries(docs):
        ref = "\n\n".join(by_id[d].text for d in q.relevant)
        count_truth = None
        if q.query_type == "count":
            text = count_answer(q.text)
            count_truth = int(text.split()[0]) if text else None
        items.append({"item_id": f"kb::{q.query_id}", "dataset": "kb", "question": q.text, "reference": ref,
                      "relevant_docs": sorted(q.relevant), "true_component": None, "count_truth": count_truth,
                      "event_id": None, "mode": None})
    events = load_events("A").set_index("event_id")
    comp_docs = {d.component: [] for d in docs}
    for d in docs:
        comp_docs[d.component].append(d.doc_id)
    for s in load_signals():
        label = events.loc[s.event_id, "event_label"]
        comp = categorize(events.loc[s.event_id, "event_description"])[0] if label == "anomaly" else None
        ref_docs = [f"fault-a-{s.event_id}"] if label == "anomaly" else []
        if comp:
            ref_docs += [f"trouble-wind-{comp}", f"manual-wind-{comp}"]
        ref = "\n\n".join(by_id[d].text for d in ref_docs if d in by_id)
        for mode in ("logged", "unlogged"):
            items.append({"item_id": f"fault::{s.event_id}::{mode}", "dataset": f"fault_{mode}",
                          "question": fault_question(s), "reference": ref, "relevant_docs": comp_docs.get(comp, []),
                          "true_component": comp, "count_truth": None, "event_id": s.event_id, "mode": mode,
                          "label": label})
    return items


# ------------------------------------------------------------- generation

def generate(items, docs, answers_path: Path) -> None:
    done = {(a["item_id"], a["system"]) for a in load_jsonl(answers_path)}
    embed = minilm_embed()
    sem = semantic_chunks(docs, embed)
    basic = CosineIndex(fixed_size_chunks(docs), embed)
    hybrid = HybridRetriever([BM25Index(sem), CosineIndex(sem, embed)], k_rrf=60, candidates=50)
    green = RerankRetriever(hybrid, cross_encoder_scorer(), candidates=20)
    gen_text, gen_json = ollama_text(GENERATOR), ollama_json_llm(GENERATOR)
    components = {d.doc_id: d.component for d in docs}
    events = load_events("A")
    diagnoser = Diagnoser(green, gen_json, load_reliability(), event_windows(events))
    signals = {s.event_id: s for s in load_signals()}
    with answers_path.open("a", encoding="utf-8") as fh:
        for it in items:
            exclude = None
            if it["mode"] == "unlogged":
                eid = it["event_id"]
                exclude = (lambda h, eid=eid: h["metadata"].get("farm") == "A" and h["metadata"].get("event_id") == eid)
            for system in SYSTEMS:
                if (it["item_id"], system) in done:
                    continue
                t = time.perf_counter()
                try:
                    if system == "llm_only":
                        ans = llm_only(gen_text, it["question"])
                    elif system == "basic_rag":
                        ans = rag_answer(system, gen_text, basic, it["question"], exclude)
                    elif system == "hybrid_rag":
                        ans = rag_answer(system, gen_text, hybrid, it["question"], exclude)
                    elif it["dataset"] == "kb":
                        ans = greenrag_answer(gen_json, green, it["question"], exclude)
                    else:
                        rep = diagnoser.diagnose(signals[it["event_id"]], components,
                                                 exclude_own_event=(it["mode"] == "unlogged"))
                        ans = Answer("greenrag", diagnosis_text(rep),
                                     [{"chunk_id": e.chunk_id, "doc_id": e.doc_id, "text": e.excerpt}
                                      for e in rep.evidence], route="diagnoser", notes=rep.notes)
                except Exception as exc:  # noqa: BLE001 - recorded as an error answer, excluded from metrics, counted
                    ans = Answer(system, "", [], route="error", notes=[f"{type(exc).__name__}: {str(exc)[:200]}"])
                fh.write(json.dumps({"item_id": it["item_id"], "system": system, "text": ans.text,
                                     "context": ans.context, "route": ans.route, "notes": ans.notes,
                                     "seconds": round(time.perf_counter() - t, 2)}, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"gen {it['item_id']:<32} {system:<11} {time.perf_counter() - t:5.1f}s {ans.route}", flush=True)


# ---------------------------------------------------------------- judging

def choose_judge(docs) -> tuple[str, dict]:
    path = OUT / "judge_calibration.json"
    if path.exists():
        cal = json.loads(path.read_text(encoding="utf-8"))
    else:
        items = calibration_set(docs, n_each=20, seed=0)
        cal = {"items": len(items), "kinds": pd.Series([i.kind for i in items]).value_counts().to_dict(),
               "results": {m: calibrate_judge(ollama_json_llm(m), items) for m in JUDGE_CANDIDATES}}
        path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    best = max(JUDGE_CANDIDATES, key=lambda m: (cal["results"][m]["balanced_accuracy"], m == JUDGE_CANDIDATES[0]))
    return best, cal


def judge_all(items, answers, judge_model: str, judged_path: Path) -> None:
    done = {(j["item_id"], j["system"]) for j in load_jsonl(judged_path)}
    judge = ollama_json_llm(judge_model)
    by_item = {it["item_id"]: it for it in items}
    with judged_path.open("a", encoding="utf-8") as fh:
        for a in answers:
            if (a["item_id"], a["system"]) in done:
                continue
            it = by_item[a["item_id"]]
            ctx = "\n\n".join(c["text"] for c in a["context"])
            ref = it["reference"] + ("\n\n" + ctx if ctx else "")
            failures = []

            def safe(fn, *args):
                try:
                    return fn(judge, *args)
                except Exception as exc:  # noqa: BLE001 - a failed judge call is an unparsed verdict, counted
                    failures.append(type(exc).__name__)
                    return None

            claims = []
            if a["route"] != "error":
                for claim in split_claims(a["text"]):
                    in_ctx = safe(judge_claim, claim, ctx) if ctx else None
                    in_ref = True if in_ctx else safe(judge_claim, claim, ref)
                    claims.append({"claim": claim, "supported_by_context": in_ctx, "supported_by_reference": in_ref})
            rec = {"item_id": a["item_id"], "system": a["system"], "claims": claims,
                   "answer_relevance": safe(judge_relevance, it["question"], a["text"]) if a["route"] != "error" else None,
                   "unsupported_numbers": unsupported_numbers(a["text"], ref), "judge_failures": failures}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"judge {a['item_id']:<32} {a['system']:<11} claims {len(claims)}", flush=True)


# ----------------------------------------------------------------- report

def per_answer_table(items, answers, judged, docs) -> pd.DataFrame:
    comp_of = {d.doc_id: d.component for d in docs}
    by_item = {it["item_id"]: it for it in items}
    ans = {(a["item_id"], a["system"]): a for a in answers}
    rows = []
    for j in judged:
        it, a = by_item[j["item_id"]], ans[(j["item_id"], j["system"])]
        if a["route"] == "error":
            continue                                   # generation failed: excluded, counted in the report
        claims = j["claims"]
        ctx_verdicts = [c["supported_by_context"] for c in claims if c["supported_by_context"] is not None]
        ref_verdicts = [c["supported_by_reference"] for c in claims if c["supported_by_reference"] is not None]
        ctx_docs = [c["doc_id"] for c in a["context"]]
        if it["dataset"] == "kb":
            ctx_rel = np.mean([d in it["relevant_docs"] for d in ctx_docs]) if ctx_docs else np.nan
        else:
            ctx_rel = (np.mean([comp_of.get(d) == it["true_component"] for d in ctx_docs])
                       if ctx_docs and it["true_component"] else np.nan)
        count_ok = np.nan
        if it["count_truth"] is not None:
            count_ok = float(extract_count(a["text"]) == it["count_truth"])
        rows.append({"item_id": j["item_id"], "dataset": it["dataset"], "system": j["system"], "route": a["route"],
                     "label": it.get("label"), "n_claims": len(claims),
                     "faithfulness": np.mean(ctx_verdicts) if ctx_verdicts and a["context"] else np.nan,
                     "claim_unsupported_rate": 1 - np.mean(ref_verdicts) if ref_verdicts else np.nan,
                     "hallucinated": float(any(v is False for v in ref_verdicts) or bool(j["unsupported_numbers"])),
                     "unsupported_number": float(bool(j["unsupported_numbers"])),
                     "answer_relevance": j["answer_relevance"], "context_relevance": ctx_rel,
                     "count_exact_match": count_ok, "seconds": a["seconds"]})
    return pd.DataFrame(rows)


def report(items, docs, judge_model, cal) -> None:
    answers, judged = load_jsonl(OUT / "answers.jsonl"), load_jsonl(OUT / "judgements.jsonl")
    t = per_answer_table(items, answers, judged, docs)
    t.to_csv(OUT / "b6_per_answer.csv", index=False, encoding="utf-8")
    env = environment_info()
    L = ["# B6 RAG evaluation", "",
         f"Generated by `python -m experiments.rag_evaluation` at base commit {env['git_commit']}.",
         f"Generator: {GENERATOR} (temperature 0, seed 0). Judge: {judge_model} (chosen on the calibration set below).",
         "Items: 64 labelled KB queries; 22 Wind Farm A events in logged and unlogged mode (12 anomaly, 10 normal).",
         "Mean over answers with 95% bootstrap CI (answers resampled, seed 0). n/a = not defined for that system.", "",
         "Metrics: faithfulness = share of the answer's claims supported by its OWN retrieved context (judge);",
         "hallucination rate = share of answers with >= 1 claim unsupported by reference + retrieved context (judge)",
         "OR >= 1 number not present in them (exact match); claim-unsupported = share of claims unsupported;",
         "answer relevance = judge 1-5 mapped to 0-1; context relevance = share of retrieved documents relevant by",
         "construction (labels, no judge); count exact match = computed count vs answer (6 count queries).", "",
         "**The judge is an LLM grading LLM output: it has its own error and bias (including self-preference if it is",
         "the generator model) and is not ground truth. Its measured accuracy is given at the end.**", ""]
    metrics = ("faithfulness", "hallucinated", "claim_unsupported_rate", "unsupported_number", "answer_relevance",
               "context_relevance")
    for ds in ("kb", "fault_logged", "fault_unlogged"):
        sub = t[t.dataset == ds]
        L += [f"## {ds} ({sub.item_id.nunique()} items)", "",
              "| system | faithfulness | hallucination rate | claim unsupported | answers w/ unsupported number | "
              "answer relevance | context relevance |", "|---|---|---|---|---|---|---|"]
        for system in SYSTEMS:
            s = sub[sub.system == system]
            cells = []
            for m in metrics:
                v = s[m].dropna()
                if v.empty:
                    cells.append("n/a")
                else:
                    mean, lo, hi = bootstrap_mean_ci(v)
                    cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
            L.append(f"| {system} | " + " | ".join(cells) + " |")
        if ds == "kb":
            L += ["", "Count questions (exact match against event_info.csv counts): "
                  + ", ".join(f"{s} {int(t[(t.system == s) & t.count_exact_match.notna()].count_exact_match.sum())}/"
                              f"{int(t[(t.system == s)].count_exact_match.notna().sum())}" for s in SYSTEMS)]
        L.append("")
    errors = pd.DataFrame(answers)
    n_err = errors[errors.route == "error"].groupby("system").size().to_dict()
    judge_fail = sum(len(j.get("judge_failures", [])) for j in judged)
    unparsed = sum(c["supported_by_reference"] is None for j in judged for c in j["claims"])
    lat = errors[errors.route != "error"].groupby("system")["seconds"].median()
    L += ["## Run health", "",
          f"Generation errors (timeout after retry or other failure; excluded from metrics): "
          f"{n_err or 'none'}. Judge call failures: {judge_fail}; claims left unjudged: {unparsed}.",
          "Median generation time per answer (s): " + ", ".join(f"{s} {lat.get(s, float('nan')):.1f}" for s in SYSTEMS)
          + ". One GreenRAG answer in the first run (fault::13::unlogged) took 39,097 s because an Ollama call hung;"
          " calls now time out after 120 s (one retry). Medians are reported because of that outlier.", ""]
    r = cal["results"]
    L += ["## Judge calibration (known labels)", "",
          f"{cal['items']} claim/context pairs built by construction: {cal['kinds']}.", "",
          "| model | accuracy | balanced accuracy | supported recognised (TPR) | unsupported recognised (TNR) | "
          "per kind | unparsed |", "|---|---|---|---|---|---|---|"]
    for m in JUDGE_CANDIDATES:
        x = r[m]
        kinds = ", ".join(f"{k} {v:.2f}" for k, v in sorted(x["by_kind"].items()))
        L.append(f"| {m} | {x['accuracy']:.3f} | {x['balanced_accuracy']:.3f} | {x['true_positive_rate']:.3f} | "
                 f"{x['true_negative_rate']:.3f} | {kinds} | {x['unparsed']} |")
    (OUT / "b6_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "b6_run.json").write_text(json.dumps({"environment": env, "generator": GENERATOR, "judge": judge_model},
                                                indent=2), encoding="utf-8")


RUN2 = OUT / "run2"


def generate_run2(items, docs, answers_path: Path) -> None:
    """Run 2: GreenRAG on the fault items only, with sibling-chunk expansion and the fixed fallback."""
    from src.backend.retrieval.hybrid_retriever import SiblingExpander

    done = {a["item_id"] for a in load_jsonl(answers_path)}
    embed = minilm_embed()
    sem = semantic_chunks(docs, embed)
    hybrid = HybridRetriever([BM25Index(sem), CosineIndex(sem, embed)], k_rrf=60, candidates=50)
    green = RerankRetriever(hybrid, cross_encoder_scorer(), candidates=20)
    diagnoser = Diagnoser(green, ollama_json_llm(GENERATOR), load_reliability(), event_windows(load_events("A")),
                          expander=SiblingExpander(sem))
    components = {d.doc_id: d.component for d in docs}
    signals = {s.event_id: s for s in load_signals()}
    with answers_path.open("a", encoding="utf-8") as fh:
        for it in items:
            if it["dataset"] == "kb" or it["item_id"] in done:
                continue
            t = time.perf_counter()
            try:
                rep = diagnoser.diagnose(signals[it["event_id"]], components,
                                         exclude_own_event=(it["mode"] == "unlogged"))
                ans = Answer("greenrag", diagnosis_text(rep),
                             [{"chunk_id": e.chunk_id, "doc_id": e.doc_id, "text": e.excerpt} for e in rep.evidence],
                             route="diagnoser", notes=rep.notes)
            except Exception as exc:  # noqa: BLE001 - recorded, excluded, counted
                ans = Answer("greenrag", "", [], route="error", notes=[f"{type(exc).__name__}: {str(exc)[:200]}"])
            fh.write(json.dumps({"item_id": it["item_id"], "system": "greenrag", "text": ans.text,
                                 "context": ans.context, "route": ans.route, "notes": ans.notes,
                                 "seconds": round(time.perf_counter() - t, 2)}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"run2 gen {it['item_id']:<28} {time.perf_counter() - t:5.1f}s", flush=True)


def report_run2(items, docs, judge_model) -> None:
    run1 = per_answer_table(items, load_jsonl(OUT / "answers.jsonl"), load_jsonl(OUT / "judgements.jsonl"), docs)
    run2 = per_answer_table(items, load_jsonl(RUN2 / "answers.jsonl"), load_jsonl(RUN2 / "judgements.jsonl"), docs)
    run2.to_csv(RUN2 / "b6_run2_per_answer.csv", index=False, encoding="utf-8")
    env = environment_info()
    metrics = ("faithfulness", "hallucinated", "claim_unsupported_rate", "answer_relevance", "context_relevance")
    L = ["# B6 run 2 - fault events after the two fixes", "",
         f"Generated by `python -m experiments.rag_evaluation --run2` at base commit {env['git_commit']}.",
         "Changes vs run 1 (GreenRAG only): sibling-chunk expansion of the retrieved evidence, and the fallback now",
         "searches all of the incident record's chunks for the logged cause. Other systems: run-1 answers unchanged",
         "(no expansion). KB questions were not re-run. Same generator, judge "
         f"({judge_model}), items and metrics as run 1. Mean [95% bootstrap CI].", ""]
    for ds in ("fault_logged", "fault_unlogged"):
        L += [f"## {ds}", "", "| system | run | faithfulness | hallucination rate | claim unsupported | answer relevance | "
              "context relevance |", "|---|---|---|---|---|---|---|"]
        rows = [(s, "run 1", run1[(run1.dataset == ds) & (run1.system == s)]) for s in SYSTEMS]
        rows.append(("greenrag", "**run 2**", run2[run2.dataset == ds]))
        for system, run, s in rows:
            cells = []
            for m in metrics:
                v = s[m].dropna()
                cells.append("n/a" if v.empty else "{:.3f} [{:.3f}, {:.3f}]".format(*bootstrap_mean_ci(v)))
            L.append(f"| {system} | {run} | " + " | ".join(cells) + " |")
        g1 = run1[(run1.dataset == ds) & (run1.system == "greenrag")].set_index("item_id")
        g2 = run2[run2.dataset == ds].set_index("item_id")
        for m in ("faithfulness", "hallucinated"):
            d = (g2[m] - g1[m]).dropna()
            L.append(f"\nGreenRAG run 2 - run 1, {m}: {d.mean():+.3f} (paired over {len(d)} events, "
                     "95% CI [{1:+.3f}, {2:+.3f}])".format(*bootstrap_mean_ci(d)))
        L.append("")
    errs = sum(a["route"] == "error" for a in load_jsonl(RUN2 / "answers.jsonl"))
    L.append(f"Run-2 generation errors: {errs}.")
    (RUN2 / "b6_run2_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="only rebuild the report from saved files")
    parser.add_argument("--run2", action="store_true", help="B6 run 2: GreenRAG fault items with the fixes")
    args = parser.parse_args()
    if args.run2:
        RUN2.mkdir(parents=True, exist_ok=True)
        docs = build_documents()
        items = build_items(docs)
        judge_model, _ = choose_judge(docs)
        generate_run2(items, docs, RUN2 / "answers.jsonl")
        fault_items = [it for it in items if it["dataset"] != "kb"]
        judge_all(fault_items, load_jsonl(RUN2 / "answers.jsonl"), judge_model, RUN2 / "judgements.jsonl")
        report_run2(items, docs, judge_model)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    docs = build_documents()
    items = build_items(docs)
    judge_model, cal = choose_judge(docs)
    print(f"judge: {judge_model} " + json.dumps({m: round(cal['results'][m]['balanced_accuracy'], 3)
                                                 for m in JUDGE_CANDIDATES}), flush=True)
    if not args.report:
        generate(items, docs, OUT / "answers.jsonl")
        judge_all(items, load_jsonl(OUT / "answers.jsonl"), judge_model, OUT / "judgements.jsonl")
    report(items, docs, judge_model, cal)


if __name__ == "__main__":
    main()
