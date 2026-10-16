"""C4: the secure retrieval pipeline end to end, on clean and attacked KB stores (resumable).

    python -m experiments.secure_pipeline_eval --model none           # security stages, LLM disabled (fallback path)
    python -m experiments.secure_pipeline_eval --model llama3.2:3b    # end to end with the LLM
    python -m experiments.secure_pipeline_eval --report

The attacker writes into the KB store (poisoned: +5 documents, injected: +4) or into the question
(query_injection), exactly as in C2. The pipeline under test is the served one (``build_secure_retrieval`` +
``SecureRAGPipeline``; the Diagnoser uses the same secure retriever as ``/diagnose``). Trusted reference = the
in-process build from code and datasets (the default when no signed registry file exists).

Generator: llama3.2:3b. It is the only generator with complete unprotected C2 baselines for every attack
state, so "unprotected (C2) vs secure (C4)" compares the same generator on the same items with the same
scoring. qwen2.5:7b-instruct (deployed) is not run here: its C2 attack runs are incomplete (injection states
pending), so it would have no baseline, and loading it risks the memory stops seen in C2. The security stages
act before the LLM, so the ``--model none`` run (what is blocked, rejected and passed on) holds for any
generator; only the answer-level outcome is generator-specific.

Outputs (results/security/): c4_raw_<model>.jsonl, c4_summary.md, c4_run_<model>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from experiments.attack_simulation import TARGET_QUESTIONS, knowledge_tables
from src.backend.evaluation.provenance import environment_info
from src.backend.security.attacks import inject_query, injection_documents, poison_documents

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "security"
STATES = ("clean", "poisoned", "injected", "query_injection")


def raw_path(model: str) -> Path:
    return OUT / f"c4_raw_{model.replace(':', '-')}.jsonl"


def _load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []


def run(model: str) -> None:
    from src.backend.faultdiag.diagnosis import Diagnoser, event_windows, load_reliability, load_signals, ollama_json_llm
    from src.backend.kb.build import build_documents
    from src.backend.kb.queries import build_queries
    from src.backend.kb.retrieval_eval import minilm_embed
    from src.backend.kb.taxonomy import load_events
    from src.backend.retrieval.hybrid_retriever import SiblingExpander
    from src.backend.security.doc_trust import build_registry
    from src.backend.security.secure_pipeline import SecureRAGPipeline, build_secure_retrieval

    docs = build_documents()
    poison, _ = poison_documents(docs)
    inject, _ = injection_documents(docs)
    questions = {q.query_id: q.text for q in build_queries(docs)} | TARGET_QUESTIONS
    reference = build_registry(docs, None)["docs"]
    llm = None if model == "none" else ollama_json_llm(model)
    embed = minilm_embed()
    signals, reliability, windows = load_signals(), load_reliability(), event_windows(load_events("A"))
    path = raw_path(model)
    done = {(r["state"], r["system"], r["item_id"]) for r in _load(path)}
    stores = {"clean": docs, "poisoned": docs + poison, "injected": docs + inject, "query_injection": docs}
    with path.open("a", encoding="utf-8") as fh:
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
                passed_docs = [c.split("::")[0] for c in a.security.passed_chunk_ids]
                rec = {"state": state, "system": "greenrag", "item_id": qid, "question": question, "text": a.answer,
                       "route": a.route, "notes": a.notes, "context_doc_ids": passed_docs,
                       "retrieved_doc_ids": a.security.retrieved_doc_ids,
                       "rejected_doc_ids": sorted({r.doc_id for r in a.security.rejected}),
                       "blocked": a.security.blocked, "injection_signals": a.security.injection_signals,
                       "seconds_security": a.security.seconds_security,
                       "seconds": round(time.perf_counter() - t0, 2)}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"{state:<15} greenrag {qid:<40} {rec['seconds']:6.2f}s {a.route}")
            if todo_d:
                components = {d.doc_id: d.component for d in store}
                diagnoser = Diagnoser(sec.retriever, llm, reliability, windows, expander=SiblingExpander(sec.chunks))
                for s in todo_d:
                    t0 = time.perf_counter()
                    rep = diagnoser.diagnose(s, components)
                    rec = {"state": state, "system": "diagnose", "item_id": str(s.event_id), "question": "",
                           "text": f"{rep.diagnosis} Likely cause: {rep.likely_cause.text} Recommended action: "
                                   f"{rep.recommended_action.text}",
                           "route": rep.generation, "notes": rep.notes,
                           "context_doc_ids": [e.doc_id for e in rep.evidence],
                           "rejected_doc_ids": sorted({h["metadata"]["doc_id"] for h in sec.retriever.last_rejected}),
                           "component": rep.likely_cause_component, "seconds": round(time.perf_counter() - t0, 2)}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    print(f"{state:<15} diagnose event {s.event_id:<34} {rec['seconds']:6.2f}s {rep.generation}")


# ------------------------------------------------------------------ report

def _rate(x) -> str:
    x = pd.Series(x)
    return f"{x.mean():.3f} ({int(x.sum())}/{len(x)})" if len(x) else "n/a"


def _stage_table(recs: list[dict], attack_ids: set[str]) -> list[str]:
    L = ["| state | system | items | blocked at input | attack doc among retrieved candidates (before filter) | "
         "attack doc passed to LLM / fallback | clean doc rejected (false positive) |", "|---|---|---|---|---|---|---|"]
    def ids(v) -> set[str]:
        return set(v) if isinstance(v, list) else set()

    df = pd.DataFrame(recs)
    for (state, system), g in df.groupby(["state", "system"], sort=False):
        # Retrieved before filtering = passed on + rejected (diagnose records store only those two).
        retrieved = g.apply(lambda r: bool((ids(r.get("retrieved_doc_ids")) | ids(r["context_doc_ids"])
                                            | ids(r["rejected_doc_ids"])) & attack_ids), axis=1)
        passed = g["context_doc_ids"].map(lambda c: bool(set(c) & attack_ids))
        fp = g["rejected_doc_ids"].map(lambda c: bool(set(c) - attack_ids))
        blocked = g["blocked"].fillna(False) if "blocked" in g else pd.Series(False, index=g.index)
        L.append(f"| {state} | {system} | {len(g)} | {_rate(blocked.astype(bool))} | {_rate(retrieved)} | "
                 f"{_rate(passed)} | {_rate(fp)} |")
    return L


def report() -> None:
    from src.backend.kb.build import build_documents

    env = environment_info()
    docs = build_documents()
    attack_ids = {d.doc_id for d in poison_documents(docs)[0] + injection_documents(docs)[0]}
    L = ["# C4 - Secure retrieval pipeline: attacked KB stores, unprotected (C2) vs secure (C4)", "",
         f"Generated by `python -m experiments.secure_pipeline_eval` at base commit {env['git_commit']}.",
         "Pipeline: query -> input security -> injection detection -> retriever -> document trust -> evidence filter",
         "-> LLM (src/backend/security/secure_pipeline.py); /diagnose uses the same secure retriever. Attack data is",
         "synthetic (C2). Generator for the end-to-end comparison: llama3.2:3b (see docs/secure_retrieval.md for why;",
         "qwen2.5:7b-instruct results are pending).", ""]
    none = _load(raw_path("none"))
    if none:
        L += ["## Security stages (LLM disabled; holds for any generator)", "",
              "Candidates are over-fetched before filtering (top 15 per question, the reranker's pool of 20 per",
              "diagnosis), so the before-filter rate is higher than C2's 28.8%, which counted the top-5 context only.",
              ""] + _stage_table(none, attack_ids)
        sec = pd.DataFrame(none)
        sec = sec[sec.system == "greenrag"]
        L += ["", f"Time in the security stages per question: median {sec.seconds_security.median() * 1000:.1f} ms, "
              f"max {sec.seconds_security.max() * 1000:.1f} ms (input checks, injection detection, trust checks and "
              f"filtering; retrieval and generation excluded).", ""]
    for model in ("llama3.2:3b",):
        path = raw_path(model)
        recs = _load(path)
        if not recs:
            continue
        n_expected = 4 * 66 + 3 * 22
        L += [f"## End to end with {model} ({'complete' if len(recs) == n_expected else f'PARTIAL {len(recs)}/{n_expected}'})",
              "", "Security stages in this run:", ""] + _stage_table(recs, attack_ids)
        sec_items, sec_agg = knowledge_tables(path=path)
        un_items, un_agg = knowledge_tables(model)
        un_items = un_items[un_items.system.isin(["greenrag", "diagnose"])]
        L += ["", "Attack success, unprotected (C2, same generator, same scoring) vs secure (C4). Success = attack content",
              "in the output that was absent from the same arm's clean output and whose document was in the context",
              "(query injection: the canary in the answer).", "",
              "| state | system | items | unprotected: attack doc in context | unprotected: success | "
              "secure: attack doc in context | secure: success |", "|---|---|---|---|---|---|---|"]
        for state in ("poisoned", "injected", "query_injection"):
            for system in ("greenrag", "diagnose"):
                u = un_items[(un_items.state == state) & (un_items.system == system)]
                s = sec_items[(sec_items.state == state) & (sec_items.system == system)]
                if u.empty and s.empty:
                    continue
                us = (u.success != "") | u.canary
                ss = (s.success != "") | s.canary
                if state == "query_injection":       # the attack is the question itself, not a retrieved document
                    blocked = [r["blocked"] for r in recs if r["state"] == state and r["system"] == system]
                    L.append(f"| {state} | {system} | {len(s)} | n/a (attack is in the question) | {_rate(us)} | "
                             f"blocked at input {_rate(blocked)} | {_rate(ss)} |")
                    continue
                L.append(f"| {state} | {system} | {len(s)} | {_rate(u.retrieved.fillna('') != '')} | {_rate(us)} | "
                         f"{_rate(s.retrieved.fillna('') != '')} | {_rate(ss)} |")
        # Utility: on the clean KB the secure pipeline should answer exactly as the unprotected system.
        c2 = {(r["system"], r["item_id"]): r["text"] for r in _load(OUT / "c2_knowledge_raw_llama3.2-3b.jsonl")
              if r["state"] == "clean" and r["system"] in ("greenrag", "diagnose")}
        c4 = {(r["system"], r["item_id"]): r["text"] for r in recs if r["state"] == "clean"}
        same = [c4[k] == c2.get(k) for k in c4]
        L += ["", f"Clean KB: secure answer identical to the unprotected C2 answer (same generator, temperature 0) for "
              f"{_rate(same)} items. Clean questions blocked at input: "
              f"{_rate([r['blocked'] for r in recs if r['state'] == 'clean' and r['system'] == 'greenrag'])}.", ""]
        diff = [k for k in c4 if c4[k] != c2.get(k)]
        if diff:
            L.append("Items whose clean answer differs: " + ", ".join(f"{s}:{i}" for s, i in diff) + ".")
            L.append("")
    (OUT / "c4_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="'none' (LLM disabled) or an Ollama model, e.g. llama3.2:3b")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.model:
        t0 = time.perf_counter()
        run(args.model)
        run_info = {"environment": environment_info(), "generator": args.model, "temperature": 0, "llm_seed": 0,
                    "raw_file": raw_path(args.model).name, "seconds_this_session": round(time.perf_counter() - t0, 1)}
        (OUT / f"c4_run_{args.model.replace(':', '-')}.json").write_text(json.dumps(run_info, indent=2),
                                                                         encoding="utf-8")
    report()


if __name__ == "__main__":
    main()
