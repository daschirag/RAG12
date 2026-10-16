"""B6 follow-up: (1) hybrid RAG with sibling-chunk expansion on the fault events; (2) corrected judge (v2).

    python -m experiments.b6_followup

1. Hybrid RAG gets the same SiblingExpander as GreenRAG run 2 (applied after the unlogged exclusion) and is
   regenerated on the 44 fault items. It is judged with the original judge (v1) and the corrected judge (v2).
2. Judge v2 (rag_eval.SUPPORT_PROMPT_V2) also sees the question and the list of retrieved documents, so true
   statements about the retrieval or the input can be verified. It is calibrated first on the original 60
   known-label items plus process/input statements with known truth (true and false versions).
   Existing answers are re-judged with v2 without regeneration: GreenRAG run 2, hybrid RAG run 1, and the new
   hybrid RAG + expansion answers.

Outputs (results/rag/run2/): hybrid_expanded_answers.jsonl, judgements_v1_hybrid_expanded.jsonl,
judgements_v2_*.jsonl, judge_v2_calibration.json, b6_followup_summary.md.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from experiments.rag_evaluation import (
    GENERATOR,
    OUT,
    RUN2,
    build_items,
    judge_all,
    load_jsonl,
    ollama_text,
    per_answer_table,
)
from src.backend.evaluation.provenance import environment_info
from src.backend.evaluation.rag_eval import (
    calibrate_judge_v2,
    calibration_set,
    judge_claim_v2,
    judge_relevance,
    process_calibration_items,
    split_claims,
    unsupported_numbers,
)
from src.backend.evaluation.rag_systems import rag_answer
from src.backend.faultdiag.diagnosis import ollama_json_llm
from src.backend.kb.build import build_documents
from src.backend.kb.chunking import semantic_chunks
from src.backend.kb.retrieval_eval import CosineIndex, bootstrap_mean_ci, minilm_embed
from src.backend.retrieval.hybrid_retriever import BM25Index, HybridRetriever, SiblingExpander

JUDGE = "qwen2.5:7b-instruct"


def generate_hybrid_expanded(items, docs, path: Path) -> None:
    done = {a["item_id"] for a in load_jsonl(path)}
    embed = minilm_embed()
    sem = semantic_chunks(docs, embed)
    hybrid = HybridRetriever([BM25Index(sem), CosineIndex(sem, embed)], k_rrf=60, candidates=50)
    expander, llm = SiblingExpander(sem), ollama_text(GENERATOR)
    with path.open("a", encoding="utf-8") as fh:
        for it in items:
            if it["item_id"] in done:
                continue
            exclude = None
            if it["mode"] == "unlogged":
                eid = it["event_id"]
                exclude = (lambda h, eid=eid: h["metadata"].get("farm") == "A" and h["metadata"].get("event_id") == eid)
            t = time.perf_counter()
            try:
                ans = rag_answer("hybrid_rag", llm, hybrid, it["question"], exclude, expander=expander)
                rec = {"text": ans.text, "context": ans.context, "route": ans.route, "notes": ans.notes}
            except Exception as exc:  # noqa: BLE001 - recorded, excluded, counted
                rec = {"text": "", "context": [], "route": "error", "notes": [f"{type(exc).__name__}"]}
            fh.write(json.dumps({"item_id": it["item_id"], "system": "hybrid_rag", **rec,
                                 "seconds": round(time.perf_counter() - t, 2)}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"hybrid+exp {it['item_id']:<28} {time.perf_counter() - t:5.1f}s", flush=True)


def judge_v2_all(items, answers, titles, judge, path: Path) -> None:
    done = {(j["item_id"], j["system"]) for j in load_jsonl(path)}
    by_item = {it["item_id"]: it for it in items}
    with path.open("a", encoding="utf-8") as fh:
        for a in answers:
            if (a["item_id"], a["system"]) in done or a["item_id"] not in by_item:
                continue
            it = by_item[a["item_id"]]
            ctx = "\n\n".join(c["text"] for c in a["context"])
            ref = it["reference"] + ("\n\n" + ctx if ctx else "")
            manifest = "\n".join(f"{d}: {titles.get(d, '')}" for d in dict.fromkeys(c["doc_id"] for c in a["context"]))
            claims = []
            if a["route"] != "error":
                for claim in split_claims(a["text"]):
                    in_ctx = judge_claim_v2(judge, claim, ctx, it["question"], manifest) if ctx else None
                    in_ref = True if in_ctx else judge_claim_v2(judge, claim, ref, it["question"], manifest)
                    claims.append({"claim": claim, "supported_by_context": in_ctx, "supported_by_reference": in_ref})
            rec = {"item_id": a["item_id"], "system": a["system"], "claims": claims,
                   "answer_relevance": judge_relevance(judge, it["question"], a["text"]) if a["route"] != "error" else None,
                   "unsupported_numbers": unsupported_numbers(a["text"], ref)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"judge v2 {a['item_id']:<28} {a['system']:<11} claims {len(claims)}", flush=True)


def is_process(claim: str) -> bool:
    return any(p in claim for p in ("was retrieved", "no troubleshooting", "Anomaly signal for", "Not established",
                                    "does not state"))


def main() -> None:
    RUN2.mkdir(parents=True, exist_ok=True)
    env = environment_info()
    docs = build_documents()
    titles = {d.doc_id: d.title for d in docs}
    items = [it for it in build_items(docs) if it["dataset"] != "kb"]
    item_ids = {it["item_id"] for it in items}

    # 1. Hybrid RAG with expansion.
    hx_path = RUN2 / "hybrid_expanded_answers.jsonl"
    generate_hybrid_expanded(items, docs, hx_path)
    judge_all(items, load_jsonl(hx_path), JUDGE, RUN2 / "judgements_v1_hybrid_expanded.jsonl")

    # 2. Judge v2 calibration, then re-judging.
    judge = ollama_json_llm(JUDGE)
    green2 = load_jsonl(RUN2 / "answers.jsonl")
    hyb1 = [a for a in load_jsonl(OUT / "answers.jsonl") if a["system"] == "hybrid_rag" and a["item_id"] in item_ids]
    cal_path = RUN2 / "judge_v2_calibration.json"
    if not cal_path.exists():
        q = {it["item_id"]: it["question"] for it in items}
        proc = process_calibration_items([{**a, "question": q[a["item_id"]]} for a in green2])
        random.Random(0).shuffle(proc)
        proc = proc[:80]
        cal = {"original_items": 60, "process_items": len(proc),
               "v2": calibrate_judge_v2(judge, calibration_set(docs, n_each=20, seed=0), proc)}
        cal_path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    sets = {"greenrag_run2": green2, "hybrid_run1": hyb1, "hybrid_expanded": load_jsonl(hx_path)}
    for name, answers in sets.items():
        judge_v2_all(items, answers, titles, judge, RUN2 / f"judgements_v2_{name}.jsonl")

    # Report.
    v1 = {"greenrag_run2": RUN2 / "judgements.jsonl", "hybrid_run1": OUT / "judgements.jsonl",
          "hybrid_expanded": RUN2 / "judgements_v1_hybrid_expanded.jsonl"}
    tables = {}
    for name, answers in sets.items():
        for judge_name, path in (("v1", v1[name]), ("v2", RUN2 / f"judgements_v2_{name}.jsonl")):
            judged = [j for j in load_jsonl(path) if j["item_id"] in item_ids and j["system"] ==
                      ("greenrag" if name.startswith("greenrag") else "hybrid_rag")]
            tables[(name, judge_name)] = per_answer_table(items, answers, judged, docs)
    L = ["# B6 follow-up: fair expansion for hybrid RAG, and a corrected judge (v2)", "",
         f"Generated by `python -m experiments.b6_followup` at base commit {env['git_commit']}.",
         f"Generator {GENERATOR}; judge {JUDGE}. Fault items only (22 events x logged/unlogged). Answers were not",
         "regenerated except hybrid RAG + expansion. Mean [95% bootstrap CI].", ""]
    for ds in ("fault_logged", "fault_unlogged"):
        L += [f"## {ds}", "", "| system | judge | faithfulness | hallucination rate | claim unsupported | answer relevance |",
              "|---|---|---|---|---|---|"]
        for name in sets:
            for jn in ("v1", "v2"):
                s = tables[(name, jn)]
                s = s[s.dataset == ds]
                cells = []
                for m in ("faithfulness", "hallucinated", "claim_unsupported_rate", "answer_relevance"):
                    v = s[m].dropna()
                    cells.append("n/a" if v.empty else "{:.3f} [{:.3f}, {:.3f}]".format(*bootstrap_mean_ci(v)))
                L.append(f"| {name} | {jn} | " + " | ".join(cells) + " |")
        L.append("")

    # Attribution for GreenRAG run 2: claims unsupported under v1, and what v2 says about them.
    j1 = {j["item_id"]: j for j in load_jsonl(RUN2 / "judgements.jsonl")}
    j2 = {j["item_id"]: j for j in load_jsonl(RUN2 / "judgements_v2_greenrag_run2.jsonl")}
    counts = {"process": [0, 0], "content": [0, 0]}          # [unsupported under v1, of those supported under v2]
    for iid, a in j1.items():
        v2_claims = {c["claim"]: c for c in j2[iid]["claims"]}
        for c in a["claims"]:
            if c["supported_by_reference"] is False:
                kind = "process" if is_process(c["claim"]) else "content"
                counts[kind][0] += 1
                counts[kind][1] += bool(v2_claims.get(c["claim"], {}).get("supported_by_reference"))
    L += ["## Where GreenRAG run 2's v1 hallucinations came from", "",
          "| claim kind (keyword rule) | unsupported under v1 | of those, supported under v2 | still unsupported |",
          "|---|---|---|---|"]
    for kind, (n, fixed) in counts.items():
        L.append(f"| {kind} | {n} | {fixed} | {n - fixed} |")
    for ds in ("fault_logged", "fault_unlogged"):
        a, b = tables[("greenrag_run2", "v1")], tables[("greenrag_run2", "v2")]
        L.append(f"\n{ds}: GreenRAG hallucination rate v1 {a[a.dataset == ds].hallucinated.mean():.3f} -> "
                 f"v2 {b[b.dataset == ds].hallucinated.mean():.3f}")
    c2 = cal["v2"]
    L += ["", "## Judge v2 calibration (known labels)", "",
          f"{c2['n']} items: the original 60 plus {cal['process_items']} process/input statements (true and false "
          "versions, built from the run-2 answers' questions and retrieved documents).", "",
          f"Balanced accuracy {c2['balanced_accuracy']:.3f}; recognises supported {c2['true_positive_rate']:.3f}, "
          f"unsupported {c2['true_negative_rate']:.3f}; unparsed {c2['unparsed']}.", "",
          "| kind | n | accuracy |", "|---|---|---|"]
    for kind, v in c2["by_kind"].items():
        L.append(f"| {kind} | {v['n']} | {v['accuracy']:.3f} |")
    (RUN2 / "b6_followup_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
