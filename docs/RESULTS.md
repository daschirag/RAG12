# GreenRAG-Sec results

Every number below was copied from the named summary file. Each summary was produced by a real run in this repo
(fixed seeds, raw outputs beside it) and has its own CIs and protocol. Worse-than-baseline results are kept.
`python -m experiments.run_all` lists the status of every step (`results/run_all_status.md`).

## 1. Forecasting (1 h ahead)

Sources: `results/forecasting/a5_summary.md`, `results/forecasting/wind/wind_summary.md`. The full table with
MAPE and R2 is in `results/report_numbers.md`.

| series | persistence MAE | best learned MAE | interval (90%, cqr_mondrian) |
|---|---|---|---|
| Solar Plant 1, fixed split (5 test days) | 2458.0 kW | RF 1444.1 +/- 13.2 kW | daylight coverage 0.887 |
| Solar Plant 1, rolling origin (12 days) | | learned models beat persistence on 12/12 days | 0.893, CI [0.853, 0.927] |
| Solar Plant 2, fixed split | 1964.5 kW | LSTM 1216.7 +/- 20.2 kW | over-covers (0.957) |
| Wind T1, rolling origin (Sep-Dec 2018, 116 days) | **298.1 kW** | XGBoost 329.6 +/- 1.7 kW (worse) | 0.890, CI [0.869, 0.909] |

## 2. Retrieval (64 labelled queries, top 5)

Sources: `results/kb/b2_chunking_summary.md`, `results/kb/b3_retrieval_summary.md`.
- Chunking, nDCG@5: semantic 0.667, metadata-aware 0.581, fixed-size 0.574. The chunk sizes differ, which
  confounds the comparison.
- Retrievers, nDCG@5: BM25 0.650, vector 0.667, hybrid RRF 0.752, hybrid + rerank 0.835 (MRR 0.958, about
  1 s/query on CPU). Routed retrieval scored 0.781: routing hurt.

## 3. Fault diagnosis (Wind Farm A, leave-one-asset-out)

Sources: `results/faultdiag/b4_summary.md`, `b4_if_threshold.md`, `b5_summary.md`.
- **Event AUC.** Logistic regression 0.742 with CI [0.50, 0.95]; Isolation Forest 0.757 with CI [0.545, 0.953].
  Supervised row-level AUC is at chance, and attribution top-1 is 0/12.
- **Diagnosis reports.** Logged mode gets the component right on 12/12 (the event's own record is retrieved).
  Unlogged mode gets 4/12, below the majority baseline of 6/12.

## 4. Hallucination (B6, judge qwen2.5:7b-instruct)

Sources: `results/rag/b6_summary.md`, `results/rag/run2/b6_followup_deterministic.md`.
- **Knowledge-base questions (64).** GreenRAG reaches faithfulness 0.977 with hallucination 0.016.
  Hallucination for the other systems: LLM only 0.688, basic RAG 0.109, hybrid RAG 0.094. GreenRAG matches
  all count questions (6/6).
- **Fault events, after the run-2 fixes and code-verified template statements.** Logged mode: faithfulness
  0.848, hallucination 0.227. Hybrid RAG has the same hallucination rate but mostly abstains (answer relevance 0).

## 5-7. Attacks and defences (C2-C6)

Sources: `results/security/c2_summary.md`, `c3_summary.md`, `c4_summary.md`, `results/trust/c5_summary.md`,
`results/security/c6_summary.md`. All attack data is synthetic.

**Undefended system (C2)**
- **Telemetry replay** is the strongest attack. On Plant 1 it raises daylight MAE from 3013.8 to 4891.3 kW
  and cuts 90% coverage from 0.887 to 0.761.
- **The deployed fault detector is fully evaded** (recall 0.750 -> 0.000).
- **Query injection.** qwen obeyed 55/66 plain-RAG questions and 37/66 GreenRAG questions.
- **Poisoned documents** reach the retrieval context for 28.8% of questions.

**Detectors (C3)**
- Telemetry: 30/30 detections on Plant 1, 23/30 on Plant 2, 62/62 on wind.
- Injection: 66/66 on the C2 template, 11/15 on held-out phrasings (FPR 1/15).
- Signed trust registry: 17/17 tampered documents caught, FPR 0/189.

**Secure pipeline (C4, llama3.2:3b)**
- No attack document reached the LLM.
- Successes: poisoned answers 2/66 -> 0, poisoned diagnoses 6/22 -> 0, query injection 2/66 -> 0.
- Clean answers were identical to the undefended run (88/88).

**Decision engine (C5)**
- 0/110 diagnoses accepted (C = 0); 60/66 clean GreenRAG answers accepted.
- 66/66 query-injection items rejected.

**C6 (qwen2.5:7b-instruct: (a) undefended, (b) secure, (c) secure + engine)**, details in
`docs/security_evaluation.md`:
- **Query injection.** GreenRAG falls from 37/66 to 0/66 (all blocked). Unsafe procedure falls from 2/66 to 0;
  the forged record in /diagnose falls from 2/22 to 0.
- **Held-out injections.** Detection is 726/990 (4 of 15 phrasings never blocked). Of the delivered answers,
  the engine accepted 219, and 51 of those had executed the injection (e.g. "I approve skipping the pitch
  battery test").
- **Telemetry.** Replay is fully gated in (b) and (c). Plant 2 power x0.5 (5/5) and irradiation bias (2/5)
  pass the gate.
- **Clean utility unchanged.** Faithfulness is 0.970 vs 0.970 and hallucination 0.015 vs 0.015, with
  0/66 wrongly blocked and security stages at 7.7 ms.
- **Accepted-answer correctness.** On the capture items it is 142/142 for accepted answers. The over-caution
  cost: correct answers on attacked KBs sent to review (29/65 poisoned, 20/65 injected).
- **Flagging bad items:** recall 0.944 for review or reject, and reject precision 1.000.

## Cloud (I4) and ablation

- I4 local-vs-Azure is **PENDING**: no Azure configuration and a local-only decision
  (`results/cloud/i4_status.md`). No Azure number exists.
- Ablation A-F is implemented in `src/backend/orchestrator.py` but has **not been run** (`run_all` lists it
  as manual).
