# Track B - Knowledge base, hybrid RAG and fault diagnosis (Tarunica S, 23BIT0185)

## Scope

The track covers seven phases:
- **B1 knowledge base:** real fault descriptions and SCADA knowledge, plus labelled synthetic manuals,
  troubleshooting, maintenance records and standard overviews.
- **B2 chunking comparison.**
- **B3 retrieval comparison:** BM25, vector, RRF hybrid, cross-encoder rerank, routed.
- **B4 fault detection on Wind Farm A.**
- **B5 diagnosis to RAG:** a pydantic-validated report with cited evidence.
- **B6 RAG evaluation:** faithfulness and hallucination.
- **B7 `/diagnose` API.**

## Knowledge base (`data/processed/knowledge_base/documents.jsonl`, `results/kb/kb_summary.md`)

| document type | real | synthetic |
|---|---|---|
| fault description (one per anomaly event, farms A-C) | 45 | 0 |
| SCADA knowledge (overviews, component histories, sensor catalogues) | 54 | 0 |
| equipment manual | 0 | 19 |
| troubleshooting procedure | 0 | 19 |
| maintenance record (one invented work order per real anomaly event) | 0 | 45 |
| standard summary (original overviews, no standard text copied) | 0 | 7 |

The total is 189 documents (99 real, 90 synthetic) in 210 build chunks.
- **Synthetic documents are marked.** They carry `synthetic: true` and the `[SYNTHETIC]` marker; the schema
  enforces both.
- **Wind, not solar faults.** The labelled faults are **wind-turbine** faults. The plan's solar-inverter fault
  list has no labelled counterpart, so inverter material exists only as synthetic manuals.

## Files and modules

| file | role |
|---|---|
| `src/backend/kb/{schema,taxonomy,content,build}.py` | B1: schema, 13-component taxonomy from `event_info.csv`, synthetic content, build |
| `src/backend/kb/{chunking,queries,retrieval_eval}.py` | B2: three chunkers, 64 labelled queries, IR metrics |
| `src/backend/retrieval/{hybrid_retriever,query_classifier}.py` | B3: BM25, RRF hybrid, cross-encoder rerank, sibling expansion |
| `src/backend/faultdiag/{data,models}.py`, `experiments/fault_diagnosis.py` | B4: leave-one-asset-out detectors |
| `src/backend/faultdiag/diagnosis.py`, `service.py` | B5/B7: `DiagnosisReport`, `Diagnoser`, `/diagnose` service |
| `src/backend/evaluation/{rag_eval,rag_systems}.py`, `experiments/rag_evaluation.py` | B6 |

## Endpoints

- `POST /diagnose {event_id | asset_id + window, mode: logged | unlogged, use_llm}` returns a `DiagnosisReport`:
  diagnosis, confidence, likely cause with cited chunk ids, evidence, recommended action, severity and generation.
  - 404 for an unknown event.
  - If the LLM times out or fails, the deterministic evidence-only report is returned (`generation = "fallback"`).
- `GET /diagnose/events`: Wind Farm A events with the B4 detector score.
- `GET /kb/chunks?ids=...`: chunk text for the console's source cards.
- `POST /assistant` (configs A-C reuse the B6 systems), `GET /assistant/suggestions`: the labelled B2 queries.

## Results

Chunking, 64 labelled queries, top 5 (`results/kb/b2_chunking_metrics.csv`):

| method | P@5 | R@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| fixed size | 0.197 | 0.602 | 0.606 | 0.574 |
| semantic | 0.231 | 0.695 | 0.707 | 0.667 |
| metadata-aware | 0.194 | 0.594 | 0.632 | 0.581 |

Retrieval (`results/kb/b3_retrieval_metrics.csv`):

| variant | P@5 | R@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| BM25 | 0.241 | 0.695 | 0.743 | 0.650 |
| vector | 0.231 | 0.695 | 0.707 | 0.667 |
| hybrid RRF | 0.266 | 0.781 | 0.826 | 0.752 |
| hybrid + rerank (production) | 0.269 | 0.789 | 0.958 | 0.835 |
| routed (quantitative -> BM25) | 0.259 | 0.766 | 0.880 | 0.781 |

RAG answers on the 64 knowledge-base questions, judged by qwen2.5:7b-instruct (`results/rag/b6_per_answer.csv`):

| system | faithfulness | hallucination rate | answer relevance |
|---|---|---|---|
| LLM only | n/a | 0.688 | 0.500 |
| basic RAG | 0.836 | 0.109 | 0.582 |
| hybrid RAG | 0.959 | 0.094 | 0.648 |
| GreenRAG | 0.977 | 0.016 | 0.688 |

**Fault events** (22, logged mode), after the run-2 fixes (sibling expansion, fallback fix) and with template
statements verified by code (`results/rag/run2/b6_followup_deterministic.md`):
- **GreenRAG:** faithfulness 0.848, hallucination 0.227.
- **Hybrid RAG:** faithfulness 0.902, also 0.227 hallucination, but answer relevance 0.000: it mostly abstains.

Fault detectors, Wind Farm A, event level, leave-one-asset-out, seeds 0-2 (`results/faultdiag/b4_event_metrics.csv`):

| model | accuracy | precision | recall | F1 | AUC |
|---|---|---|---|---|---|
| logistic regression (deployed) | 0.682 | 0.692 | 0.750 | 0.720 | 0.742 |
| random forest | 0.545 | 0.625 | 0.417 | 0.500 | 0.594 |
| XGBoost | 0.455 | 0.500 | 0.417 | 0.455 | 0.558 |
| Isolation Forest (fixed threshold) | 0.455 | 0.000 | 0.000 | 0.000 | 0.757 |

- **Deployed detector AUC.** 0.742 with 95% CI [0.500, 0.950]: **not distinguishable from chance**.
- **Isolation Forest threshold.** A nested leave-one-asset-out threshold (`results/faultdiag/b4_if_threshold.md`)
  gives F1 0.783 / 0.783 / 0.560 for seeds 0-2.
- **Diagnosis component correct** (12 anomaly events, `results/faultdiag/b5_results.csv`):
  - logged mode 12/12 (the event's own record is retrieved);
  - **unlogged mode 4/12**, below the majority-class baseline of 6/12 (`docs/fault_diagnosis.md`).
- **C6 re-run with qwen** (`results/security/c6_summary.md`): 12/12 component correct both without and with the
  secure retriever. The decision engine sends all 22 diagnoses to review.

## Limitations (honest)

- **The detector is near chance.** Event AUC CI [0.500, 0.950], and attribution top-1 was 0/12 (B4). So
  `confidence.level` is never "high", and the engine never accepts a diagnosis (C = 0).
- **Unlogged mode is the honest test**, and it is below the majority baseline. The good logged-mode score comes
  from retrieving the event's own record.
- **LLM text adds content.** The free text sometimes mixes in other events or adds unsupported causes. The demo
  report for event 26 cited fault-a-73 and fault-a-84 chunks.
- **The judge is an LLM.** It is qwen2.5:7b-instruct, the same family as the generator, not people.
  Its calibration set was small and easy.
- **Many documents are synthetic.** 90/189 documents are synthetic, and answers grounded in them are only as good
  as that text.
- **Chunk sizes differ between methods.** This confounds the B2 comparison; a size-matched run was not done
  because it would tune on the evaluation set.
- **Only Farm A is modelled.** Farms B and C are future work.

## Likely viva questions

1. **Why hybrid + rerank?** On the 64 labelled queries it has the best nDCG@5 (0.835) and MRR (0.958). The paired
   CIs exclude 0 against each single retriever. The cost is about 1 s per query for the cross-encoder on CPU.
2. **Why did routing hurt?** Count questions routed to BM25 scored poorly. GreenRAG instead answers count questions
   deterministically from `event_info.csv` (6/6 exact matches in B6).
3. **How is hallucination measured?**
   - The answer is split into claims, and each claim is judged against the retrieved context and the reference.
   - An answer counts as hallucinated if any claim, or any number, is unsupported.
   - Template sentences are checked by code.
4. **Why is the detector's confidence "low" or "moderate", never "high"?** The level is derived from the measured
   reliability (AUC CI) and from the evidence support. A detector whose CI touches 0.5 cannot justify "high".
5. **What is the difference between logged and unlogged mode?** Unlogged hides the event's own fault and
   maintenance documents, which simulates a new anomaly with no log entry yet.
6. **Why `status_type_id` was excluded:** in Farm A it encodes the label (0% normal in every anomaly window), so
   using it would leak the answer.

## Demo script (live)

1. Open the console's **Assistant** tab and click a suggested question (they are the labelled B2 queries). Show:
   - the answer with its citation;
   - the ACCEPT badge with E/R/C/S;
   - one source card expanded to show the chunk text.
2. Switch the configuration to **C (hybrid RAG)** and ask the same question: no trust panel, because the
   ablation removes security and trust.
3. Open **Fault diagnosis**, pick event 73 (logged), then **Diagnose**. Show:
   - severity and confidence;
   - the cited evidence and the ground-truth comparison;
   - the REVIEW decision with the reason "C = 0.000 below floor 0.5".
4. Switch to **Unlogged** and diagnose again: the event's own records are hidden, and the component may differ.
5. **Results** tab, Track B: the chunking, retrieval, RAG and detector tables with their caveats.

## What is not done

- No supervised component classifier: there are 1-6 events per class, too few to train one.
- Farms B and C are not modelled.
- No human-labelled answer quality; the only judge is an LLM.
- No size-matched chunking rerun.
- No incremental or event-driven ingestion: the KB is rebuilt by `python -m src.backend.kb.build`.
