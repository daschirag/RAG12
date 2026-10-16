# Track B - RAG evaluation (B6)

Code: `src/backend/evaluation/rag_eval.py` (claim splitting, judge, numeric checks, calibration),
`src/backend/evaluation/rag_systems.py` (the four systems). Run: `python -m experiments.rag_evaluation`
(resumable) -> `results/rag/b6_summary.md` (all tables with 95% CIs), `answers.jsonl`, `judgements.jsonl`,
`b6_per_answer.csv`, `judge_calibration.json`. Tests: `tests/test_rag_eval.py`.

## Design
- **Items**: the 64 labelled KB queries (B2/B3) and the 22 Wind Farm A events (12 anomaly, 10 normal) in logged and
  unlogged mode (B5): 108 items x 4 systems = 432 answers.
- **Systems** (same generator, qwen2.5:7b-instruct, temperature 0, seed 0): `llm_only` (no retrieval); `basic_rag`
  (MiniLM vector search over fixed-size chunks, the original approach); `hybrid_rag` (BM25 + MiniLM, RRF, semantic
  chunks); `greenrag` (hybrid + cross-encoder rerank, citations validated against the evidence with an extractive
  fallback, count questions answered deterministically from event_info.csv, and the full B5 Diagnoser for fault
  events). Unlogged mode hides the event's own documents from every retrieval system.
- **Faithfulness is claim-level on the full answer text**, not citation-level: citations are stripped, the answer
  is split into sentences, and each sentence is judged against the system's own retrieved context. A report with
  valid citations but an invented sentence (the B5 "environmental factors" case) is caught (tested).
- **Hallucination**: an answer is hallucinated if >= 1 claim is unsupported by the reference documents (relevant by
  construction) plus its retrieved context, OR it contains a number found in neither (exact match, no judge).
- **Numeric facts by exact match**: the 6 count questions are scored against counts computed from event_info.csv.
- **Context relevance from labels** (no judge); **answer relevance** from the judge (1-5 -> 0-1).
- **Judge**: chosen on a calibration set with known labels (60 claim/context pairs: 20 copied from the context,
  20 from a document about another component, 20 with one number changed). qwen2.5:7b-instruct scored 1.000
  balanced accuracy, llama3.2:3b 0.912, so qwen is the judge.

## Judge limitations (read before the numbers)
- **The judge is an LLM grading LLM output. It has its own error and bias and is not ground truth.**
- It is the same model as the generator, so self-preference bias is possible.
- The calibration set is easy (verbatim copies, unrelated sentences, changed numbers). Real answers paraphrase, so
  the judge's accuracy on real claims is **not measured** and is certainly below the calibration score.
- It judges process statements ("no troubleshooting procedure was retrieved") as unsupported even though they
  describe the system truthfully; this penalises GreenRAG's deterministic fallback text.
- Answer relevance is coarse: llm_only fault answers all received the same score (0.750), and faithful abstentions
  ("the context does not say") score 0.

## B6 run 1 results (mean [95% CI] over answers; full tables in results/rag/b6_summary.md)

KB queries (64):

| system | faithfulness | hallucination rate | answer relevance | context relevance | count exact match |
|---|---|---|---|---|---|
| llm_only | n/a | 0.688 [0.578, 0.797] | 0.500 | n/a | 0/6 |
| basic_rag | 0.836 [0.742, 0.914] | 0.109 [0.031, 0.188] | 0.582 | 0.197 | 5/6 |
| hybrid_rag | 0.959 [0.914, 0.992] | 0.094 [0.031, 0.172] | 0.648 | 0.287 | 4/6 |
| **greenrag** | **0.977** [0.938, 1.000] | **0.016** [0.000, 0.047] | **0.688** | **0.297** | **6/6** |

Fault events, logged mode (22):

| system | faithfulness | hallucination rate | answer relevance | context relevance |
|---|---|---|---|---|
| llm_only | n/a | 1.000 | 0.750 | n/a |
| basic_rag | 0.580 | 0.818 | 0.102 | 0.050 |
| hybrid_rag | **0.902** | **0.227** | 0.000 | 0.433 |
| greenrag | 0.288 | 0.955 | 0.602 | 0.433 |

Fault events, unlogged mode (22): llm_only hallucination 0.955; basic_rag faithfulness 0.617 / hallucination 0.818;
hybrid_rag 0.795 / 0.364; greenrag 0.167 / 0.955.

- **KB questions: GreenRAG is best on every metric**, with hallucination 1.6% vs 68.8% for LLM-only; retrieval alone
  (basic/hybrid) already cuts hallucination to about 10%. GreenRAG answers all 6 count questions exactly (computed,
  not generated). The GreenRAG vs hybrid difference in hallucination (0.016 vs 0.094) has overlapping CIs.
- **Fault events: GreenRAG is the worst RAG system on faithfulness and hallucination** (logged: 0.288 / 0.955).
  Hybrid RAG is the most faithful but mostly abstains ("the context does not say"), hence answer relevance 0.
  LLM-only invents a cause for every event (hallucination 1.000 logged).
- These fault results are reported as measured; nothing was re-run after seeing them.

## Why the fault results are poor (diagnosed after the run, not fixed)
1. **Semantic chunking separates the cause from the event.** 44 of 45 fault descriptions are split into two chunks:
   `::0` holds the event header (turbine, dates), which is what a question about "turbine 0 in June" matches, and
   `::1` holds the logged cause. Retrieval returns `::0` and misses `::1`, so no RAG system sees the cause.
   **B2's document-level relevance could not detect this** (retrieving `::0` counted as retrieving the document),
   so B2's choice of semantic chunking has a cost its metric did not measure.
2. **GreenRAG's fallback contradicts itself.** When the incident record is retrieved (support strong, confidence
   moderate) but the retrieved chunk lacks the logged description, the fallback still writes "Likely cause: not
   established for this incident". This is a bug: the fallback searches only the retrieved excerpt for the
   description line.
3. The fallback's process statements ("no troubleshooting procedure was retrieved") are counted as unsupported by
   the judge (see judge limitations).
Both were fixed afterwards at the user's request and re-run as B6 run 2 (below). Options considered: retrieve sibling chunks
of an incident document (parent-document retrieval) or keep fault descriptions in one chunk; fix the fallback
contradiction; re-run B6 as a clearly labelled second run, keeping this run for comparison.

## Run health
No generation errors, no judge failures, no unjudged claims. Median generation time per answer: llm_only 3.3 s,
basic_rag 3.1 s, hybrid_rag 2.9 s, greenrag 4.4 s. The first attempt was stopped by Claude Code for low system
memory; one GreenRAG answer in it took 39,097 s because an Ollama call hung. Ollama calls now time out after 120 s
with one retry (tested); failures are recorded, excluded from metrics and counted in the report.

## B6 run 2: fault events after fixing the two root causes

Run 1 (above, `results/rag/b6_summary.md`, `answers.jsonl`, `judgements.jsonl`) is kept unchanged. Run 2 is in
`results/rag/run2/` (`b6_run2_summary.md`, `answers.jsonl`, `judgements.jsonl`, `b6_run2_per_answer.csv`).
Run: `python -m experiments.rag_evaluation --run2`.

### The fixes (general, not specific to these events)
1. **Sibling-chunk expansion at retrieval time** (`retrieval.hybrid_retriever.SiblingExpander`): each retrieved
   chunk of a multi-chunk document is followed by its adjacent chunks (index +/- 1) from the same document. The
   reranked top 5 hits are unchanged; only text of already-retrieved documents is added. In the fault pipeline it
   is applied after the unlogged-mode exclusion, so hidden documents stay hidden. The knowledge base was not
   re-chunked, so B2/B3 stand.
2. **Fallback fix** (`faultdiag.diagnosis.Diagnoser._fallback`): the logged cause is searched in every evidence
   chunk of the incident record; if it is still not present, the text says the record was retrieved but does not
   state the cause, instead of "not established".

### Regression check on general retrieval (64 KB queries; `results/kb/b6_expansion_check.md`)

| variant | Precision@5 | Recall@5 | MRR | nDCG@5 | chunks in context | chars in context |
|---|---|---|---|---|---|---|
| A: B3 production (no expansion) | 0.269 | 0.789 | 0.958 | 0.835 | 5.00 | 1917 |
| B: production + expansion, documents in whole context | 0.269 | 0.789 | 0.958 | 0.835 | 8.77 | 3046 |
| B: same, first 5 chunks only | 0.263 | 0.773 | 0.948 | 0.799 | - | - |
| C: expansion capped at 5 chunks | 0.263 | 0.773 | 0.958 | 0.824 | 5.00 | 1764 |

- **No regression in which documents are retrieved**: with expansion (B) the documents in the context are
  identical to B3's, so all four metrics are unchanged. The cost is a larger context: +75% chunks, +59% characters.
- Looking only at the first 5 chunks of B's context, nDCG drops from 0.835 to 0.799 (CIs overlap), because
  siblings take some of the first slots; the LLM receives the whole context, so the document-level row is the
  relevant one. The capped variant (C) has a small real drop (nDCG 0.824) and was not used.
- Benefit: on the 24 queries with a relevant fault record, the record's logged cause is in the context for 23/24
  with expansion vs 6/24 without.
- KB answers (B6) were not re-generated; the KB results of run 1 stand.

### Fault events: run 1 vs run 2 (GreenRAG changed; other systems' run-1 answers unchanged, no expansion)

Logged mode (22 events):

| system | run | faithfulness | hallucination rate | answer relevance |
|---|---|---|---|---|
| llm_only | 1 | n/a | 1.000 | 0.750 |
| basic_rag | 1 | 0.580 | 0.818 | 0.102 |
| hybrid_rag | 1 | **0.902** [0.803, 0.985] | **0.227** [0.045, 0.409] | 0.000 |
| greenrag | 1 | 0.288 [0.167, 0.424] | 0.955 [0.864, 1.000] | 0.602 |
| greenrag | **2** | 0.758 [0.652, 0.864] | 0.409 [0.227, 0.636] | 0.636 |

Unlogged mode (22 events):

| system | run | faithfulness | hallucination rate | answer relevance |
|---|---|---|---|---|
| llm_only | 1 | n/a | 0.955 | 0.750 |
| basic_rag | 1 | 0.617 | 0.818 | 0.023 |
| hybrid_rag | 1 | **0.795** [0.689, 0.894] | **0.364** [0.182, 0.545] | 0.011 |
| greenrag | 1 | 0.167 [0.076, 0.258] | 0.955 [0.864, 1.000] | 0.534 |
| greenrag | **2** | 0.629 [0.492, 0.758] | 0.591 [0.409, 0.773] | 0.557 |

- **GreenRAG improves substantially**: faithfulness +0.470 (logged, paired over 22 events, 95% CI [+0.333,
  +0.606]) and +0.462 (unlogged, [+0.326, +0.598]); hallucination rate -0.545 (logged, [-0.773, -0.318]) and
  -0.364 (unlogged, [-0.591, -0.136]).
- **GreenRAG still underperforms hybrid RAG on faithfulness and hallucination** in both modes (logged 0.758 vs
  0.902 and 0.409 vs 0.227; unlogged 0.629 vs 0.795 and 0.591 vs 0.364). Hybrid RAG's advantage comes largely from
  abstaining ("the context does not say"), which is faithful but answers nothing (relevance 0.000-0.011);
  GreenRAG answers the question (relevance 0.56-0.64) and pays for it in unsupported claims.
- What remains unsupported in GreenRAG run 2 (22 answers with 35 unsupported claims): 17 LLM content claims
  (invented actions such as "replace the hydraulic pump", invented causes such as "wear and tear over time", or
  mixing in other events of the same turbine), 13 truthful process statements from the deterministic fallback
  ("the record ... was retrieved") that the judge counts as unsupported, and 5 fallback content claims. 32 of 44
  run-2 reports were LLM-written, 12 used the fallback.
- Hybrid RAG was not given expansion in run 2, so this is not a like-for-like test of expansion itself; the
  comparison is GreenRAG-after-fixes against the run-1 baselines, as requested.

## B6 follow-up: fair expansion for hybrid RAG, and fixing how process statements are scored

Files: `results/rag/run2/b6_followup_summary.md` (hybrid + expansion, judge v2), `b6_followup_deterministic.md`
(final scoring), `b6_followup_template_claims.csv`. Runs: `python -m experiments.b6_followup`,
`python -m experiments.b6_followup_report`.

### 1. Hybrid RAG with the same sibling-chunk expansion (fault items, regenerated)
With expansion, hybrid RAG stops abstaining (answer relevance 0.000 -> 0.409 logged) and starts asserting causes;
its faithfulness falls (logged 0.902 -> 0.572; unlogged 0.795 -> 0.455) and hallucination rises (logged 0.227 ->
0.455; unlogged 0.364 -> 0.909). Its run-1 numbers were good largely because it declined to answer.

### 2. Judge prompt fix (v2): tried, failed its calibration, not used
Judge v2 adds the question and the list of retrieved documents to the prompt and states how retrieval/input
statements are checked. On 140 known-label items it recognised **only 34% of true process statements** (11/32),
and became stricter overall (supported claims recognised 0.677, unsupported 1.000). It raised GreenRAG's logged
hallucination rate (0.409 -> 0.545) instead of correcting it. A judge that fails the calibration for the very
statements it was meant to fix is not a valid measurement fix, so v2 numbers are reported in
`b6_followup_summary.md` but not used.

### 2b. What is used instead: template statements verified by code
GreenRAG's fallback writes a small set of statements from templates filled in by code from the actual evidence
("The record of this incident (X) was retrieved, but the retrieved text does not state the logged cause",
"no troubleshooting procedure was retrieved", "Logged description for this incident: ...", "Anomaly signal for
Wind Farm A turbine N; evidence points to ..."). Their truth is checked deterministically against the retrieved
evidence (`rag_eval.verify_template_claim`, true and false cases tested); all other claims keep the original judge
(v1) verdict. This changes only the measurement, not the answers.

- GreenRAG run 2 contained **35 template statements; all 35 are true**. The v1 judge had marked **18** of them
  unsupported (and none of the false ones supported, since there were none). The earlier estimate of "13 process
  statements" came from a keyword rule that missed some template forms; the code check gives 18.
- The remaining unsupported GreenRAG claims are the **17 invented content claims** (LLM embellishment).
- GreenRAG hallucination rate: logged **0.409 -> 0.227**, unlogged **0.591 -> 0.227**. So in logged mode 0.182 of
  the 0.409 (4 of 9 flagged answers) and in unlogged mode 0.364 of the 0.591 (8 of 13) came from miscounted true
  statements; the remaining 0.227 (5 answers per mode) is genuine invention.

### Final fault-event comparison (template statements verified by code, judge v1 otherwise)

| mode | system | faithfulness | hallucination rate | answer relevance |
|---|---|---|---|---|
| logged | greenrag run 2 | 0.848 [0.742, 0.939] | 0.227 [0.091, 0.409] | 0.636 |
| logged | hybrid run 1 (no expansion) | 0.902 [0.803, 0.985] | 0.227 [0.045, 0.409] | 0.000 |
| logged | hybrid + expansion | 0.572 [0.470, 0.682] | 0.455 [0.273, 0.682] | 0.409 |
| unlogged | greenrag run 2 | 0.833 [0.697, 0.940] | 0.227 [0.045, 0.409] | 0.557 |
| unlogged | hybrid run 1 (no expansion) | 0.795 [0.689, 0.894] | 0.364 [0.182, 0.545] | 0.011 |
| unlogged | hybrid + expansion | 0.455 [0.360, 0.549] | 0.909 [0.773, 1.000] | 0.080 |

- **Like for like (both with expansion), GreenRAG is more faithful than hybrid RAG**: faithfulness +0.277 (logged,
  paired 95% CI [+0.148, +0.398]) and +0.379 (unlogged, [+0.231, +0.527]); hallucination -0.227 (logged,
  CI [-0.500, +0.045], not resolved) and -0.682 (unlogged, [-0.909, -0.409]).
- **Against hybrid RAG without expansion** (which mostly abstains), GreenRAG is roughly level on faithfulness and
  hallucination (logged 0.848 vs 0.902, 0.227 vs 0.227; unlogged 0.833 vs 0.795, 0.227 vs 0.364; CIs overlap)
  while actually answering the question (relevance 0.56-0.64 vs about 0).
- Caveats: the code check only applies to GreenRAG because only GreenRAG emits templated statements; hybrid RAG's
  claims are all free text and are scored by the judge alone. Content claims of all systems still depend on the
  v1 judge, whose accuracy on paraphrased real claims is not measured (see judge limitations). GreenRAG still
  invents content in 5 of 22 answers per mode.
