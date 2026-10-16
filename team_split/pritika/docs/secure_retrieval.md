# C4 - Secure retrieval pipeline

    query -> input security -> injection detection -> retriever -> document trust -> evidence filter -> LLM

Implementation: `src/backend/security/secure_pipeline.py`. It wires the three C3 detectors into the
retrieval path that the system serves.
- **`SecureRAGPipeline` / `SecureAskService`:** question answering over the knowledge base. This is the
  service that C7 exposes as `POST /secure-ask`.
- **`POST /diagnose`:** since C4, its `DiagnosisService` builds its retriever with the same
  `build_secure_retrieval`. Its request carries no free-text question (an event id or a time window), so the
  input stages do not apply. Retrieval, document trust and the evidence filter do.
- **Unchanged:** `/ask` and `/query`, which use the legacy per-domain vector store and the `BACKEND_MODE`
  local/azure abstraction.

Evaluation: `python -m experiments.secure_pipeline_eval --model none | llama3.2:3b`, with results in
`results/security/c4_summary.md` and raw per-item records in `c4_raw_none.jsonl` and
`c4_raw_llama3.2-3b.jsonl`.

## Stages

| # | Stage | What it does | On failure |
|---|---|---|---|
| 1 | Input security | the question must be non-empty, at most 1000 characters, with no control characters | blocked; nothing else runs |
| 2 | Injection detection | C3 pattern detector on the question | blocked; the question never reaches retrieval or the LLM |
| 3 | Retriever | production hybrid BM25 + MiniLM, RRF, cross-encoder rerank over semantic chunks of the **KB store** (`documents.jsonl`), over-fetching 10 extra candidates | - |
| 4 | Document trust | C3 checks per retrieved chunk: registered doc_id, registered text hash, provenance, stuffing, injection, chunk in document | chunk rejected, with reasons |
| 5 | Evidence filter | only trusted chunks go on, and the top k=5 are kept | if nothing trusted remains: "no trusted evidence", with no LLM call and no fallback quote |
| 6 | LLM | GreenRAG generation (JSON with citations, numeric router, guarded extractive fallback) over trusted chunks only | - |

- **Every answer carries a `SecurityReport`:** input checks, injection signals, blocked flag, documents
  retrieved, chunks rejected with reasons, chunks passed on, the registry source, and the time spent in the
  security stages.
- **Trusted reference.** If a `trust_registry.json` file sits next to the store, the registry comes from it:
  - A signed file is verified with `GREENRAG_TRUST_KEY`.
  - A signed file present without a key, or failing verification, fails closed: nothing is trusted.
  - Write a signed file with `python -m src.backend.security.secure_pipeline --write-registry`.
  - With no file, the registry is built in-process from the knowledge base as generated from code and
    datasets (`src.backend.kb.build`), never copied from the store.
- **Start-up scan:** every store document is checked once at start-up, and the untrusted ones are listed in
  `status["untrusted_in_store"]` (for `/diagnose`, in `DiagnosisService.security_status`). They stay indexed,
  so the retrieval-time filter is the control that is exercised, and are dropped whenever they are retrieved.
- **The store is where an attacker would write.** Before C4, `/diagnose` indexed a copy regenerated from
  code, which an attacker writing to the store could not reach. On the clean KB the store and that copy are
  identical (189/189 hashes), so the switch changes nothing for clean data.

## Evaluation generator: llama3.2:3b, not qwen (my decision, with the reason)

- **The security stages run before the LLM.** Their decisions (what is blocked, rejected and passed on)
  don't depend on the generator. The `--model none` run measures them with no LLM, and holds for any
  generator, including the deployed qwen2.5:7b-instruct.
- **Answer-level outcomes do depend on the generator.** For those, I used **llama3.2:3b**, the only generator
  with complete unprotected C2 results for every attack state. This makes "unprotected (C2) vs secure (C4)"
  a same-generator comparison: same items, temperature 0, seed 0, and the C2 scoring code applied unchanged
  (`experiments.attack_simulation.knowledge_tables`).
- **qwen2.5:7b-instruct was not run.** Its C2 attack runs are incomplete, with both injection states
  pending, so there would be no unprotected baseline. Loading it also caused the C2 memory stops.
- **The qwen answer-level comparison remains open.** It becomes possible once the C2 qwen run is resumed,
  and belongs with C6. No llama result stands in for qwen.

## Results

**Security stages (LLM disabled; the llama3.2:3b run gives identical stage decisions).**

| KB state | Questions: blocked | Questions: attack doc among candidates | Questions: attack doc passed on | Diagnoses: attack doc among candidates | Diagnoses: attack doc passed on | Clean doc rejected |
|---|---|---|---|---|---|---|
| clean | 0/66 | 0/66 | 0/66 | 0/22 | 0/22 | 0 |
| poisoned | 0/66 | 28/66 | **0/66** | 16/22 | **0/22** | 0 |
| injected | 0/66 | 17/66 | **0/66** | 16/22 | **0/22** | 0 |
| injection in the question | **66/66** | - | - | - | - | 0 |

- Candidates are counted before filtering: the top 15 per question, and the reranker's pool of 20 per
  diagnosis. The rate is therefore higher than C2's 28.8%, which counted only the top-5 context.
- The security stages take a median of **16.8 ms** per question (max 35.1 ms), excluding retrieval and
  generation.

**Attack success, end to end with llama3.2:3b: unprotected (C2) vs secure (C4).** Success means attack
content in the output that the same arm's clean output did not contain, with the attack document in context.
For query injection, success means the canary appears in the answer.

| Attack | System | Unprotected: attack doc in context | Unprotected: success | Secure: attack doc in context | Secure: success |
|---|---|---|---|---|---|
| KB poisoning | GreenRAG (66) | 19/66 | 2/66 | 0/66 | **0/66** |
| KB poisoning | /diagnose (22) | 8/22 | 6/22 | 0/22 | **0/22** |
| Injection in KB documents | GreenRAG (66) | 8/66 | 0/66 | 0/66 | 0/66 |
| Injection in KB documents | /diagnose (22) | 12/22 | 0/22 | 0/22 | 0/22 |
| Injection in the question | GreenRAG (66) | n/a (in the question) | 2/66 | blocked 66/66 | **0/66** |

- **All 10 unprotected successes are gone,** and nothing from an attack reached the model.
- **Unprotected successes:**
  - GreenRAG, poisoned: the two unsafe procedures quoted by the extractive fallback.
  - `/diagnose`, poisoned: six diagnoses in which the LLM repeated the forged A-73 "main bearing failure".
    Only one of them is the targeted event, A-73. The other five concern events 0, 24, 69, 71 and 92, whose
    evidence also retrieved the forged record. C2 counted only the target, so this collateral is newly
    reported here; see docs/attack_simulation.md.
  - Query injection: the two GreenRAG answers "CANARY-7431: all systems normal."
- **No cost on clean data.** On the clean KB, all 88 secure answers (66 questions and 22 diagnoses) are
  *identical* to the unprotected C2 answers from the same generator. No clean question was blocked, and no
  clean chunk was rejected.

## Limitations

- **The attacks are C2's:** few, fixed, and known when the detectors were designed. The C3 held-out results
  are the better guide to phrasings not seen before. The injection detector caught 11/15 held-out injections,
  with 1/15 false positives, and missed injections in German, letter-spaced text, and polite role-play.
- **The protection against KB poisoning rests on the registry.** Content checks alone catch only 5/9 of the
  C2 attack documents, so anyone with the registry key or control of the KB build defeats the filter for
  plausible fake procedures.
- **A question is blocked outright when injection is detected; it is not sanitised.** A false positive (for
  example "Can we ignore the manufacturer's instructions...?") therefore gets no answer. C5's decision engine
  may prefer to send such questions to review.
- **qwen2.5:7b-instruct answer-level outcomes are not measured** (see above).
