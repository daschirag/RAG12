# run_all status

Generated 2026-10-16T10:41:15+00:00 by `python -m experiments.run_all`. Complete steps are reused, never re-run. Numbers live in the linked summaries (real runs only).

## Experiment 1: Forecasting (solar A1-A6, wind T1)

Summaries: `results/forecasting/a5_summary.md`, `results/forecasting/wind/wind_summary.md`

| step | state | detail | command |
|---|---|---|---|
| A1 preprocessing | complete | outputs present | `python -m experiments.preprocess_solar` |
| A2 features | complete | outputs present | `python -m experiments.features_solar` |
| A3 models | complete | outputs present | `python -m experiments.train_forecasters` |
| A4 uncertainty | complete | outputs present | `python -m experiments.uncertainty_solar` |
| A5 rolling origin | complete | outputs present | `python -m experiments.rolling_origin` |
| A5 evaluation | complete | outputs present | `python -m experiments.evaluate_forecasting` |
| Wind rolling origin | complete | outputs present | `python -m experiments.wind_rolling` |
| Wind evaluation | complete | outputs present | `python -m experiments.wind_evaluate` |

## Experiment 2: Retrieval (B2 chunking, B3 retrievers)

Summaries: `results/kb/b2_chunking_summary.md`, `results/kb/b3_retrieval_summary.md`

| step | state | detail | command |
|---|---|---|---|
| B2 chunking comparison | complete | outputs present | `python -m experiments.chunking_comparison` |
| B3 retrieval comparison | complete | outputs present | `python -m experiments.retrieval_comparison` |

## Experiment 3: Fault diagnosis (B4 detectors, B5 diagnosis -> RAG)

Summaries: `results/faultdiag/b4_summary.md`, `results/faultdiag/b4_if_threshold.md`, `results/faultdiag/b5_summary.md`

| step | state | detail | command |
|---|---|---|---|
| B4 fault diagnosis | complete | outputs present | `python -m experiments.fault_diagnosis` |
| B4 IF threshold | complete | outputs present | `python -m experiments.fault_threshold` |
| B5 fault RAG | complete | outputs present | `python -m experiments.fault_rag` |

## Experiment 4: Hallucination (B6 RAG evaluation)

Summaries: `results/rag/b6_summary.md`, `results/rag/run2/b6_run2_summary.md`, `results/rag/run2/b6_followup_summary.md`

| step | state | detail | command |
|---|---|---|---|
| B6 run 1 | complete | outputs present | `python -m experiments.rag_evaluation` |
| B6 run 2 | complete | outputs present | `python -m experiments.rag_evaluation --run2` |
| B6 follow-up | complete | outputs present | `python -m experiments.b6_followup` |

## Experiment 5: Cyberattack on telemetry (C2 FDI/spoofing, C3 detector, C6 telemetry)

Summaries: `results/security/c2_summary.md`, `results/security/c3_summary.md`, `results/security/c6_summary.md`

| step | state | detail | command |
|---|---|---|---|
| C2 telemetry attacks | complete | outputs present | `python -m experiments.attack_simulation --part telemetry` |
| C3 telemetry detector | complete | outputs present | `python -m experiments.detector_evaluation --part telemetry` |
| C6 telemetry re-run (CPU) | complete | outputs present | `python -m experiments.security_evaluation --part telemetry` |

## Experiment 6: KB poisoning (C2 knowledge, C3 trust/filter, C4 secure pipeline, C5 engine, C6)

Summaries: `results/security/c2_summary.md`, `results/security/c3_summary.md`, `results/security/c4_summary.md`, `results/trust/c5_summary.md`, `results/security/c6_summary.md`

| step | state | detail | command |
|---|---|---|---|
| C2 knowledge, qwen | complete | 594/594 items | `python -m experiments.attack_simulation --part knowledge` |
| C2 knowledge, llama | complete | 594/594 items | `python -m experiments.attack_simulation --part knowledge --model llama3.2:3b` |
| C3 text + trust | complete | outputs present | `python -m experiments.detector_evaluation --part text` |
| C3 evidence filter | complete | outputs present | `python -m experiments.detector_evaluation --part filter` |
| C4 secure pipeline, llama | complete | 330/330 items | `python -m experiments.secure_pipeline_eval --model llama3.2:3b` |
| C5 decision engine (llama capture) | complete | 330/330 items | `python -m experiments.decision_engine_eval --llm llama3.2:3b` |
| C6 capture, qwen | complete | 330/330 items | `python -m experiments.security_evaluation --part capture` |
| C6 qwen judge | complete | run record 'judge' in security/c6_run.json | `python -m experiments.security_evaluation --part judge` |

## Experiment 7: Prompt injection (C2 query/KB injection, C3 detector, C6 held-out phrasings)

Summaries: `results/security/c2_summary.md`, `results/security/c3_summary.md`, `results/security/c6_summary.md`

| step | state | detail | command |
|---|---|---|---|
| C6 held-out injections, qwen | complete | 990/990 items | `python -m experiments.security_evaluation --part heldout` |
| C6 report | complete | outputs present | `python -m experiments.security_evaluation --report` |

## Experiment A-F: Ablation configs A-F (orchestrator, src/backend/orchestrator.py)

| step | state | detail | command |
|---|---|---|---|
| Ablation A-F | manual | no ablation experiment has been designed or run: needs an LLM run per config (A-F) on a fixed item set; not started without a decision on items and budget | `python -m experiments.ablation` |

