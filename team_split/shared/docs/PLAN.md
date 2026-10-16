# GreenRAG-Sec implementation plan

Source: `docs/GreenRAG-Sec_Work_Split.docx`. One phase at a time; each phase ends with its tests passing,
PROGRESS.md updated, and a stop for user approval ("next"). No git commands are run; all changes are left
uncommitted for the user.

## Phase 0 - Setup
Read the work-split doc; write CLAUDE.md, docs/PLAN.md, PROGRESS.md; create packages
`src/backend/{forecasting,faultdiag,kb,security,trust,evaluation}`, `experiments/`, `results/`, `tests/`;
install pytest; regression tests for existing endpoints.

## Track A - Forecasting (Minmini)
- **A1 Preprocessing**: missing values, duplicates, outliers, noise, time sync, normalization. Solar, plant-level first.
- **A2 Features**: hour, day, month, day-of-week, lags 1/3/6/24, rolling mean/std, weather features. State and justify
  the resampling interval. Test that no future information leaks.
- **A3 Models**: persistence, Random Forest, XGBoost, LSTM, one GPU-light advanced temporal model (justified).
- **A4 Uncertainty**: prediction interval + confidence (e.g. quantile regression + conformal calibration);
  empirical coverage and interval width.
- **A5 Evaluation**: MAE, RMSE, MAPE, R2, coverage, width; comparison table and grayscale plots in
  `results/forecasting/`; honest error analysis (dataset is only ~34 days).
  - **Rolling-origin (expanding-window) evaluation** across the 34 days: for each successive test day d
    (after a minimum initial training window), train on all days before d (with the same val/calib carve-out
    rule at the end of the window), forecast day d, and collect the day-d errors. This gives one out-of-sample
    score per day and per model instead of a single 5-day test block.
  - **Paired day-level bootstrap** for model differences: for each pair of models, compute the per-day mean
    loss difference on the same days, resample days with replacement (fixed seed, e.g. 10,000 resamples), and
    report the mean difference with a 95% percentile CI and the share of resamples favouring each model.
    Days, not 15-min rows, are the resampling unit because errors within a day are strongly autocorrelated.
- **A6 API**: `POST /forecast` -> prediction, lower_bound, upper_bound, confidence; API + schema tests.
- Wind T1 replication only after user approval.

## Track B - RAG + fault diagnosis + explainability (Tarunica)
- **B1 Knowledge base**: manuals, maintenance records, fault descriptions, troubleshooting, standards summaries,
  historical SCADA knowledge, built from event_info.csv plus labelled synthetic docs; provenance metadata on every doc.
- **B2 Chunking comparison**: fixed-size vs semantic vs metadata-aware; labelled query set (>= 50 queries,
  relevance known by construction); Precision@5, Recall@5, MRR, nDCG.
- **B3 Retrieval**: query classifier (conceptual vs quantitative); BM25, vector, hybrid (rank fusion),
  hybrid + cross-encoder rerank; compare all four.
- **B4 Fault diagnosis ML**: features, anomaly detection, classification, probability; >= 3 models
  (accuracy, precision, recall, F1, AUC); split by asset/event. Taxonomy derived from event_info.csv.
- **B5 Fault -> RAG**: diagnosis -> RAG query -> retrieve manual/maintenance/fault/standard -> LLM ->
  pydantic-validated output: Diagnosis, Confidence, Likely Cause, Evidence (real chunk ids), Recommended Action,
  Severity (deterministic mapping).
- **B6 RAG evaluation**: LLM-only vs basic RAG vs hybrid RAG vs GreenRAG; faithfulness, answer relevance, context
  relevance, hallucination rate. Numeric facts by exact match vs data-derived ground truth; free text by local LLM
  judge (limitations documented).
- **B7 API**: `POST /diagnose` + tests.

## Track C - Security + trust (Pritika)
- **C1 Threat model** `docs/threat_model.md`: per layer (IoT/SCADA, ingestion, cloud, database, vector DB,
  retriever, LLM, decision engine): attack, vulnerability, impact, detection, mitigation.
- **C2 Attack simulator** (local, own data only): false data injection, sensor spoofing, KB poisoning, prompt
  injection (queries, retrieved docs, maintenance records, KB entries); effect on forecasting, anomaly detection,
  diagnosis, RAG answers.
- **C3 Detectors**: telemetry anomaly/spoofing (physical bounds, rate of change, cross-sensor consistency,
  statistical model), prompt-injection detector, document trust verification, evidence filtering.
- **C4 Secure retrieval**: query -> input security -> injection detection -> retriever -> doc trust -> evidence
  filter -> LLM.
- **C5 Trust-aware decision engine**: LLM recommendation -> evidence verification -> security verification ->
  operational constraints -> risk -> trust score -> accept/review/reject. `T = w1*E + w2*R + w3*C + w4*S`
  (configurable weights, sensitivity analysis). LLM output never acts as a decision directly.
- **C6 Security evaluation**: normal vs attacked vs secure; detection rate, FPR, attack success rate, detection
  time, mitigation success, robustness, hallucination under attack; multiple seeds.
- **C7 API**: `POST /secure-ask`, `POST /decision` + tests.

## Integration and experiments
- **I1 Orchestrator**: forecast + diagnosis + security + trust with feature flags for ablation configs A-F
  (A LLM only; B +RAG; C +hybrid retrieval; D +forecasting; E +fault diagnosis; F full GreenRAG-Sec);
  `POST /assistant` routes by intent and returns the trust score.
- **I2 Console UI**: keep design tokens; add forecast interval chart, diagnosis card with severity, trust score.
- **I3 Experiments 1-7 + ablation**: `python -m experiments.run_all`; CSV, markdown tables, grayscale plots in
  `results/`; real runs only.
  1 forecasting, 2 retrieval, 3 fault diagnosis, 4 hallucination, 5 cyberattack (FDI), 6 KB poisoning,
  7 prompt injection.
- **I4 Cloud portability**: local vs Azure functional equivalence + latency harness; PENDING if no Azure keys
  (never simulated).
- **I5 Final**: full test suite, `docs/RESULTS.md`, README, honest limitations (small solar dataset, synthetic KB,
  wind faults vs inverter faults, local LLM judge), release tag.
