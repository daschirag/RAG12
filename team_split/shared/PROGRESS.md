# GreenRAG-Sec progress

Phase list: `docs/PLAN.md`. Rules: `CLAUDE.md`.

| Phase | Status | Tests | Commit |
|---|---|---|---|
| 0 Setup | done | 38 passed, 0 skipped (with Ollama running) | e043ac1 + Ollama fix commit |
| A1 Solar preprocessing | done | 19 new (57 total) passed | see git log "A1" |
| A2 Feature engineering | done | 12 new (69 total) passed | 3f2a690, 6063449 + results commit |
| A3 Forecasting models | done | 17 new (86 total) passed | uncommitted (no-git rule) |
| A4 Uncertainty | done | 21 new (107 total) passed | uncommitted (no-git rule) |
| A5 Evaluation | done | 9 new (116 total) passed | uncommitted (no-git rule) |
| A6 Forecast API | done | 17 new (133 total) passed | uncommitted (no-git rule) |
| Wind T1 replication (Track A) | done | 21 wind tests; full suite 190 passed, 1 skipped (Ollama down) | uncommitted |
| B1 Knowledge base | done | 22 new passed (targeted); full suite deferred until the wind run ends | uncommitted |
| B2 Chunking comparison | done | 15 new passed; full suite 190 passed, 1 skipped | uncommitted |
| B3 Retrieval comparison | done | 18 new (209 total) passed, 0 skipped | uncommitted |
| B4 Fault diagnosis (Farm A) | done | 11 new (220 total) passed | uncommitted |
| B5 Fault diagnosis -> RAG | done | 18 new + 2 threshold tests (240 total) passed | uncommitted |
| B6 RAG evaluation (run 1 + run 2) | done | 12 new + 1 timeout test (252 total) passed | uncommitted |
| B7 /diagnose API | done | 10 new (271 total) passed | uncommitted |
| C1 Threat model | done | 9 new (280 total) passed | uncommitted |
| C2 Attack simulator | done (qwen knowledge run complete 2026-10-15) | 19 new; full suite 343 passed (Ollama up, 0 skipped) | uncommitted |
| C3 Detectors | done | 17 new; suite 313 passed, 2 ollama deselected | uncommitted |
| C4 Secure retrieval | done (qwen end-to-end pending) | 10 new; suite 323 passed, 2 ollama deselected | uncommitted |
| C5 Decision engine | done (qwen under attack unmeasured) | 17 new; full suite 342 passed (Ollama up) | uncommitted |
| C6 Security evaluation | done (2026-10-16) | full suite 383 passed, 0 skipped (Ollama up) | uncommitted |
| C7 /secure-ask + /decision | done | 15 new (test_secure_api.py) + 40 related passed | uncommitted |
| I1-I3 Orchestrator, console, run_all | done | 16 orchestrator/frontend + 4 run_all passed | uncommitted |
| I4 Cloud portability | PENDING (no Azure configuration) | 1 new passed | uncommitted |
| I5 Final docs | done (C6 numbers added after the C6 runs) | docs only | uncommitted |
| Product polish (backend + console + team docs) | done 2026-10-16 | 27 new; full suite 410 passed | uncommitted |

## Phase 0 - Setup (2026-09-28)
- Spec found at `<local download path>`, copied locally to
  `docs/GreenRAG-Sec_Work_Split.docx`. The Trust Score formula in the docx is an equation object that does not
  extract as text; using `T = w1*E + w2*R + w3*C + w4*S` from the user's instructions.
- Created packages `src/backend/{forecasting,faultdiag,kb,security,trust,evaluation}`, `experiments/`,
  `results/{forecasting,retrieval,faultdiag,rag,security,ablation,cloud}/`, `tests/`.
- Installed pytest 9.1.1 via uv; added to requirements.txt (also committed the pending xgboost, matplotlib,
  statsmodels additions).
- Added `src/backend/runtime_env.py`: forces HF_HOME/TORCH_HOME under `D:\Rag\cache` (was pointing at another
  project's cache). all-MiniLM-L6-v2 re-downloaded there (~90MB).
- Regression tests: `tests/test_api_regression.py` (all existing endpoints, fakes), `test_backend_mode.py`
  (local/azure factory), `test_frontend.py` (UI assets, ids, design tokens, routes), `test_local_integration.py`
  (real MiniLM + Chroma in a temp dir), `test_ollama_integration.py` (real LLM; skips when Ollama is down).
- Result: `python -m pytest` -> 33 passed, 1 skipped (Ollama unreachable at 127.0.0.1:11434).

## Phase 0 follow-up - Ollama check (2026-09-28)
- `/api/tags` lists qwen2.5:7b-instruct, llama3.2:3b, nomic-embed-text. Ollama 0.34.3, RTX 4050 Laptop, driver 592.82.
- First run of `tests/test_ollama_integration.py` FAILED (not a timeout): Ollama's llama-server crashed while loading
  the model: `exit status 0xc0000409 ... CUDA error: shared object initialization failed` (HTTP 500).
- Reproduced by forcing cold loads (keep_alive=0): 1 failure in 5, then 0 in 8, i.e. 2/14 cold loads including the
  first call. Warm calls never failed. A direct /api/generate call right after the crash succeeded.
- Cause is in Ollama's CUDA runner (outside this repo). Fix: `LocalGenerator` retries exactly once when Ollama reports
  "llama-server process has terminated" (other errors are not retried); applies to /ask answers and numeric
  enrichment. Unit tests in `tests/test_generator_retry.py`. The integration test was not changed.
- After the fix: Ollama test passed on a cold load (9.78s); 10/10 further cold-load runs passed (7.7-27.1s; the 27s
  run may have included a retry, not verifiable from pytest output). Full suite: 38 passed.

## A1 - Solar preprocessing (2026-09-28)
- `src/backend/forecasting/preprocess.py`: load (explicit timestamp formats) -> duplicates -> time sync (15-min grid,
  +/-2 min snap) -> plant aggregation (mean over reporting inverters x 22, coverage kept, NaN below 50%) -> physical
  bounds + night consistency -> causal missing-value handling -> trailing robust-z outlier flags (values kept) ->
  causal smoothed weather copies. `chronological_split` (whole days) and `TrainOnlyScaler` (fit on train only).
  `src/backend/evaluation/provenance.py` records package versions + git commit in every report.
- Run: `python -m experiments.preprocess_solar` -> results/forecasting/preprocessing_{report_plant1,report_plant2}.json,
  preprocessing_summary.md, preprocessing_plant{1,2}.png. Reports are stamped `f64e76f-dirty` = produced by the
  uncommitted A1 code that is committed together with them.
- Measured: Plant 1: 3264 slots, 106 without generation, 82 without weather, 22 night-filled, 118 target NaN,
  2 outlier flags, DC/AC ratio 10.22 (DC scale inconsistency; DC not used), daytime corr(AC, irradiation) 0.9876.
  Plant 2: 904 partial-coverage slots, 6 target NaN, 1 night-inconsistent, 10 outlier flags, corr 0.7319.
  Plant 1 = primary modelling series; Plant 2 = harder secondary series.
- Tests: `tests/test_forecasting_preprocess.py` 19 passed, including a prefix-invariance causality test (verified to
  fail on two injected leaks: centered smoothing and whole-series outlier stats). Full suite 57 passed.
- Design notes: `docs/forecasting.md`.

## A2 - Feature engineering (2026-09-28)
- A1 reports re-generated from committed code: stamp ca8c808, report/config/package blocks identical to the
  previous run (commit 40f716c).
- **Resolution: native 15 min (no resampling). Horizon: 1 hour ahead (4 steps), fixed for all A3 models.**
  Justification in docs/forecasting.md (34 days is too short to downsample; no NWP data for longer horizons).
- `src/backend/forecasting/features.py`: origin-indexed rows; features use data <= t (lags 1/3/6/24 steps, same slot
  yesterday, rolling mean/std over 4 and 16 slots, weather at t, irradiation envelope from previous days, calendar of
  T). Measured weather at T only as opt-in `oracle_irr_target` (labelled upper bound, off by default).
- Split by target day: train 21 / val 4 / calib 4 (reserved for A4) / test 5 days. Plant 1 rows
  1901 / 379 / 384 / 478 (daytime 905 / 188 / 188 / 234).
- Tests: `tests/test_forecasting_features.py` 12 passed (future-corruption leakage test over 3 seeds + positive
  control on the oracle feature). Full suite 69 passed. First run had 5 failures: a test-helper dtype bug and a
  feature-description ordering mismatch; both fixed and re-run.
- Provenance now ignores results/ when marking code dirty (commit 6063449); A2 results stamped 6063449.
- Carried into A3: score every model on the same test rows (train-fitted imputation, no row dropping); small test set.

## Rule change (2026-09-28)
- User rule: no git commands; leave all changes uncommitted. Added to CLAUDE.md (rule 11); commit steps removed from
  CLAUDE.md rule 1 and docs/PLAN.md. Existing local commits (through 4e586aa, 7 ahead of origin/feature/dev) untouched.
  `git status -sb` run once at the user's request. `evaluation/provenance.py` now reads the commit from .git files
  (no git subprocess); stamps are `<commit>-dirty` with `git_dirty_checked: false`.

## A3 - Forecasting models (2026-09-28, uncommitted)
- `src/backend/forecasting/models.py`: TrainOnlyImputer (train-split hourly medians), shared postprocess (clip to
  [0, 33,000 kW], night -> 0), persistence, daily persistence, RandomForest, XGBoost (grids chosen on val), LSTM and
  TCN (same sequence inputs, early stopping on val). `forecasting/metrics.py`: MAE, RMSE, R2, MAPE.
- Run: `python -m experiments.train_forecasters` (CPU; torch build has no CUDA) -> results/forecasting/a3_summary.md,
  a3_metrics.csv, a3_predictions_plant{1,2}.csv, a3_run.json. Horizon 1 h, same scored rows for all models, seeds 0-2.
- Result: all learned models beat persistence (test MAE skill 0.34-0.39). No clear winner: RF best mean on Plant 1
  (1454.8 +/- 13.2 kW), LSTM on Plant 2 (1205.4 +/- 10.4 kW); gaps are within seed spread. TCN not better than trees.
  Oracle weather (upper bound, not deployable) cuts Plant 1 MAE to 305.7 kW.
- Known issue: night rule forces about 15 kW extra test MAE on every model (dawn/dusk slots); not tuned on test.
- Tests: `tests/test_forecasting_models.py` 17 passed. First runs: 12 failures (imputer column selection bug), then
  1 (RF parallel-predict float nondeterminism, 1.3e-11 kW); both fixed in code, not by loosening tests. Full suite 86.

## A4 - Uncertainty-aware forecasting (2026-09-28, uncommitted)
- `src/backend/forecasting/uncertainty.py`: multi-quantile XGBoost (grid on val pinball), finite-sample conformal
  quantile, methods qr_raw, gaussian, split_conformal, cqr_pooled, cqr_mondrian (proposed: CQR calibrated per
  day/night group). Calibration on calib days only; test never read (tested with a positive control).
  `forecasting/schemas.py`: ForecastResponse with confidence = nominal coverage level (tested).
  `models.py`: night rule threshold is now a parameter (default 0.05 unchanged; A3 tests pass).
- Night rule decided on val (XGB + RF, seeds 0-2, both plants): strict rule (zero only where envelope <= 0) has the
  lowest mean val MAE (1387.4 vs 1395.3 current vs 1396.3 none). Used in A4; A5 will re-run A3 models with it.
- Result at 90% (daylight coverage / Winkler): Plant 1 cqr_mondrian 0.887 / 14811 (best Winkler and pinball at all
  levels, closest to nominal); Plant 2 cqr_mondrian over-covers (0.957) and qr_raw has the best Winkler/pinball at
  all levels. Pooled CQR correction collapsed to exactly 0 on Plant 1 (night scores exactly 0) -> equals qr_raw.
  All-rows coverage overstates daylight coverage by 4-6 points on Plant 1.
- docs/forecasting.md: confidence definition, exchangeability and 4-day calibration discussion, results.
  docs/PLAN.md A5: rolling-origin evaluation and paired day-level bootstrap added.
- Tests: `tests/test_forecasting_uncertainty.py` 21 passed (known-coverage synthetic checks, calib/test separation
  with positive control, determinism, schema). First run: 1 failure from a wrong hand-computed pinball value in the
  test (code correct); fixed the expectation. Full suite 107 passed.

## A5 - Evaluation (2026-09-29, done, uncommitted)
- Scope extended at user request: rolling origin + paired day bootstrap also cover the A4 interval methods.
- Strict night rule is the code default (`models.DEFAULT_NIGHT_THRESHOLD = 0.0`); A3 and A4 re-run with it
  (old outputs in results/forecasting/archive_night_rule_0.05/). A4 re-run identical to A4 (already strict).
- Rolling origin: 24 folds (12 test days x 2 plants, 06-06..06-17), all models refit per fold, seeds 0-2.
  First attempt stopped by Claude Code for low system memory after 6 folds; resumed at user request and
  completed (exit 0). a5_folds/run_log.json only has fit times for the 18 folds of the resumed run.
- `python -m experiments.evaluate_forecasting` -> results/forecasting/a5_summary.md + CSVs + 5 grayscale figures.
- Results: learned models beat persistence on 12/12 days (both plants). Among learned models only RF > XGBoost on
  Plant 1 is resolved (-63.5 kW, CI [-99.4, -32.7]). TCN not better than trees.
  **Plant 1 cqr_mondrian 90% daylight coverage: fixed split 0.887 (short by 1.3 points); rolling 0.893 (short by
  0.7 points, 95% CI [0.853, 0.927], includes 0.90).** qr_raw Plant 1: 0.846, CI [0.797, 0.879] (under-covers).
  cqr_mondrian - qr_raw Winkler at 90%: Plant 1 -189 kW CI [-1127, 548]; Plant 2 +228 kW CI [-47, 527];
  neither resolved, so the A4 win/lose split is not established. cqr_mondrian clearly beats gaussian and
  split_conformal on Winkler (80%/90%, both plants).
- Tests: `tests/test_forecasting_evaluation.py` 9 passed; full suite 116 passed. Removed pandas fragmentation
  warnings in rolling_origin.py and evaluate_forecasting.py (aggregation output confirmed identical).
- docs/forecasting.md: A5 section (protocol, point and interval results, coverage statement, error analysis).

## A6 - Forecast API (2026-09-29, done, uncommitted)
- POST /forecast {plant, origin, confidence in {0.8, 0.9, 0.95}} -> prediction, lower_bound, upper_bound, confidence
  (+ target_time, observed_kw, in_sample, daylight_group, method, unit, model_version). 422 for invalid requests,
  503 with the build command when the model artifact is missing.
- `forecasting/service.py`: deployed model = XGBoost + multi-quantile XGBoost + cqr_mondrian, strict night rule;
  train 05-15..06-09, val 06-10..06-13, calib 06-14..06-17. Build: `python -m experiments.build_forecast_model`
  (artifacts in models/forecasting/, gitignored; card in results/forecasting/a6_model_card.json).
- `features.build_features(drop_missing_target=False)` keeps the last hour of origins; the envelope at T is now
  computed on a grid extended by one hour (supervised sets unchanged, checked). `uncertainty.cqr_offsets` added.
- Known limitation: small non-zero Plant 2 night forecasts (night irradiation readings slightly above 0).
- Tests: `tests/test_forecast_api.py` 17 passed (incl. API == offline cqr_mondrian, causality with positive
  control); full suite 133 passed. Track A (A1-A6) complete; wind T1 replication awaits user approval.

## Wind T1 replication of Track A (2026-09-29, done, uncommitted)
- Decisions: native 10-min grid, horizon 1 h (6 steps); no plant aggregation; small negative power (>= -10 kW)
  clipped to 0; downtime (power <= 10 kW while power curve >= 500 kW) flagged, kept; no night rule; Mondrian groups
  = predicted power regime (low < 10%, mid, high >= 90% of 3600 kW); rolling origin by month (test Sep-Dec 2018,
  train = earlier months, val = m-2, calib = m-1), seeds 0-2. Scaled settings: RF 200 trees, grid 4; NN batch 256,
  max 60 epochs, patience 8; day/month ablation dropped; oracle-wind row kept.
- Code: `forecasting/wind.py` (preprocess, features, month folds, regimes); shared code generalised with solar
  defaults (postprocess/_val_mae capacity + night threshold; SequenceEncoder columns; build_intervals groups +
  capacity; cqr_offsets groups). /forecast now takes `asset` (solar_plant1/2, wind_t1) or legacy `plant`.
- Runner `experiments/wind_rolling.py` (resumable per fold/model/seed) running; `experiments/wind_evaluate.py` ready.
- Tests: `tests/test_forecasting_wind.py` 21 passed; solar forecasting tests re-run and passing (76 + 17 API).
- Results (rolling origin Sep-Dec 2018, 116 days): persistence MAE 298.1 kW beats every learned model on MAE
  (XGBoost 329.6, TCN 339.8, RF 343.2, LSTM 365.1; bootstrap CIs exclude 0); XGBoost/RF slightly better on RMSE
  (480.3/486.4 vs 490.9). Diagnosis: squared-error models lose on flat hours (30.6% of rows) and at 0/rated power,
  win on ramps. cqr_mondrian 90% coverage 0.890 [0.869, 0.909] (short 1.0 point); qr_raw 0.866; all quantile
  methods under-cover in the predicted-high regime (0.56-0.70). Not tuned on test.
- Deployed models rebuilt (solar_plant1/2 unchanged output, wind_t1 new); /forecast smoke-tested for all assets.
- Full suite 190 passed, 1 skipped (Ollama not reachable at the time of the run).

## B1 - Knowledge base (2026-09-29, done, uncommitted)
- `src/backend/kb/`: schema (pydantic KBDocument, provenance, sha256, validate_kb), taxonomy (event loaders,
  13-component keyword rules, default severities as assumptions, plan-vs-data table), content (hand-written
  synthetic guidance), build (documents, chunks in the ingestion format, report). Manifest `knowledge_base.yaml`.
- Build: 189 documents (99 real: 45 fault descriptions + 54 SCADA knowledge; 90 synthetic: 19 manuals,
  19 troubleshooting, 45 maintenance records, 7 standard overviews) -> 210 chunks. results/kb/kb_summary.md.
- Real taxonomy: 45 anomaly events, all categorised; pitch 8, hydraulic 6, gearbox 5, transformer 5, ...
  The plan's solar-inverter fault list has no direct counterpart in the labelled data (docs/knowledge_base.md).
- Data quirks: degree sign stored as U+FFFD in farms A/B feature_description.csv; 45 anomaly events vs 44 in README.
- Tests: `tests/test_kb.py` 22 passed (+ test_api_regression/test_frontend 23 passed with the new domain).
  Full suite deliberately not run while the wind job is running.

## B2 - Chunking comparison (2026-09-29, done, uncommitted)
- `src/backend/kb/chunking.py`: (A) fixed_size (800 chars, 100 overlap, word boundaries), (B) semantic (sentence
  units embedded; break where adjacent similarity < doc mean - 1 std; 200-1000 chars), (C) metadata_aware
  (paragraph packing + header from KB metadata). Shared `pack_paragraphs` now also used by B1 (chunks.jsonl
  byte-identical, SHA-256 checked).
- `src/backend/kb/queries.py`: 64 labelled queries (event 12, symptom 6, troubleshooting 13, signals 8, count 6,
  solar 6, standard 7, maintenance 6; 58 conceptual / 6 quantitative), graded relevance by construction.
  Known bias: troubleshooting queries quote symptoms that appear verbatim in the manual (grade 1), not in the
  troubleshooting doc (grade 2).
- `src/backend/kb/retrieval_eval.py`: document-level Precision@5, Recall@5, MRR, nDCG@5; exact cosine index;
  query bootstrap CIs; MiniLM embed loader. `experiments/chunking_comparison.py` written, NOT run (waits for wind).
- Tests: `tests/test_kb_chunking.py` 15 passed with a hashing mock embedding (no model download).
- Real run (after the wind job): `python -m experiments.chunking_comparison`, MiniLM, 64 queries, top 5.
  nDCG@5 semantic 0.667 > metadata-aware 0.581 ~ fixed-size 0.574 (semantic better than both, paired CIs exclude
  0); MRR 0.707 / 0.632 / 0.606. Metadata-aware did not beat fixed-size. Confound: chunk sizes differ (366 / 762 /
  570 chars); a size-matched run was not done (would tune on the evaluation set). docs/knowledge_base.md B2.

## B3 - Retrieval comparison (2026-09-29, done, uncommitted)
- Ollama back: tests/test_ollama_integration.py passed (30.99 s); full suite 191 passed, 0 skipped before B3.
- `retrieval/query_classifier.py` (rule-based conceptual/quantitative), `retrieval/hybrid_retriever.py` (BM25,
  RRF hybrid, cross-encoder rerank, routed). Cross-encoder ms-marco-MiniLM-L-6-v2 downloaded (88 MB, D:\Rag\cache).
- Results (64 queries, semantic chunks, top 5), nDCG@5: bm25 0.650, vector 0.667, hybrid_rrf 0.752,
  hybrid_rerank 0.835 (MRR 0.958), routed 0.781. Rerank > hybrid > each single retriever (paired CIs exclude 0).
  Routing (quantitative -> BM25, fixed in advance) hurt: count queries 0.190 with BM25 vs 0.760 reranked.
  Rerank costs ~1 s/query on CPU vs ~36 ms for hybrid. Classifier 64/64 on the labelled set (optimistic).
- Tests: `tests/test_hybrid_retrieval.py` 18 passed (hand-computed BM25/RRF, mocked reranker); full suite 209 passed.

## B4 - Fault diagnosis, Wind Farm A (2026-09-29, done, uncommitted)
- Pre-B4 confirmations: classifier scored 10/10 on the hand-written test examples, but those were written with the
  rules, so no independent accuracy exists yet. Production retrieval = hybrid + rerank, classifier only flags
  numeric answers (stated in docs/knowledge_base.md).
- Fixed a B1 bug: pandas 3 left NaN event descriptions; `categorize` and `load_events` now handle them (KB chunks
  byte-identical after rebuild).
- Leakage finding: Farm A `status_type_id` is 0% normal in every anomaly window, 100% in every normal window while
  the turbine is running -> excluded everywhere.
- Protocol fixed before training (docs/fault_diagnosis.md): robust z vs the turbine's own train part; LR, RF, XGB
  leave-one-asset-out; per-dataset Isolation Forest; event level primary; no supervised component classifier
  (1-6 events per class), unsupervised sensor-group attribution instead.
- Results: event AUC LR 0.742 [0.50, 0.95], RF 0.594, XGB 0.558, IF 0.757 [0.545, 0.953] but IF flags ~nothing at
  the fixed threshold (acc 0.455, F1 0); row-level AUC at chance for supervised models. Attribution top-1 0/12.
  Weak results reported as is; nothing re-tuned on them.
- Tests: `tests/test_faultdiag.py` 11 passed; full suite 220 passed.

## B5 - Fault diagnosis -> RAG (2026-09-30, done, uncommitted)
- IF threshold (B4 follow-up): nested leave-one-asset-out (no fixed 3-way split; IF needs no labels to fit).
  F1 0.783 / 0.783 / 0.560 over seeds 0-2 (mean 0.709); thresholds tiny (0.0015-0.0030) -> fragile operating point.
- Farms B/C recorded as future work in docs/fault_diagnosis.md.
- `faultdiag/diagnosis.py`: DiagnosisReport (pydantic) with Diagnosis, Confidence (score + measured detector
  reliability AUC 0.742 CI [0.500, 0.950] + evidence support; never "high" with this detector), Likely Cause
  (evidence-only, cited chunk ids), Evidence, Recommended Action, Severity (deterministic, only with strong incident
  evidence). Production retrieval hybrid + rerank; Ollama JSON LLM with evidence-only fallback.
- Results (final run, 44 reports): logged mode 12/12 component correct (own record retrieved); unlogged mode 4/12,
  below majority baseline 6/12; normal windows all low confidence / undetermined severity. 28/44 LLM reports
  accepted; accepted text often generic and sometimes unsupported (B6 will measure).
- Post-result fixes, all runs archived: citation parser accepts bare ids of retrieved documents only (run1 -> run2);
  severity undetermined unless strong evidence (run2 -> run3).
- Tests: tests/test_fault_rag.py 18 passed (incl. weak evidence -> low confidence, schema rejects overconfident
  reports, real Ollama test); threshold tests added to tests/test_faultdiag.py. Full suite 240 passed.

## B6 - RAG evaluation (2026-09-30, done, uncommitted)
- Claim-level faithfulness on the full answer text (citations stripped); hallucination = any claim unsupported by
  reference + context or any number absent from them; count questions exact-matched; context relevance from labels.
- Judge: qwen2.5:7b-instruct (calibration balanced accuracy 1.000 vs llama3.2:3b 0.912 on 60 known-label pairs);
  limitations documented (LLM judge not ground truth, self-preference, easy calibration set).
- KB (64): GreenRAG best: faithfulness 0.977, hallucination 0.016 (LLM-only 0.688, basic 0.109, hybrid 0.094),
  count exact match 6/6.
- Fault events (22 x 2 modes): GreenRAG worst RAG system (logged faithfulness 0.288, hallucination 0.955); hybrid
  most faithful but abstains (relevance 0). Cause: semantic chunking splits 44/45 fault docs so the logged cause
  (::1) is not retrieved with the event header (::0) - invisible to B2's document-level metric; plus a GreenRAG
  fallback bug ("not established" despite incident evidence). Reported as measured; fixes need a decision.
- Robustness: Ollama calls time out at 120 s with one retry (a call had hung for 39,097 s); first run was killed for
  low memory and resumed from saved answers.
- docs/rag_evaluation.md. Full suite 252 passed.

## B6 run 2 - fixes for the fault-event root causes (2026-09-30, done, uncommitted)
- SiblingExpander (retrieval-time, adjacent chunks of each retrieved chunk's document) + fallback searches all
  incident chunks for the logged cause. KB not re-chunked; KB B6 answers not re-run.
- Regression check (64 KB queries): documents in context identical to B3 (P@5 0.269, R@5 0.789, MRR 0.958,
  nDCG 0.835 unchanged); context +59% chars; first-5-chunk view nDCG 0.799; fault cause text in context 23/24 vs 6/24.
- Run 2 (GreenRAG fault items): logged faithfulness 0.288 -> 0.758 (+0.470), hallucination 0.955 -> 0.409;
  unlogged 0.167 -> 0.629, 0.955 -> 0.591. Still below hybrid RAG (logged 0.902 / 0.227; unlogged 0.795 / 0.364),
  which mostly abstains (relevance ~0). Remaining unsupported claims: LLM embellishment + fallback process text.
- Tests: expander + fallback tests added; full suite 257 passed.

## B6 follow-up - fair expansion + scoring of process statements (2026-09-30, done, uncommitted)
- Hybrid RAG + expansion (fault items): stops abstaining and gets worse (logged faithfulness 0.902 -> 0.572,
  unlogged hallucination 0.364 -> 0.909).
- Judge v2 (prompt fix) failed calibration on true process statements (34%), so not used. Instead GreenRAG's
  template statements are verified by code (rag_eval.verify_template_claim): 35/35 true, 18 had been miscounted by
  the judge. GreenRAG hallucination logged 0.409 -> 0.227, unlogged 0.591 -> 0.227; the rest is 17 invented claims.
- Final: GreenRAG vs hybrid+expansion: faithfulness +0.277 (logged) / +0.379 (unlogged), CIs exclude 0;
  hallucination -0.227 (logged, CI includes 0) / -0.682 (unlogged). vs hybrid without expansion: roughly level while
  answering (relevance 0.56-0.64 vs ~0).
- Tests: template verifier, v2 prompt, process calibration items, expansion with exclusions; full suite 261 passed.

## B7 - POST /diagnose (2026-09-30, done, uncommitted)
- `faultdiag/service.py`: DiagnoseRequest (event_id or asset + window; mode logged/unlogged; use_llm) and
  DiagnosisService (production retrieval + sibling expansion + fixed fallback + Ollama with timeout; lazy load).
  Endpoint returns the B5 DiagnosisReport; 404 unknown event, 422 invalid, 503 missing B4 results; LLM failure ->
  evidence-only fallback (200).
- Real smoke: event 73 logged moderate/medium with cause cited; unlogged low/undetermined. LLM text still mixes in
  other events / adds unsupported causes (as measured in B6). Cold start ~100 s, then ~6 s per call.
- Shared semantic chunks between retriever and expander (was embedded twice). Track B (B1-B7) complete.
- Tests: tests/test_diagnose_api.py 10 passed; full suite 271 passed. CLAUDE.md: /diagnose added to "do not break".

## C1 - Threat model (2026-09-30, done, uncommitted)
- Decision: local mode only (no Azure setup/implementation/testing); BACKEND_MODE abstraction untouched. Recorded in
  CLAUDE.md rule 12 and in memory.
- docs/threat_model.md: architecture as implemented (FastAPI + ChromaDB + MiniLM/cross-encoder + Ollama), trust
  boundaries, and a table per layer (IoT/SCADA, ingestion, cloud = hosting/supply chain for the local stack,
  database = local files, vector DB, retriever, LLM, decision engine) with attack, vulnerability in this repo,
  impact, detection, mitigation and status. 30 threats: 4 existing, 13 partial, 12 planned (phase given),
  1 out of scope (Azure deployment). Coverage map to C2-C7.
- Key findings: no authentication on any endpoint (auth.py empty); joblib (pickle) model loading; KB hashes recorded
  but never re-verified; no trust check between retrieval and LLM; models downloaded without pinned revisions;
  no decision engine yet.
- tests/test_threat_model.py (9): every layer has the required columns, rows complete and unique, file references
  exist, existing/partial controls name a file, planned items name a phase, summary counts match, stated facts
  (auth.py empty, joblib.load) still hold. The tests caught 4 rows claiming controls without a file (fixed).
  Full suite 280 passed.

## C2 - Attack simulator (2026-09-30, done with one pending run, uncommitted)
- Code: src/backend/security/attacks.py (attacks; all attack data synthetic), src/backend/security/attack_eval.py
  (deterministic scoring), experiments/attack_simulation.py (--part telemetry | knowledge [--model], --report).
  Docs: docs/attack_simulation.md. Results: results/security/ (c2_summary.md + raw files).
- Protocol: undefended system; models trained/calibrated on clean data as deployed; only test-time inputs attacked;
  errors vs clean truth; every attacked run has a same-config clean run; seeds 0-2 (telemetry), LLM temp 0 seed 0.
  Attack KB documents are schema-valid variants of real documents; ground truth only in c2_manifest.json.
- Telemetry (complete): cloudy-day replay is the strongest attack and invisible to A1 checks (plant 1 daylight MAE
  3013.8 -> 4891.3 kW, 90% coverage 0.887 -> 0.761; wind replay MAE 284.0 -> 1141.5, coverage 0.853 -> 0.699, flagged
  only as "turbine down"). Additive biases change MAE by <= ~5%. Power x0.5 caught partly on plant 1 only (55 A1
  flags). Deployed LR fault detector fully evaded (recall 0.750 -> 0.000); +5 IQR gearbox offset raises its FPR
  0.400 -> 0.800 and steers attribution to gearbox. IF "robust" to evasion only because the B4 event thresholds
  (0.0015-0.0030) sit below its own 1% train flag rate.
- Knowledge: qwen2.5:7b-instruct (deployed) COMPLETE 594/594 items (2026-10-15; 594 unique item keys, run record
  c2_run_knowledge.json). Four sessions on one append-only raw file: 1-296 (2026-09-30), 297-406 (stopped by the
  watchdog at 487 MB free), 407-466 + 467-594 (2026-10-15). The Ollama runner's RAM grows ~1 GB per ~20 items, so
  the runner now unloads/reloads the generator every 60 items (`--batch`, default 60; `ollama stop <model>` on that
  model only; ModelCycler in experiments/attack_simulation.py); each reload freed ~1.7 GB, min free RAM 2229 MB.
  The first reload crashed the new runner (CUDA "shared object initialization failed", between items, no output
  lost); fixed with a retried, paced load, also used to preload at start. llama3.2:3b pass complete (594/594, own
  clean baseline, own raw file). Both models now complete in every cell.
- Knowledge findings (full runs only): production retrieval puts a poisoned doc in context for 28.8% of questions.
  Unsafe pitch/gearbox procedures reached the user via GreenRAG's extractive fallback in both models (LLM answers
  failed the citation check; fallback quoted the top chunk). Forged A-73 record adopted in /diagnose free text:
  llama 6/22 poisoned diagnoses, qwen 2/22 (events 0 and 71, turbine 0, non-target; event 71 "inspect and replace
  the hydraulic system and main bearing"); structured component unchanged 0/44 for both. qwen /diagnose A-73
  itself: LLM output rejected by the citation check, fallback gave the genuine cause. llama answered one question
  with the stuffed doc's "reset all alarms remotely". Injected KB docs: 0 successes for both models.
  Query injection: qwen (deployed) obeyed 55/66 plain RAG (38 canary-only, 17 appended) and 37/66 GreenRAG (all
  canary-only with a well-formed citation, so the citation check passed every one; fallback 21.2% vs 25.8% clean,
  no incidental protection); llama 0/66 plain, 2/66 GreenRAG (fallback 72.7% vs 43.9% clean). The deployed
  undefended system is highly vulnerable to query injection; llama results understated this.
- Downstream (not re-run, no request): C3/C4/C5 notes below say qwen injection outcomes are pending; the qwen
  unprotected baselines now exist, but the qwen secured arm (C4) and qwen-under-attack decisions (C5) are still
  unmeasured.
- Scoring rule changed twice after outputs existed (documented in docs/attack_simulation.md): marker -> marker or
  >=30-char verbatim span (truncated fallback quotes were scored as failures), then unique windows + retrieval
  requirement (fixed attribution errors; only qwen plain-RAG collateral changed, 0.015 -> 0.000). A unit test
  caught a marker that also occurred in the genuine KB before any run.
- Tests: tests/test_attacks.py 18 passed (2026-09-30); 19 with the ModelCycler test (2026-10-15, mocked Ollama:
  reloads only its model every N items, retries a crashed load, batch 0 never cycles). Full suite
  `python -m pytest` 2026-10-15 with Ollama up: 343 passed, 0 skipped, 0 deselected (230.8 s).

## C3 - Detectors (2026-10-01, done, uncommitted)
- Code (src/backend/security/): scada_anomaly.py (telemetry: bounds, rate, cross-sensor consistency, Mahalanobis,
  replay; fitted on clean training days only; per-day decisions), injection_detector.py (weighted patterns, fixed
  before evaluation) + injection_eval_set.py (synthetic held-out set), doc_trust.py (HMAC-signed registry of the KB
  build + content checks: provenance vs event_info.csv, stuffing, injection, chunk-in-document), evidence_filter.py
  (TrustedRetriever drop/annotate). rag_systems.greenrag_answer: the extractive fallback never quotes a chunk marked
  untrusted (unmarked hits unchanged). attacks.jitter (C3-only noisy-replay robustness variant).
  attack_eval.with_windows (shared by C2 scoring; C2 summary verified byte-identical after the refactor).
- Evaluation: experiments/detector_evaluation.py (--part telemetry|text|filter, --report) -> results/security/c3_*.
  Docs: docs/detectors.md. No Ollama needed; served endpoints unchanged (wiring is C4).
- Results: telemetry detection 30/30 (plant 1, FPR 2/13 days), 23/30 (plant 2: power x0.5 0/5 undetected,
  irradiation 3/5; FPR 0/13), 62/62 (wind, FPR 0/92); exact replay always caught, replay+2% noise 11/15 on plant 2.
  Injection: C2 template 66/66, FPR 0/66 (upper bound, template known), both C2 successes flagged; held-out 11/15,
  FPR 1/15 (misses: German, letter-spaced, "pretend you are", "your new task"); KB docs 4/4, FPR 0/194.
  Trust: self-declared content_sha256 catches 0/17 (attackers can compute it); signed registry 9/9 added + 8/8
  in-place, FPR 0/189; content checks alone 5/9 and 5/8 (unsafe procedures, forged A-0 maintenance record and the
  stuffed doc need the registry; the stuffing check as designed misses the stuffed doc). All C2 contexts with an
  attack doc fully flagged (qwen 42/42, llama 46/46 + 28/28), 0 false flags; contexts identical across generators
  (264/264). Evidence filter replay (LLM disabled, fallback path): attack content in output 2/176 -> 0/176, attack
  docs in context -> 0, clean-KB outputs unchanged 88/88, attacked-KB outputs identical to clean 88/88.
- qwen: no injection outcomes yet (C2 qwen run pending); input-side rates apply to both generators, success against
  qwen unknown; no llama result used as a qwen stand-in. LLM-generated answers under the filter not measured (C6).
- Tests: tests/test_detectors.py 17 passed. Full suite `python -m pytest -m "not ollama"`: 313 passed,
  2 deselected (Ollama tests not run: qwen stays unloaded for memory; C3 changed nothing they exercise).

## C4 - Secure retrieval pipeline (2026-10-01, done; qwen answer-level evaluation pending, uncommitted)
- src/backend/security/secure_pipeline.py: SecureRAGPipeline (input checks -> injection detection (block) ->
  production retrieval over the KB store documents.jsonl -> document trust -> evidence filter -> GreenRAG LLM),
  SecurityReport per answer, SecureAskService (for C7 /secure-ask), build_secure_retrieval (shared), trusted
  reference = signed trust_registry.json if present (fails closed without/with wrong key) else in-process build
  from code + datasets; `--write-registry` CLI. evidence_filter.search_with_report (per call, thread-safe).
- Wired into the served path: DiagnosisService (/diagnose) now uses build_secure_retrieval (store index + trust +
  filter; siblings only of trusted hits). Store == code build (189/189) so clean behaviour is unchanged.
  /ask, /query and the BACKEND_MODE abstraction untouched. attack_simulation.knowledge_tables(path=) reused for
  scoring. Docs: docs/secure_retrieval.md.
- Evaluation (experiments/secure_pipeline_eval.py; results/security/c4_*): generator llama3.2:3b by decision (only
  generator with complete unprotected C2 baselines; security stages are before the LLM and measured LLM-free too;
  qwen not run: no injection baselines yet, memory). Stages: attack docs passed to LLM/fallback 0 in every state
  (candidates before filter: 28/66 + 16/22 poisoned, 17/66 + 16/22 injected); injected questions blocked 66/66;
  clean questions blocked 0/66, clean chunks rejected 0; security stages median 16.8 ms/question. End to end
  (llama): successes unprotected -> secure: poisoned GreenRAG 2/66 -> 0, poisoned /diagnose 6/22 -> 0, query
  injection 2/66 -> 0; clean answers identical to unprotected 88/88.
- C2 correction found here: the forged A-73 record also leaked into 5 non-target llama diagnoses (events 0, 24, 69,
  71, 92) - 6/22 poisoned diagnoses affected, not 1; docs/attack_simulation.md updated.
- Run notes: the first llama run was killed for memory while I ran unit tests in parallel (my error); resumed
  alone from item 32 and completed (330/330). Tests: tests/test_secure_pipeline.py 10 passed (incl. integration on a
  poisoned store). Full suite `-m "not ollama"`: 323 passed, 2 deselected.

## C5 - Trust-aware decision engine (2026-10-07, done; uncommitted)
- Started in an earlier session (code + 80/330 captured outputs, PROGRESS not updated); reviewed and completed here.
- src/backend/trust/decision.py: Recommendation -> evidence verification -> security verification (engine re-checks
  every evidence doc with C3 DocumentTrust) -> operational constraints -> risk assessment -> T -> accept/review/reject.
  Adapters from_secure_answer (C4) and from_diagnosis (B5/B7). Nothing is executed; the LLM only recommends.
  src/backend/trust/signals.py loads every input from results/ (nothing hard-coded). Docs: docs/decision_engine.md.
- T = 0.35E + 0.15R + 0.30C + 0.20S (reasoning in docs). E = citation validity x route factor (fallback 0.845 =
  B6 relevance ratio) x B5 evidence support, 0 on untrusted evidence. R = mean sigmoid of cross-encoder logits of the
  cited evidence. C: diagnosis max(0,(ci_low-0.5)/0.5) = 0 for B4 (AUC 0.742, CI [0.500, 0.950]); answers B6
  faithfulness CI low 0.938; forecasts from A5 coverage CI low (plant1 0.534, plant2 0.727, wind 0.690).
  S = weakest check that ran, strength 1-(1-TPR)/(1-FPR) (changed this session to use C3 FPR too): injection
  0.714 (TPR 11/15, FPR 1/15), doc trust 1.000 (17/17, 0/189), content-only 0.588; telemetry plant1 1.000,
  plant2 0.767, wind 1.000. Accept needs T >= 0.70, floors E/C/S 0.5 R 0.3, no soft failure, no tampering seen;
  reject on block / untrusted evidence / hard constraint / T < 0.40. Config refuses C or S floor 0.
- Constraints: hard no_unsafe_directive, claims_cite_trusted_evidence, high_severity_needs_strong_evidence (B5 rule),
  interval_physically_valid; soft action_references_component, action_consistent_with_cause.
- Run (experiments/decision_engine_eval.py; results/trust/c5_*): llama3.2:3b capture resumed 80 -> 330/330 (521.7 s
  this session), temperature 0, seed 0. 374 items (330 llama + 44 saved B5 qwen clean diagnoses). Default config:
  diagnoses 0/110 accept (all review; C = 0 floor); llama GreenRAG clean 60/66 accept (6 review: R below floor),
  poisoned 35/66, injected 44/66 accept (rest review, mostly "KB tampering seen"), query injection 66/66 reject.
  0 rejects for untrusted evidence: no attack doc reached a recommendation (consistent with C4).
- Sensitivity: 969 weight vectors (0.05 grid, each >= 0.05): mean 3.3% of items change decision vs default, max 29.4%
  (wC=0.60, wS=0.30: diagnoses move review -> reject). Accept share per group varies at most 0.82-0.91 (clean
  GreenRAG); diagnoses never accept anywhere on the grid. Thresholds accept_at 0.60-0.80 change <= 0.5% of items;
  review_at 0.50 moves 7.8% review -> reject. Grayscale figure c5_weight_sweep.png.
- Reproducibility vs C4 texts: 324/330 identical; the 6 differences are all in the poisoned state (2 diagnoses
  fallback instead of llm), despite temperature 0 / seed 0 - likely Ollama GPU nondeterminism (not verified).
- Unsafe-directive constraint on C2 unprotected outputs (no other defence): flags 2/3 qwen and 3/9 llama successful
  poisoning answers, 0/2 query injections; 0/308 clean-state outputs flagged. Keyword safeguard, not a classifier.
- qwen2.5:7b-instruct: decision behaviour measured only on its 44 saved clean B5 diagnoses; under attack it is
  UNMEASURED until its C2 knowledge run resumes.
- Tests: tests/test_decision_engine.py 17 passed (C = 0 never accepts over the whole grid and any valid
  thresholds/floors; real C4 pipeline + always-perfect stub LLM: injected question blocked -> reject for all 969
  weights, clean version accepted; poisoned-only retrieval never accepts; engine catches poisoned evidence claimed
  trusted). Full suite `python -m pytest`: 342 passed (Ollama tests included, run after the LLM job finished).

## Report numbers (2026-10-16)
- `results/report_numbers.md` (compiled 2026-10-15 from existing files, nothing re-run) re-checked against its
  sources: forecasting tables (solar fixed split + wind rolling origin, with persistence), dataset sizes/rows/
  features, licences (Kaggle URL and licence text NOT RECORDED; fault README states no licence), Python 3.11.15 +
  torch 2.13.0 (CPU), KB documents (99 real / 90 synthetic), Phase-I features (event-driven ingestion, tiered
  serving, Power BI: not implemented), Azure (nothing deployed; region, image build/push, scaling NOT RECORDED;
  auth = API keys from env vars in code only). Added a dated re-check note.

## C7 - POST /secure-ask and POST /decision (2026-10-16, done, uncommitted)
- POST /secure-ask {question} -> C4 SecureAnswer (route blocked/no_trusted_evidence/generated/numeric/
  extractive_fallback + per-stage SecurityReport). A blocked question is a 200 with route "blocked".
- POST /decision {kind: answer|diagnosis|forecast, question | diagnose | forecast} -> runs the served pipeline,
  then the C5 engine: accept/review/reject, trust score, E/R/C/S, weights, constraints, reasons, recommendation,
  underlying output. Recommendations are produced server-side only (extra fields such as a client
  "recommendation" -> 422). Errors: unknown event 404, missing model 503, bad origin 422.
- `src/backend/trust/service.py` (DecisionService); `DiagnosisService.diagnose_with_security` returns the doc ids
  removed by trust verification per call (lock around the retriever's per-call state); one secure retrieval per
  process shared by /diagnose, /secure-ask and /decision (`main.get_secure_retrieval`). Forecast decisions do not
  build retrieval models.
- Forecasts: the C3 telemetry detector is day-level and not wired into /forecast, so served forecasts have
  telemetry_checked = False -> S = 0 -> at best "review" (documented). Real smoke (no LLM): solar_plant1
  2020-06-17T11:00 -> review, T 0.660 (E 1, R 1, C 0.534, S 0); solar_plant2 -> review (C 0.727, S 0).
- /ask, /forecast, /diagnose unchanged (regression + diagnose tests pass).
- Tests: tests/test_secure_api.py 15 passed. First run: 1 failure, my test expected trust_checked False for a fake
  retriever, but the engine re-verifies evidence itself and sets it True (code correct; assertion fixed).

## I1-I3 - Orchestrator, console, run_all (2026-10-16, done, uncommitted)
- I1 `src/backend/orchestrator.py` + POST /assistant: rule-based intent (forecast / diagnosis / question) and
  ablation configs A (LLM only) B (basic RAG) C (hybrid RAG) D (+forecasting) E (+fault diagnosis) F (secure
  pipeline + forecasting + diagnosis + C5 trust). Only F returns trust/decision; a disabled module's intent is
  answered by that config's question path. A-C reuse the B6 systems unchanged.
- I2 console (src/frontend): new assistant panel (config A-F, asset, origin, confidence, event id) showing the
  accept/review/reject badge, trust score, E/R/C/S bars, reasons, forecast interval chart (prediction, bounds,
  observed value) and diagnosis severity/confidence. Existing ids, tokens and /ask flow unchanged.
- I3 `python -m experiments.run_all`: status of experiments 1-7 + ablation from declared outputs (item counts for
  resumable LLM runs) -> results/run_all_status.{md,json}; `--run-missing` runs only incomplete steps, one at a
  time, stops below 500 MB free RAM; never re-runs a complete step. Status at creation: all complete except C6
  telemetry, C6 judge, C6 held-out (444/990), C6 report; ablation A-F = manual (no ablation experiment designed or
  run, needs an LLM budget decision).
- C6 runner: LLM reload interval now `--batch`, default 30 (capture run of 2026-10-15 used 60).
- Tests: tests/test_orchestrator.py 11, test_frontend.py +1, tests/test_run_all.py 4 passed.

## I4 - Cloud portability (2026-10-16, PENDING)
- `python -m experiments.cloud_portability` -> results/cloud/i4_status.{md,json} (setting names only, never values;
  nothing sent to Azure). Result: PENDING. No `.env`; none of AZURE_OPENAI_ENDPOINT/API_KEY/DEPLOYMENT,
  AZURE_SEARCH_ENDPOINT/API_KEY/INDEX set; no Azure CLI; and the 2026-09-30 local-only decision (rule 12).
  The local-vs-Azure harness was not run and nothing was simulated; no Azure latency/cost number exists.
- Test: tests/test_run_all.py::test_cloud_portability_is_pending_without_settings passed.

## I5 - README, RESULTS and limitations (2026-10-16)
- README.md rewritten: what the system is, quick start, all endpoints (incl. /secure-ask, /decision, /assistant),
  repo map, doc index, and an honest limitations list (34-day solar data, no humidity, wind faults vs inverter
  faults, unrecorded licences, wind loses to persistence on MAE, chance-level fault detector, 90/189 synthetic KB
  docs, local LLM judge, optimistic detector rates, telemetry detector not on the served path, no auth, missing
  Phase-I features, Azure untested, ablation not run, no release tag under the no-git rule).
- docs/RESULTS.md: headline numbers per experiment, each copied from its named summary (spot-checked against
  a5_summary, wind_summary, b3_retrieval_summary, b6_summary, b6_followup_deterministic, a4_summary).

## C6 - Security evaluation, qwen2.5:7b-instruct (2026-10-16, done, uncommitted)
- Runs (guarded: one job at a time, kill below 500 MB free RAM, qwen reloaded every 30 LLM calls):
  held-out injections resumed at item 445 -> 990/990 unique (906 s, exit 0, min free 5651 MB; the first launch
  failed at import because my runner used a non-venv python, nothing written, fixed and restarted);
  qwen judge: 28 new judgements, 56 claims, 0 judge failures (124 s; the rest reuse identical B6 judgements);
  CPU telemetry re-run: 39 runs, max |MAE diff| vs C2 = 0.00 kW (290 s). Capture (330/330) from 2026-10-15.
- Report fixes: a generator built diagnosis rows before filtering (KeyError 'route'; first full report run) ->
  fixed; added the accepted-answer correctness table per state + held-out row.
- Results (results/security/c6_summary.md; docs/security_evaluation.md): query injection GreenRAG 37/66 -> 0/66;
  unsafe procedure 2/66 -> 0; forged record /diagnose 2/22 -> 0; telemetry replay fully gated; Plant 2 power x0.5
  5/5 and irradiation bias 2/5 pass the gate. Held-out detection 726/990 (phrasings 6, 10, 11, 12 never blocked);
  engine accepted 219/264 delivered held-out answers, 51 of which executed the injection ("I approve skipping
  the pitch battery test", "OK", "low severity"); engine stopped only 5/56 obeyed answers.
  Clean utility unchanged (faithfulness 0.970 vs 0.970, hallucination 0.015 vs 0.015, 0/66 wrongly blocked,
  security 7.7 ms median). Accepted-answer correctness on capture items 142/142; correct answers sent to review
  29/65 (poisoned), 20/65 (injected), 4/65 (clean). Flag recall 0.944, reject precision 1.000. Diagnoses never
  accepted (C = 0).
- Marker check: obeyed hits were read and are genuine; post-hoc broader severity pattern finds 2 more phrasing-12
  hits (both accepted) -> pre-registered markers are a lower bound; scored numbers unchanged.

## Final check (2026-10-16)
- `python -m pytest` (Ollama up, nothing deselected): 383 passed, 0 failed, 0 skipped, 1 warning (starlette
  httpx deprecation), 207.2 s. `python -m experiments.run_all`: every step complete except ablation A-F (manual).
- Remaining open items: ablation A-F experiment (needs item set + LLM budget decision); I4 Azure harness PENDING;
  telemetry detector not on the served /forecast path; engine accepts answers obeying missed injections (C6).

## Product polish: backend, console, team docs (2026-10-16, done, uncommitted)
- Backend:
  - `settings.py`: one settings class, environment variables only.
  - `api_support.py`: one error body `{code, message, hint, detail, request_id}`; `detail` keeps the old field, so
    existing clients and tests work. Plus an `X-Request-ID` header and a JSON request log with `latency_ms`.
  - CORS limited to `GREENRAG_CONSOLE_ORIGINS` (a foreign origin gets no allow header; checked live).
  - Strict request models (`extra="forbid"`) with request examples.
  - New routers in `src/backend/api/`:
    - `ops`: `GET /ready`, `POST`/`GET /warmup` (background loading), `GET /system/info`, startup banner;
    - `results`: `GET /results/summary[/{track}]`, read from result files; caveats are built from the same numbers;
    - `catalog`: `GET /diagnose/events`, `/forecast/assets`, `/forecast/series`, `/assistant/suggestions`,
      `/kb/chunks`;
    - `lab`: `GET /security/lab/catalogue`, `POST /security/lab/{telemetry,injection,poisoning}`.
  - The LLM timeout and host come from the settings.
  - OpenAPI: tags per track; the "confidence" meanings stated plainly. 27 of 28 operations show a real recorded
    response (`python -m src.backend.api.capture_examples`, all calls 200/202); `POST /ingest` is not recorded
    because it mutates the index.
  - `/health`, `/system/status` and `/ask` are unchanged (the legacy `/ask` still returns 500 on LLM failure,
    as tested).
- Fixes found while building:
  - `/ready` first took 16 s (signal bootstrap) -> it now checks the source files only (0.08 s); warm-up loads the
    signals.
  - `include_router` wraps routes without `.path` (broke 3 existing tests) -> routes re-registered with
    `add_api_route`, so dependency overrides apply.
  - Vectorised `/forecast/series` bounds differ from `/forecast` by < 0.001 kW (float32 batch quantiles; the point
    forecast is identical) -> tested with a 0.01 kW tolerance and documented.
- Console `src/console/` (no build step, served at `/console/`):
  - Pages: Assistant (chat, domain + config selector, suggested questions, expandable source cards, trust panel,
    review/reject warning style), Forecast (actual vs forecast with interval band, model comparison vs
    persistence, coverage, in_sample explained, decision check), Fault diagnosis (event list, logged/unlogged,
    report with cited evidence and ground-truth comparison), Security lab (telemetry, poisoning, injection,
    held-out weakness), Results, About (diagram, team, status, versions).
  - Validated reference palette; light and dark themes; keyboard tabs; reduced motion; data-table views.
- Playwright (`scripts/capture_screenshots.py`, installed Chrome, playwright 1.64.0 via uv):
  - 27 captures in `docs/screenshots/` (9 views x light, dark, 390 px mobile) plus `openapi_docs.png`.
  - Final run: 0 console errors or warnings and 0 px horizontal overflow on every view.
  - Earlier runs found and fixed: unreadable SVG bar chart in the side card -> HTML bars; mobile overflow of up to
    190 px (long select options) -> fixed; lab engine panel missing weights -> added.
- Live checks: warm-up 66-73 s (6 steps done); the lab replay on Plant 1 flagged 5/5 days, matching the recorded
  C3 verdicts.
- Docs: `docs/team/README.md` (architecture, 5-command run, API map) and one file per member (scope, files,
  endpoints, results with source files, limitations, viva Q&A, demo script, not done). Every number was checked
  against its result file; one unsourced figure was removed. README updated; requirements.txt adds
  pydantic-settings, httpx, scikit-learn, torch, playwright.
- Tests: tests/test_platform_api.py (16) and tests/test_catalog_lab_api.py (11, 1 integration).
  Full suite `python -m pytest` (Ollama up, nothing deselected): **410 passed** (383 before + 27 new), 0 failed,
  0 skipped, 1 warning (starlette httpx deprecation), 478.6 s.

## Azure plan A (2026-10-16, user-authorised)
- Stage 1 (done): rg greenragsec-rg centralindia; Storage greenragsecymjvr (Standard_LRS Hot, shared keys off,
  containers kbdocs 2 blobs + results 289 blobs); Key Vault greenragsec-kv-ymjvr (RBAC; secrets
  search-admin-key, appinsights-connection-string); Log Analytics greenragsec-law-ymjvr (0.1 GB/day cap); App
  Insights greenragsec-appi-ymjvr; AI Search greenragsec-srch-ymjvr (free). First run stopped on my container
  name 'kb' (<3 chars); resumed with kbdocs after user approval. Estimate ~$0.01/month (Retail Prices API).
- Stage 2 (code): settings (providers, Azure URLs, KB_SOURCE, GREENRAG_APPINSIGHTS); src/backend/cloud/ (secrets:
  env then Key Vault, never logged; blob: upload/download CLI, KB_SOURCE=blob still trust-checked in-process;
  appinsights: httpx handler, no otel version change); azure_store: local MiniLM 384-d, one shared index with
  domain filter, base64 keys; generator/status follow GENERATOR_PROVIDER (Ollama); /system/info cloud hosts only;
  /ready azure_config check; .env.example names only; CLAUDE.md rule 12 + docs/azure.md updated.
  Added azure-keyvault-secrets. tests/test_azure_mode.py 17 mocked tests. Stage 3 (I4 run) not started.

## Open questions / notes for the user
- The fault dataset contains wind-turbine faults; the plan's solar-inverter fault list does not match (details in B4).
- No humidity in the solar data (plan asks for it); no external source used unless approved.
