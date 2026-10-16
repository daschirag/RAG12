# Track B - Fault diagnosis (B4)

Code: `src/backend/faultdiag/data.py` (loading, leakage safeguards, normalisation), `src/backend/faultdiag/models.py`
(models, leave-one-asset-out, metrics, component attribution). Run: `python -m experiments.fault_diagnosis` ->
`results/faultdiag/b4_summary.md`, `b4_event_metrics.csv`, `b4_row_metrics.csv`, `b4_event_predictions.csv`,
`b4_attribution.csv`. Tests: `tests/test_faultdiag.py`.

## Data and a leakage finding
Scope: **Wind Farm A only** (22 event datasets, 5 turbines, 12 anomaly / 10 normal events), following the rule to
develop on Farm A. Farms B and C have different sensor sets (257 and 957 columns) and would need their own models;
they are not used yet.

Each dataset is one turbine's 10-minute SCADA history: a `train` part (about 52,000 rows, one year) followed by a
`prediction` part that contains the event window from event_info.csv.

**`status_type_id` leaks the label in Farm A.** Inside every anomaly window it is 0% "normal operation" (almost all
4 = downtime), except event 40 (56%); inside every normal window it is 100% normal. At the same time rotor speed and
grid power in the anomaly windows look like normal operation (e.g. event 73: rotor 9.6 rpm, power 0.4 vs 9.5 rpm,
0.3 in the train part). So the status code follows the fault logbook, not the machine state. It is excluded
everywhere: not a feature and not a row filter (filtering to normal-status rows would remove almost every anomaly
row).

Other data notes: columns `sensor_44`-`sensor_51` have no statistic suffix and are not monotonic (not counters);
`sensor_46` and `sensor_49` are constant zero. "Total active power" (sensor_50) is not the grid power signal
(`power_30_avg`, "Grid power").

## Protocol (fixed before training)
- Features: all sensor columns; angle averages (wind directions, pitch, nacelle direction) as sin/cos; each value
  as a robust z-score (median / IQR) against the same turbine's train part (label-free, precedes the window;
  constant columns scale 1). 85 features.
- Rows: every 3rd window row (30 min) for learning/scoring; label 1 in anomaly windows, 0 in normal windows.
- Supervised: logistic regression (C 0.1), random forest (300 trees, min leaf 5), XGBoost (300 trees, depth 4);
  **leave-one-asset-out** (5 folds): a turbine is never in train and test at once. Probability threshold 0.5.
- Unsupervised: Isolation Forest per dataset, fitted on its own train part (every 6th row); a row is flagged above
  the 99th percentile of its training scores.
- Event level (primary): event probability = mean row probability (supervised) or fraction of flagged rows
  (Isolation Forest); anomalous if >= 0.5. 95% CI by resampling events.
- **Class imbalance (component diagnosis)**: Farm A anomalies are hydraulic 6, gearbox 3, generator bearing 2,
  transformer 1. No supervised component classifier is trained (too few examples per class) and classes are not
  collapsed. Instead, component attribution is unsupervised: components are ranked by the mean absolute robust z
  of their sensor group (B1 sensor mapping) inside the window. Results are reported per event and per class, with
  the statement that counts of 1-6 per class are not a trustworthy accuracy estimate.
- Fault probability for B5: the supervised model with the best mean leave-one-asset-out event AUC (selected on
  these cross-validation results).

## Results (seeds 0-2; event level, 22 events)

| model | accuracy | precision | recall | F1 | AUC [95% CI, seed 0] | row-level AUC |
|---|---|---|---|---|---|---|
| logistic regression | 0.682 | 0.692 | 0.750 | 0.720 | 0.742 [0.500, 0.950] | 0.503 |
| random forest | 0.545 | 0.625 | 0.417 | 0.500 | 0.594 [0.333, 0.833] | 0.530 |
| XGBoost | 0.455 | 0.500 | 0.417 | 0.455 | 0.558 [0.299, 0.804] | 0.474 |
| Isolation Forest | 0.455 | 0.000 | 0.000 | 0.000 | 0.757 [0.545, 0.953] | 0.616 |

- **The supervised models do not generalise across turbines.** Row-level AUC is at chance (0.47-0.53). The best
  event-level result (logistic regression, AUC 0.742) has a 95% CI that reaches 0.50, i.e. it is not
  distinguishable from chance with 22 events.
- **Isolation Forest ranks events best (AUC 0.757, CI above 0.5) but its fixed operating point fails**: with the
  99th-percentile row threshold, 0-7% of window rows are flagged, so every event is called normal (accuracy
  0.455 = share of normal events, F1 0). The thresholds were fixed before training and are not re-tuned on these
  results; choosing an operating point would need data not used for evaluation (e.g. Farm B/C or a held-out asset
  set).
- **Component attribution fails**: top-1 correct for 0 of 12 anomaly events, top-2 for 1 of 12 (chance top-1 about
  0.10). Hydraulic events rank the hydraulic group 3rd-9th of 10. Farm A has only one hydraulic sensor (oil
  temperature in the hydraulic group) and the faults do not show up as larger deviations in the matching sensor
  group within these windows. With 1-6 events per class these counts are not a trustworthy accuracy estimate, but
  there is no sign that the method works.
- Possible reasons (not tested): the anomaly windows are defined from the logbook and may cover periods when the
  sensors do not yet deviate; five turbines are too few for cross-turbine supervised learning; 10-minute averages
  hide short transients.

## Isolation Forest operating threshold (follow-up)
A fixed train/validation/test split of 22 events would leave about 7 events per part (1-3 per class for some
parts), too few to be honest. It is also unnecessary: Isolation Forest uses no labels for fitting (each model is
fitted on its own turbine's unlabelled train part). Only the event threshold needs labels, so it is chosen by
**nested leave-one-asset-out**: for each held-out turbine, the threshold that maximises F1 on the other four
turbines' events (ties -> higher threshold, i.e. fewer alarms; rule fixed before running) is applied unchanged to
the held-out turbine's events. Run: `python -m experiments.fault_threshold` -> `results/faultdiag/b4_if_threshold.md`.

| seed | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| 0 | 0.773 | 0.818 | 0.750 | 0.783 |
| 1 | 0.773 | 0.818 | 0.750 | 0.783 |
| 2 | 0.500 | 0.538 | 0.583 | 0.560 |
| mean | 0.682 | 0.725 | 0.694 | 0.709 |

- Better than the fixed a-priori rule (F1 0), but **unstable**: seed 2 drops to F1 0.560.
- The selected thresholds are tiny (0.0015-0.0030 of window rows flagged, and effectively 0 in two seed-2 folds):
  the rule becomes "flag the event if one or two window rows exceed the turbine's 99th-percentile training
  score". This operating point depends on a handful of rows and on the Isolation Forest seed, so it should not be
  relied on. With 22 events, even the seed-0/1 result has wide uncertainty.
- B5 keeps the protocol's detector choice (logistic regression, best mean event AUC); this follow-up does not
  change it.

## Future work (not done, by decision)
- **Farms B and C.** Not used, by decision against the current deadline. Their sensor sets differ from Farm A
  (257 and 957 columns vs 86) and from each other, so each farm needs its own feature set and models; the
  protocol above (turbine-own robust z, leave-one-asset-out, event-level metrics, no status_type) would carry over
  unchanged. Farm C also has the most detailed logbook text (27 anomaly events) and 22 turbines, which would give a
  fairer cross-turbine test than Farm A's 5.

## What B5 can use
- The fault probability selected by the protocol is the logistic-regression event probability. Its measured
  reliability (event AUC 0.742, CI [0.50, 0.95]) must be reported alongside any diagnosis; it is not a reliable
  detector.
- Component attribution is not reliable (0/12). B5's "Likely Cause" must therefore come from retrieved evidence
  (the logged event description, maintenance records, troubleshooting procedures) and be labelled as such, not from
  the B4 attribution.

## B5 Fault diagnosis connected to RAG (`src/backend/faultdiag/diagnosis.py`)

Run: `python -m experiments.fault_rag` (needs Ollama) -> `results/faultdiag/b5_reports.jsonl` (every validated
report), `b5_summary.md`, `b5_results.csv`. Tests: `tests/test_fault_rag.py`.

Pipeline: fault signal (Farm A turbine, event window, out-of-fold logistic-regression event score from B4) ->
query -> **production retrieval** (hybrid BM25 + MiniLM, RRF, cross-encoder rerank; top 20 reranked, top 5 kept) ->
local LLM (qwen2.5:7b-instruct, JSON mode, temperature 0, seed 0) writes the report from the evidence only ->
`DiagnosisReport` (pydantic).

### Report schema and rules (fixed before running)
- **Diagnosis**: text from the LLM (or the fallback).
- **Confidence**: NOT a probability of correctness. Holds the anomaly score, the detector's **measured reliability
  read from the B4 results** (event AUC 0.742, 95% CI [0.500, 0.950], 22 events, leave-one-asset-out) and the
  evidence support: *strong* = a retrieved real-data record about this incident (same farm and turbine, event
  window within +/- 7 days, positive rerank score), *weak* = evidence but nothing about this incident, *none*.
  Level: *low* if support is weak/none; *high* only if support is strong AND the detector CI lower bound > 0.5 AND
  score >= 0.5; otherwise *moderate*. With the current detector (CI lower bound 0.500) no report can be *high*.
  The schema rejects a report whose level is not *low* when support is weak or none.
- **Likely cause**: from retrieved evidence only, citing chunk ids that must be in the evidence (schema-enforced).
  B4's component attribution (chance level) is not used.
- **Evidence**: the retrieved chunks with ids, document type, synthetic flag, rerank score, excerpt.
- **Recommended action**: from evidence, cited.
- **Severity**: deterministic. The B1 default severity of the evidence-derived component, **only when support is
  strong**; otherwise *undetermined*, with the basis naming the closest component and what it would have implied.
- If the LLM output is invalid (bad JSON, missing fields, uncited cause, citation of a document that was not
  retrieved) or the LLM is unavailable, a deterministic evidence-only fallback is used and noted in the report.

Two modes: **logged** (the event's own logbook-derived fault description may be retrieved; for a historical event
this makes the cause correct almost by construction) and **unlogged** (the event's own fault and maintenance
documents are excluded, simulating a new anomaly with no log yet). Unlogged is the honest test.

### Results (final run; 22 events x 2 modes = 44 reports, all schema-valid)

| mode | label | events | LLM / fallback | support strong/weak/none | confidence low/moderate/high | evidence component correct | severity high/medium/low/undetermined |
|---|---|---|---|---|---|---|---|
| logged | anomaly | 12 | 7 / 5 | 12/0/0 | 0/12/0 | 12/12 | 6/6/0/0 |
| logged | normal | 10 | 7 / 3 | 0/10/0 | 10/0/0 | n/a | 0/0/0/10 |
| unlogged | anomaly | 12 | 7 / 5 | 2/10/0 | 10/2/0 | 4/12 | 2/0/0/10 |
| unlogged | normal | 10 | 7 / 3 | 0/10/0 | 10/0/0 | n/a | 0/0/0/10 |

- **Logged mode**: retrieval finds the event's own fault description for all 12 anomaly events, so the
  evidence-derived component is 12/12 correct. This measures retrieval of known records, not diagnosis.
  Normal windows get weak support, low confidence and undetermined severity: the system does not invent a fault.
- **Unlogged mode (the honest test)**: 4/12 correct, **below the majority-class baseline of 6/12** (always
  "hydraulic"). Two of the four (events 51, 72) are gearbox events on turbine 21 three days apart, so each
  retrieves the other's overlapping record (support strong); two (45, 84) are hydraulic events on turbine 13 that
  retrieve the turbine's other hydraulic event (support weak). Without a log entry, the knowledge base does not
  identify the cause of a new anomaly; the reports correctly say low confidence and undetermined severity for 10 of 12.
- **LLM quality is weak even when valid**: accepted reports are often generic ("recurring anomalies", "similar
  underlying issues"), and some contain claims no evidence supports (e.g. event 73 unlogged: "environmental
  factors or operational issues"). Citation validity is enforced by the schema; faithfulness of the text is
  measured in B6.
- Time: 5.3 s per report (retrieval + LLM) on this machine.

### Changes made after seeing results (all runs kept)
1. Run 1 (`b5_run1_*`): the LLM output was rejected in 32/44 reports. 44 of the invalid citations were real
   evidence documents with the `::k` chunk suffix dropped (none were invented). The parser now accepts a bare
   document id **only if that document was retrieved**, resolving it to its evidence chunks; ids of non-retrieved
   documents are still rejected (tested). Run 2 (`b5_run2_*`): 28/44 LLM reports accepted; remaining rejections:
   13 causes without any citation, 3 citations of non-retrieved documents.
2. Run 2 showed severity "high" on low-confidence reports (e.g. event 73 unlogged took the severity of an
   unrelated generator-bearing record). Severity is now only asserted with strong incident evidence (above).
   Run 3 (final, `b5_*`) is identical to run 2 except for severity.
These were fixes to parsing and to presenting weak evidence, not tuning against labels; the component and support
numbers did not change between runs.

## B7 POST /diagnose

Code: `src/backend/faultdiag/service.py` (DiagnoseRequest, DiagnosisService), endpoint in `src/backend/main.py`.
Tests: `tests/test_diagnose_api.py` (TestClient with an injected service; one real-model test without the LLM).

Request (Wind Farm A only; other farms -> 422):
```json
{"event_id": 73}
{"event_id": 73, "mode": "unlogged"}
{"asset_id": 0, "window_start": "2023-06-10T11:40", "window_end": "2023-06-17T11:40", "use_llm": false}
```
- `event_id`: a known event; the report uses the B4 detector's out-of-fold score. A window request has no detector
  score (`anomaly_score` null; the confidence explanation says n/a).
- `mode: "unlogged"` hides the event's own fault and maintenance documents (requires event_id).
- `use_llm: false` returns the deterministic evidence-only report.

Response: the B5 `DiagnosisReport` (pydantic), unchanged. OpenAPI documents that `confidence` is measured
reliability plus evidence support, not a probability of correctness, and that `likely_cause` comes from retrieved
evidence with cited chunk ids.

Production configuration: hybrid BM25 + MiniLM, RRF, cross-encoder rerank over semantic chunks, **with sibling-chunk
expansion** and the fixed fallback (B6 run 2), local LLM via Ollama (JSON mode, 120 s timeout, one retry). If the
LLM fails or returns invalid output, the evidence-only fallback is returned (200) and noted in `notes`.
Errors: unknown event 404; invalid request 422; missing B4 results 503 (with the command to create them).

Real calls (TestClient, default service): event 73 logged -> confidence moderate/strong, severity medium, cause
cites fault-a-73 chunks; unlogged -> confidence low/weak, severity undetermined. The LLM text still shows the B6
weaknesses: in logged mode it mixed in another event of the same turbine (event 0, generator bearing), and in
unlogged mode it asserted an unsupported cause ("wear and tear"), labelled low confidence.

Latency: the first call loads the models and embeds the knowledge base (about 100 s on this CPU, measured by the
real-service test); later calls take about 6 s. The semantic chunks are built once and shared by the retriever
and the sibling expander.
