# C5 - Trust-aware decision engine

Code: `src/backend/trust/decision.py` (engine), `src/backend/trust/signals.py` (measured inputs).
Evaluation: `python -m experiments.decision_engine_eval [--llm llama3.2:3b] [--report]` -> `results/trust/`.
Tests: `tests/test_decision_engine.py`.

## Principle

The LLM (or the evidence-only fallback, or a forecaster) produces a **recommendation**. It is never executed.
The engine turns it into a **decision**: `accept` (may be acted on without review), `review` (a person must
check it first) or `reject` (do not act on it). Nothing in the engine performs an operational action.

```
LLM recommendation -> evidence verification -> security verification -> operational constraint check
                   -> risk assessment -> trust score -> final decision (accept / review / reject)
```

Adapters turn real pipeline outputs into a `Recommendation`: `from_secure_answer` (C4 `SecureAnswer`,
knowledge questions) and `from_diagnosis` (B5/B7 `DiagnosisReport`). Forecasts are built directly
(`kind="forecast"`, interval, asset, capacity).

## Stages

1. **Evidence verification**: are the cited chunk ids among the evidence actually given to the model, and
   did the recommendation come from the LLM, a computed (numeric) route or the extractive fallback?
2. **Security verification** (`DecisionEngine.verify_security`): every evidence document is re-checked
   against the trusted registry with the C3 `DocumentTrust`, whatever the upstream pipeline reported
   (defence in depth). The engine also reads which checks ran on this request (injection detection, document
   trust, telemetry check) and whether the request was blocked.
3. **Operational constraints** (`check_constraints`), see below.
4. **Risk assessment + trust score** (`decide_core`), see below.

## Trust score

`T = wE*E + wR*R + wC*C + wS*S`, every component in [0, 1], weights non-negative and summing to 1
(`TrustWeights` validates this). All inputs are measured values loaded from `results/` (`signals.py`); no
constant is assumed good.

| component | how it is computed | source of the measured inputs |
|---|---|---|
| E evidence quality | citation validity (share of cited chunks that were in the trusted evidence; 0 with no citation; 1 for computed numeric answers) x route factor (1 LLM/computed; extractive fallback = B6 answer relevance of fallback / generated) x evidence support for diagnoses (strong 1, weak 0.5, none 0). 0 if any evidence used failed trust verification or nothing passed the C4 evidence filter. | B5 evidence support, B6 `b6_per_answer.csv`, C4 evidence filter |
| R retrieval relevance | mean sigmoid of the production cross-encoder (ms-marco, a binary relevance classifier) logit over the evidence the recommendation cites (else all evidence given). 1 for computed outputs (no retrieval). | B3 production reranker scores carried in the C4 / B5 outputs |
| C confidence of the underlying model | diagnosis: B4 detector event AUC at the **lower** end of its 95% CI, rescaled so chance gives 0: `max(0, (ci_low - 0.5) / 0.5)`. B4: AUC 0.742, CI [0.500, 0.950] -> **C = 0**. Knowledge answers: lower end of a seeded bootstrap 95% CI of GreenRAG's B6 faithfulness. Forecasts: `1 - shortfall / miss budget` with shortfall = nominal 0.9 - lower end of the A5 rolling-origin coverage CI of the deployed CQR-Mondrian interval. | `b4_event_predictions.csv`, `b6_per_answer.csv`, `a5_coverage_ci.csv`, `wind/coverage_ci.csv` |
| S security confidence | the weakest check that ran on this request, at its strength `1 - (1 - TPR) / (1 - FPR)` (one minus the negative likelihood ratio: how much a "pass" lowers the odds of an attack). Question present: injection detector (held-out phrasings, not the C2 template the author knew). Evidence present: document trust (signed/in-process registry, or content checks only). Forecast: telemetry detector of that asset. A check that should have run but did not counts 0; a blocked request or untrusted evidence gives 0. | `c3_injection.csv`, `c3_trust_docs.csv`, `c3_telemetry_days.csv` |

### Default weights: E 0.35, C 0.30, S 0.20, R 0.15

- **E 0.35 (largest)**: whether the recommendation is supported by trusted, correctly cited evidence is the
  most direct check on the LLM's own output; unsupported or hallucinated content is the main failure B6
  measured.
- **C 0.30**: the recommendation is only as good as the model it rests on (the fault detector, the
  forecaster, the RAG answerer). It is weighted almost as high as E because a well-cited explanation of an
  unreliable detection is still unreliable.
- **S 0.20**: security failures are already handled by hard gates (blocked request, untrusted evidence ->
  reject). In the score S only carries the residual risk of checks that pass but are imperfect, so it gets
  less weight than E and C.
- **R 0.15 (smallest)**: relevance is partly already reflected in E (the cited evidence has to exist and be
  trusted), and the cross-encoder sigmoid is a ranking score that was not calibrated on this KB.

These are design choices, not fitted values (there is no labelled "correct decision" set to fit them on). The
sensitivity analysis below shows how much the decisions depend on them.

## Operational constraints

| constraint | hard? | applies to |
|---|---|---|
| `no_unsafe_directive`: no instruction to bypass/bridge a safety chain or protection, disable a test/alarm/brake, ignore alarms or temperatures, defer maintenance, reset alarms remotely, or declare an inspection unnecessary (a preceding negation, e.g. "never bridge the safety chain", is allowed) | hard | all |
| `claims_cite_trusted_evidence`: the recommendation cites chunk ids, all of them in the trusted evidence given (computed numeric answers exempt) | hard | answers, diagnoses |
| `high_severity_needs_strong_evidence`: severity "high" requires strong incident evidence and confidence at least "moderate" (the B5 severity rule) | hard | diagnoses |
| `action_references_component`: the recommended action names the component of the likely cause | soft | diagnoses |
| `action_consistent_with_cause`: the action does not target a different component instead | soft | diagnoses |
| `interval_physically_valid`: `0 <= lower <= prediction <= upper <= capacity` | hard | forecasts |

The unsafe-directive patterns were checked against the C2 attack procedures (all flagged) and the genuine KB
(no document flagged). They are a keyword safeguard, not a semantic safety classifier.

## Risk assessment (decision rules)

- **reject** if the request was blocked (input check or injection detection), any evidence used failed trust
  verification, any hard constraint failed, or `T < review_at` (default 0.40).
- **accept** only if all hold: `T >= accept_at` (default 0.70); every component at least its floor
  (E 0.5, R 0.3, C 0.5, S 0.5); no soft constraint failed; and no KB tampering was seen in this request
  (retrieved documents removed by trust verification -> review, because the store is under attack).
- **review** otherwise.

`DecisionConfig` validates `0 <= review_at < accept_at <= 1`, requires floors for all four components, and
refuses a C or S floor of 0. Consequence: a recommendation resting on a model not shown to beat chance (C = 0),
or on a request where a security check did not run or failed (S = 0), can never be accepted under any valid
configuration, whatever its evidence. For B4's detector this means **every fault diagnosis is at best
"review"** until a detector with a CI lower bound clearly above 0.5 is available.

## Evaluation and limitations

See `results/trust/c5_summary.md` (generated) for the decisions on real outputs, the sensitivity analysis on
the weights (969-vector grid with every weight >= 0.05, one-at-a-time sweep, thresholds), and the unsafe-directive
constraint applied to C2's unprotected outputs.

- Generator for new LLM outputs: llama3.2:3b (the only generator with complete C2 attacked baselines, same as
  C4). qwen2.5:7b-instruct is measured only on its saved clean B5 diagnoses; **qwen's decision-engine behaviour
  under attack is unmeasured** until its C2 knowledge run resumes.
- There is no ground truth of "correct decisions", so the evaluation reports decision distributions,
  gate behaviour and weight sensitivity, not decision accuracy.
- S uses point estimates of small-sample rates (e.g. injection held-out n = 15, document trust attacks n = 17);
  C uses CI lower bounds as required. The attack data is synthetic (C2).
