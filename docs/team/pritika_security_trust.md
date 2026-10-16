# Track C - Security and trust (Pritika Kasinathan, 23BIT0232)

## Scope

- **C1 threat model.**
- **C2 attack simulator.** Every attack is synthetic: false data injection, replay, scaling, spoofing, KB
  poisoning, and prompt injection in questions and in KB documents.
- **C3 detectors.** Telemetry, prompt injection, document trust and an evidence filter.
- **C4 secure retrieval pipeline.**
- **C5 trust-aware decision engine.**
- **C6 security evaluation** with the deployed qwen2.5:7b-instruct.
- **C7 API.**
- **Security lab** in the console.

## Files and modules

| file | role |
|---|---|
| `docs/threat_model.md` | C1: 30 threats across 8 layers (4 existing, 13 partial, 12 planned, 1 out of scope) |
| `src/backend/security/attacks.py`, `attack_eval.py`, `experiments/attack_simulation.py` | C2 attacks and scoring |
| `src/backend/security/scada_anomaly.py` | C3 telemetry detector (bounds, rate, cross-sensor consistency, Mahalanobis, replay) |
| `src/backend/security/injection_detector.py`, `injection_eval_set.py` | C3 prompt-injection detector + held-out set |
| `src/backend/security/doc_trust.py`, `evidence_filter.py` | C3 HMAC-signed registry + content checks; drop untrusted chunks |
| `src/backend/security/secure_pipeline.py` | C4: input checks -> injection detection (block) -> retrieval -> trust -> filter -> LLM |
| `src/backend/trust/decision.py`, `signals.py`, `service.py` | C5 engine (signals read from results/), C7 `/decision` |
| `src/backend/api/lab.py` | security lab endpoints |
| `experiments/security_evaluation.py`, `security_evaluation_report.py`, `docs/security_evaluation.md` | C6 |

## Endpoints

- `POST /secure-ask {question}` returns a `SecureAnswer`: the route (`blocked`, `no_trusted_evidence`, `generated`,
  `numeric` or `extractive_fallback`) and a per-stage `SecurityReport`.
- `POST /decision {kind, question | diagnose | forecast}` returns accept / review / reject with the trust score,
  its components, constraints and reasons.
  - The recommendation is always produced server-side: a client-supplied one is refused (422).
- `GET /security/lab/catalogue`; `POST /security/lab/telemetry | injection | poisoning`.
  - These run the C2 attacks against the live detectors and the engine.
  - Forecast harm and LLM outcomes are the recorded C6 values for the same attack.

## Decision engine (C5)

`T = 0.35E + 0.15R + 0.30C + 0.20S`. All four components are measured, not assumed:

| component | meaning | where it comes from |
|---|---|---|
| E | evidence quality | citation validity x route factor x evidence support; 0 if evidence failed trust |
| R | retrieval relevance | mean sigmoid of the cross-encoder logits of the cited evidence |
| C | model confidence | diagnosis: B4 AUC CI low rescaled, which gives 0 at chance; answer: B6 faithfulness CI low; forecast: A5 coverage CI shortfall |
| S | security | C3 strength of the weakest check that ran, 1 - (1 - TPR) / (1 - FPR); 0 if a check did not run |

Decision rules:
- **Reject:** blocked request, untrusted evidence, a failed hard constraint, or T < 0.40.
- **Accept:** T >= 0.70, every floor met (E, C and S >= 0.5; R >= 0.3), no soft failure, and no tampering seen.
- **Review:** everything else.

Weight sensitivity: 969 weight vectors were tested (`results/trust/c5_summary.md`).

## Results

Undefended system (C2, `results/security/c2_summary.md`), all attack data synthetic:
- **Telemetry replay** on Plant 1: daylight MAE 3013.8 -> 4891.3 kW, and 90% coverage 0.887 -> 0.761.
- **Query injection** (C2 template): qwen obeyed 55/66 in plain RAG and 37/66 in GreenRAG.

Detectors (C3, `results/security/c3_summary.md`):

| detector | measured |
|---|---|
| telemetry, Plant 1 | irradiation bias 5/5, temperature spikes 15/15, power x0.5 5/5, replay 5/5; FPR 2/13 clean days |
| telemetry, Plant 2 | **power x0.5 0/5**, irradiation bias 3/5, spikes 15/15, replay 5/5 (replay + 2% noise 11/15); FPR 0/13 |
| telemetry, wind | bias 31/31, replay 31/31; FPR 0/92 |
| injection detector | C2 template 66/66 (FPR 0/66); **held-out 11/15** (FPR 1/15); KB documents 4/4 (FPR 0/194) |
| document trust (signed registry) | 9/9 added and 8/8 in-place tampered documents flagged; 0/189 false flags |

Secure pipeline (C4, llama3.2:3b, `results/security/c4_summary.md`):
- No attack document reached the LLM.
- Successes, undefended -> secure: poisoned answers 2/66 -> 0, poisoned diagnoses 6/22 -> 0, query injection
  2/66 -> 0.
- Clean answers were identical to the undefended run (88/88). The security stages took a median 16.8 ms.

Decision engine (C5, `results/trust/c5_summary.md`): 0/110 diagnoses accepted, 60/66 clean answers accepted, and
66/66 query injections rejected.

C6 with the deployed qwen2.5:7b-instruct (`results/security/c6_summary.md`):
- **Query injection** (GreenRAG): 37/66 undefended -> 0/66 secure -> 0/66 accepted.
- **Unsafe procedure:** 2/66 -> 0 -> 0. **Forged record** (/diagnose): 2/22 -> 0 -> 0.
- **Telemetry:** replay is fully gated (0 successes in (b) and (c)). **Plant 2 power x0.5 passes 5/5** and
  irradiation bias 2/5.
- **Clean utility unchanged:**
  - faithfulness 0.970 vs 0.970, hallucination 0.015 vs 0.015;
  - 0/66 wrongly blocked;
  - security stages 7.7 ms median, engine 1.2 ms.
- **Accepted-answer correctness on the capture items: 142/142.** The over-caution cost is that correct answers on
  attacked KBs go to review: 29/65 poisoned, 20/65 injected.
- **Flagging bad items:** recall 0.944 for review or reject, and reject precision 1.000.
- **Held-out weakness:**
  - The detector blocked 726/990 held-out items; 4 of the 15 phrasings were never blocked.
  - Of 264 delivered answers the engine accepted 219, and **51 of those 219 executed the injected instruction**
    (e.g. "I approve skipping the pitch battery test").
  - The engine stopped only 5 of 56 obeyed answers.

Live check in this session (security lab): applying the C2 replay to Plant 1 and running the C3 detector now
flagged 5/5 attacked days, all matching the recorded C3 verdicts, and the engine rejected each day.

## Limitations (honest)

- **Missed injections are accepted.** The engine's unsafe-directive constraint is a keyword list; it does not
  stop instruction-following on missed held-out phrasings (51/219 accepted answers obeyed).
- **Same authors.** Attacks and detectors were written by the same project. The known-template rate (66/66) is an
  upper bound; the held-out set is the only out-of-design evidence.
- **The telemetry detector is day-level and offline.** It is not in `/forecast`, so served forecasts are at best
  "review".
- **Some attacks are invisible.** Plant 2 halved power is invisible to every telemetry check (0/5). The Farm A
  SCADA features have no detector, and the deployed fault detector is fully evaded (recall 0.750 -> 0.000).
- **No authentication** on any endpoint, and no rate limiting.
- **Untuned decision rules.** The thresholds and floors are design choices with no labelled accept/review data to
  tune them; the sensitivity analysis shows how decisions move with them.
- **The judge is not ground truth.** The same qwen family judges answer quality, and the obedience markers
  undercount.

## Likely viva questions

1. **Why can't the LLM decide on its own?** It only recommends. The engine re-verifies every evidence document
   against the signed registry and checks the constraints. Only "accept" may be acted on, and nothing is executed.
2. **Why is every diagnosis "review"?** C is the detector's AUC CI lower bound rescaled. With CI low = 0.500 the
   detector is not shown to beat chance, so C = 0, which is below the 0.5 floor. That floor cannot be configured to 0.
3. **Why a signed registry, when each document already carries a hash?** An attacker can compute a document's own
   hash: the self-declared hash caught 0/17 tampered documents. Membership in the HMAC-signed build plus a hash
   match caught 17/17.
4. **What is S, and why use the likelihood ratio?** S is 1 - (1 - TPR) / (1 - FPR): how much a "pass" lowers the
   odds of an attack. Both rates are measured in C3 (e.g. injection: TPR 11/15, FPR 1/15).
5. **What is the biggest weakness?** Instruction-following on missed held-out injections, which the engine accepts.
   The fix would be a semantic injection classifier, or an output-side policy check rather than keywords.
6. **Did security cost answer quality?** Not measurably on clean questions: faithfulness 0.970 vs 0.970 and
   0/66 wrongly blocked. Under attack, the cost is review load.

## Demo script (live)

1. Open the console's **Security lab** tab. Under **Telemetry attacks**, run `solar_plant1 - replay`. Show:
   - the clean vs attacked signal;
   - 5/5 days flagged live, matching the recorded verdicts;
   - recorded MAE clean -> attacked;
   - REJECT for every day.
2. Run `solar_plant2 - power_scale`: 0/5 flagged. That is the detector miss, shown honestly in red.
3. **KB poisoning:** check `trouble-wind-pitch_system-rev2`. Content checks alone trust it; only the registry
   check rejects it. The genuine original is trusted.
4. **Prompt injection:** the C2 template preset is blocked and rejected. Then held-out #6 ("Pretend you are the
   plant manager...") is **not blocked**. Optionally tick "run the full pipeline" to show what qwen does.
5. **Held-out weakness:** show the per-phrasing table and the 51/219 callout.
6. **Assistant tab:** ask an injected question and show the REJECT trust panel.

## What is not done

- No semantic or LLM-based injection classifier, and no output-side policy model.
- No authentication or authorization, rate limiting, or audit log beyond the request log.
- No live telemetry gate on the served forecast path; no streaming detection.
- No detector for the Farm A SCADA features (fault-detector evasion is undefended).
- No calibration of the engine thresholds against labelled human decisions.
- No real (non-synthetic) attack data, and no red-team by people outside the project.
