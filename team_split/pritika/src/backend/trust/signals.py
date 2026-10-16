"""C5 - Measured inputs of the trust score, loaded from the repository's saved results (never hard-coded).

- Detector reliability (C for diagnoses): B4 logistic-regression event AUC and its 95% bootstrap CI.
- GreenRAG answer reliability (C for knowledge-base answers): B6 claim-level faithfulness of GreenRAG on the KB
  questions; lower end of a seeded bootstrap 95% CI of the mean.
- Fallback penalty (part of E): B6 answer relevance of GreenRAG's extractive fallback relative to its
  LLM-generated answers (a quote is faithful but answers the question less well).
- Forecast interval reliability (C for forecasts): A5 rolling-origin coverage CIs of the deployed CQR-Mondrian
  intervals (solar per plant, wind T1).
- Security check strength (S): C3 detection rate (TPR) and false-positive rate (FPR) of each check - injection
  detector on the held-out phrasings (not the C2 template, which the detector's author knew), document trust
  (full registry check vs content checks only; FPR on the clean KB), telemetry detector per asset (C2-attacked
  days vs clean days). A check's strength is ``1 - LR-`` with the negative likelihood ratio
  ``LR- = (1 - TPR) / (1 - FPR)``: how much a "pass" lowers the odds that the input is an attack (1 = a pass
  rules the attack out, 0 = a pass says nothing).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class Rate:
    value: float
    n: int
    source: str


@dataclass(frozen=True)
class Check:
    """A security check's measured detection rate (TPR) and false-positive rate (FPR)."""

    tpr: Rate
    fpr: Rate

    @property
    def strength(self) -> float:
        if self.fpr.value >= 1.0:
            return 0.0
        return max(0.0, 1.0 - (1.0 - self.tpr.value) / (1.0 - self.fpr.value))


@dataclass(frozen=True)
class Signals:
    detector_auc: float
    detector_ci_low: float
    detector_ci_high: float
    answer_faithfulness_low: Rate
    fallback_relevance_ratio: Rate
    injection_check: Check
    trust_check_full: Check
    trust_check_content_only: Check
    forecast_coverage_low: dict           # asset -> (nominal level, ci_low, source)
    telemetry_check: dict                 # asset -> Check of the C3 telemetry detector (C2 attacks vs clean days)


def _bootstrap_low(x: np.ndarray, seed: int = 0, n: int = 10_000) -> float:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.quantile(means, 0.025))


def _rate(flags: pd.Series, source: str) -> Rate:
    x = flags.astype(bool).to_numpy()
    return Rate(float(x.mean()), len(x), source)


@lru_cache(maxsize=1)
def load_signals(results: Path = RESULTS) -> Signals:
    from ..faultdiag.diagnosis import load_reliability

    rel = load_reliability(results / "faultdiag")
    b6 = pd.read_csv(results / "rag" / "b6_per_answer.csv")
    g = b6[(b6.system == "greenrag") & (b6.dataset == "kb")]
    faith = g["faithfulness"].dropna().to_numpy(dtype=float)
    ar = g.groupby("route")["answer_relevance"].mean()
    ratio = float(min(1.0, ar["extractive_fallback"] / ar["generated"]))
    inj = pd.read_csv(results / "security" / "c3_injection.csv")
    held = inj[inj.source == "held_out"]
    src_inj = "results/security/c3_injection.csv held_out"
    injection = Check(_rate(held[held.label == 1].flagged, src_inj), _rate(held[held.label == 0].flagged, src_inj))
    trust = pd.read_csv(results / "security" / "c3_trust_docs.csv")
    attacks = trust[trust.scenario.isin(["added", "in_place"])]
    clean = trust[trust.scenario == "clean_kb"]
    src_tr = "results/security/c3_trust_docs.csv"
    full = Check(_rate(~attacks.trusted.astype(bool), src_tr), _rate(~clean.trusted.astype(bool), src_tr))
    content = Check(_rate(~attacks.content_only_trusted.astype(bool), src_tr),
                    _rate(~clean.content_only_trusted.astype(bool), src_tr))
    cov = pd.read_csv(results / "forecasting" / "a5_coverage_ci.csv")
    wind = pd.read_csv(results / "forecasting" / "wind" / "coverage_ci.csv")
    forecast = {}
    for plant in (1, 2):
        r = cov[(cov.plant == plant) & (cov.level == 0.9) & (cov.method == "cqr_mondrian")].iloc[0]
        forecast[f"solar_plant{plant}"] = (0.9, float(r.ci_low), "results/forecasting/a5_coverage_ci.csv")
    r = wind[(wind.level == 0.9) & (wind.method == "cqr_mondrian")].iloc[0]
    forecast["wind_t1"] = (0.9, float(r.ci_low), "results/forecasting/wind/coverage_ci.csv")
    tel = pd.read_csv(results / "security" / "c3_telemetry_days.csv")
    tel = tel[tel.attack != "replay_noisy"]                               # the C2 attacks and clean days only
    src_te = "results/security/c3_telemetry_days.csv"
    telemetry = {a: Check(_rate(g[g.label == 1]["any"].astype(bool), src_te),
                          _rate(g[g.label == 0]["any"].astype(bool), src_te)) for a, g in tel.groupby("asset")}
    return Signals(
        detector_auc=rel.value, detector_ci_low=rel.ci95_low, detector_ci_high=rel.ci95_high,
        answer_faithfulness_low=Rate(_bootstrap_low(faith), len(faith), "results/rag/b6_per_answer.csv (greenrag, kb)"),
        fallback_relevance_ratio=Rate(ratio, int((g.route == "extractive_fallback").sum()),
                                      "results/rag/b6_per_answer.csv answer_relevance fallback / generated"),
        injection_check=injection, trust_check_full=full, trust_check_content_only=content,
        forecast_coverage_low=forecast, telemetry_check=telemetry,
    )
