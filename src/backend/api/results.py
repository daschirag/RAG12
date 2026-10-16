"""GET /results/summary and GET /results/summary/{track}: measured numbers read from results/ (never typed in).

Every value is read from (or aggregated over) a result file produced by a real run; ``sources`` lists the files.
The caveat sentences are built from the same numbers, so they change if the results change.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from .examples import ex
from ..api_support import ApiError
from ..settings import PROJECT_ROOT

router = APIRouter(tags=["Results"])
RES = PROJECT_ROOT / "results"
Track = Literal["forecasting", "rag-diagnosis", "security-trust", "cloud"]
TRACKS: dict[str, dict[str, str]] = {
    "forecasting": {"title": "Track A - Forecasting with uncertainty", "owner": "Minmini K (23BIT0198)"},
    "rag-diagnosis": {"title": "Track B - Knowledge base, hybrid RAG and fault diagnosis",
                      "owner": "Tarunica S (23BIT0185)"},
    "security-trust": {"title": "Track C - Security and trust-aware decisions",
                       "owner": "Pritika Kasinathan (23BIT0232)"},
    "cloud": {"title": "I4 - Cloud portability", "owner": "shared"},
}
SUMMARIES = {
    "forecasting": ["forecasting/a5_summary.md", "forecasting/wind/wind_summary.md", "forecasting/a4_summary.md"],
    "rag-diagnosis": ["kb/kb_summary.md", "kb/b2_chunking_summary.md", "kb/b3_retrieval_summary.md",
                      "faultdiag/b4_summary.md", "faultdiag/b5_summary.md", "rag/b6_summary.md",
                      "rag/run2/b6_followup_deterministic.md"],
    "security-trust": ["security/c2_summary.md", "security/c3_summary.md", "security/c4_summary.md",
                       "trust/c5_summary.md", "security/c6_summary.md"],
    "cloud": ["cloud/i4_status.md"],
}


class TrackSummary(BaseModel):
    track: str
    title: str
    owner: str
    metrics: dict[str, Any] = Field(description="Numbers read from the result files in 'sources'.")
    caveats: list[str] = Field(description="Honest limitations, built from the same numbers.")
    sources: list[str]
    markdown: dict[str, str] | None = Field(None, description="Full summary files (include_markdown=true).")


class ResultsSummary(BaseModel):
    tracks: list[TrackSummary]
    headline_caveats: list[str]


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and math.isnan(o):
        return None
    if hasattr(o, "item"):
        return _clean(o.item())
    return o


def _csv(rel: str, used: list[str]) -> pd.DataFrame:
    path = RES / rel
    if not path.is_file():
        raise ApiError(503, f"result file results/{rel} is missing", hint="Run python -m experiments.run_all to "
                       "see which experiment produces it.")
    used.append(f"results/{rel}")
    return pd.read_csv(path)


def _json(rel: str, used: list[str]) -> dict:
    path = RES / rel
    if not path.is_file():
        raise ApiError(503, f"result file results/{rel} is missing", hint="Run python -m experiments.run_all.")
    used.append(f"results/{rel}")
    return json.loads(path.read_text(encoding="utf-8"))


def _agg(df: pd.DataFrame, by: list[str], cols: list[str], count: str = "seeds") -> list[dict]:
    """Mean and std per group; ``count`` names the group size (seeds for models, queries for retrieval)."""
    g = df.groupby(by, sort=False)[cols]
    mean, std, n = g.mean(), g.std(), g.size()
    out = []
    for key in mean.index:
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row[count] = int(n.loc[key])
        for c in cols:
            row[c] = float(mean.loc[key, c])
            sd = std.loc[key, c]
            row[f"{c}_std"] = None if pd.isna(sd) else float(sd)
        out.append(row)
    return out


# ------------------------------------------------------------------ tracks

def forecasting(used: list[str]) -> tuple[dict, list[str]]:
    fixed = _csv("forecasting/a5_fixed_split.csv", used)
    solar = _agg(fixed, ["plant", "model"], ["mae", "rmse", "mape_pct", "r2"])
    wind_pm = _csv("forecasting/wind/point_metrics.csv", used)
    wind = _agg(wind_pm, ["model"], ["mae", "rmse", "mape_pct", "r2"])
    cov = _csv("forecasting/a5_coverage_ci.csv", used)
    wcov = _csv("forecasting/wind/coverage_ci.csv", used)
    boot = _csv("forecasting/a5_bootstrap_point.csv", used)
    coverage = [{"asset": f"solar_plant{int(r.plant)}", "level": r.level, "method": r.method, "coverage": r.coverage,
                 "ci_low": r.ci_low, "ci_high": r.ci_high, "days": int(r.days)} for r in cov.itertuples()]
    coverage += [{"asset": "wind_t1", "level": r.level, "method": r.method, "coverage": r.coverage,
                  "ci_low": r.ci_low, "ci_high": r.ci_high, "days": int(r.days)} for r in wcov.itertuples()]
    vs_persistence = [{"plant": int(r.plant), "model": r.model_b, "persistence_minus_model_mae": r.mean_diff,
                       "ci_low": r.ci_low, "ci_high": r.ci_high, "days": int(r.n_days),
                       "days_model_better": int(r.n_days - r.days_a_better)}
                      for r in boot[boot.model_a == "persistence"].itertuples()]
    wm = {r["model"]: r for r in wind}
    rep = _json("forecasting/preprocessing_report_plant1.json", used)["report"]
    solar_days = (pd.Timestamp(rep["grid_end"]).normalize() - pd.Timestamp(rep["grid_start"]).normalize()).days + 1
    learned = [m for m in ("random_forest", "xgboost", "lstm", "tcn") if m in wm]
    best = min(learned, key=lambda m: wm[m]["mae"])
    p1 = {r["model"]: r for r in solar if r["plant"] == 1}
    c90 = {c["asset"]: c for c in coverage if c["level"] == 0.9 and c["method"] == "cqr_mondrian"}
    caveats = [
        f"Wind T1: persistence MAE {wm['persistence']['mae']:.1f} kW is lower than every learned model "
        f"(best {best} {wm[best]['mae']:.1f} kW); learned models only win on RMSE.",
        f"Solar has only {solar_days} days per plant ({rep['grid_start'][:10]} to {rep['grid_end'][:10]}); the "
        f"rolling-origin evaluation covers {c90['solar_plant1']['days']} test "
        f"days, so the Plant 1 90% coverage {c90['solar_plant1']['coverage']:.3f} has a wide CI "
        f"[{c90['solar_plant1']['ci_low']:.3f}, {c90['solar_plant1']['ci_high']:.3f}].",
        "The deployed models were fit on the whole record (train/val/calib), so forecasts served by the API are "
        "in-sample (in_sample = true); out-of-sample accuracy is the rolling-origin result above.",
        f"Solar Plant 1 fixed split: persistence MAE {p1['persistence']['mae']:.1f} kW vs random forest "
        f"{p1['random_forest']['mae']:.1f} kW.",
    ]
    return {"solar_fixed_split": solar, "wind_rolling_origin": wind, "coverage": coverage,
            "solar_vs_persistence_rolling": vs_persistence}, caveats


def rag_diagnosis(used: list[str]) -> tuple[dict, list[str]]:
    from ..trust.signals import load_signals

    docs_path = PROJECT_ROOT / "data" / "processed" / "knowledge_base" / "documents.jsonl"
    types: dict[str, dict[str, int]] = {}
    if docs_path.is_file():
        used.append("data/processed/knowledge_base/documents.jsonl")
        for line in docs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                t = types.setdefault(d["doc_type"], {"real": 0, "synthetic": 0})
                t["synthetic" if d.get("synthetic") else "real"] += 1
    chunking = _agg(_csv("kb/b2_chunking_metrics.csv", used), ["method"],
                    ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"], count="queries")
    retrieval = _agg(_csv("kb/b3_retrieval_metrics.csv", used), ["variant"],
                     ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"], count="queries")
    b6 = _csv("rag/b6_per_answer.csv", used)
    kb = b6[b6.dataset == "kb"]
    rag = [{"system": s, "faithfulness": g.faithfulness.mean(), "hallucination_rate": g.hallucinated.mean(),
            "answer_relevance": g.answer_relevance.mean(), "items": len(g)} for s, g in kb.groupby("system", sort=False)]
    det = _agg(_csv("faultdiag/b4_event_metrics.csv", used), ["model"],
               ["accuracy", "precision", "recall", "f1", "auc"])
    b5 = _csv("faultdiag/b5_results.csv", used)
    an = b5[b5.label == "anomaly"]
    diag = [{"mode": m, "component_correct": int(g.component_correct.sum()), "events": len(g)}
            for m, g in an.groupby("mode", sort=False)]
    sig = load_signals()
    used.append("results/faultdiag/b4_summary.md (detector AUC CI via trust.signals)")
    green = next(r for r in rag if r["system"] == "greenrag")
    caveats = [
        f"The deployed fault detector (logistic regression) has event AUC {sig.detector_auc:.3f} with 95% CI "
        f"[{sig.detector_ci_low:.3f}, {sig.detector_ci_high:.3f}]: not distinguishable from chance, so the "
        "decision engine never accepts a diagnosis.",
        f"{sum(t['synthetic'] for t in types.values())} of {sum(sum(t.values()) for t in types.values())} knowledge-base "
        "documents are synthetic (marked [SYNTHETIC]); the labelled faults are wind-turbine faults, not solar-inverter "
        "faults.",
        f"GreenRAG faithfulness {green['faithfulness']:.3f} and hallucination rate {green['hallucination_rate']:.3f} on "
        f"{green['items']} knowledge-base questions are judged by a local LLM (qwen2.5:7b-instruct), not by people.",
    ] + [f"Unlogged-mode diagnosis: likely-cause component correct for {d['component_correct']}/{d['events']} anomaly "
         "events (the event's own record hidden)." for d in diag if d["mode"] == "unlogged"]
    return {"kb_documents_by_type": types, "chunking": chunking, "retrieval": retrieval, "rag_kb_questions": rag,
            "fault_detectors": det, "diagnosis_component_correct": diag,
            "detector_auc": {"auc": sig.detector_auc, "ci_low": sig.detector_ci_low,
                             "ci_high": sig.detector_ci_high}}, caveats


def security_trust(used: list[str]) -> tuple[dict, list[str]]:
    att = _json("security/c6_attack_success.json", used)
    inj = _json("security/c6_injection_defence.json", used)
    util = _json("security/c6_utility.json", used)
    de = _json("security/c6_decision_engine.json", used)
    fc = _json("security/c6_forecast_confidence.json", used)
    held = de["accepted_answer_correctness"]["heldout_injection"]
    acc = held["accepted"]
    obey = held["accepted_that_obeyed"]
    hd = inj["held_out"]["detected"]
    caveats = [
        f"Held-out injection phrasings: the detector blocked {hd['k']}/{hd['n']}; of {held['delivered']} delivered "
        f"answers the engine accepted {acc['k']}, and {obey['k']} of those {obey['n']} accepted answers executed the "
        "injected instruction.",
        "All attacks and detectors were written by the same project; detection rates on the known C2 template are an "
        "upper bound.",
        "The C3 telemetry detector decides per whole day and is not wired into the served /forecast endpoint.",
    ]
    return {"attack_success": att, "injection_defence": inj, "utility": util, "decision_engine": de,
            "forecast_confidence": fc}, caveats


def cloud(used: list[str]) -> tuple[dict, list[str]]:
    st = _json("cloud/i4_status.json", used)
    return ({"status": st["status"], "settings_present": st["settings_present"], "checked": st.get("timestamp_utc")},
            ["Nothing is deployed to any cloud; local-vs-cloud latency and equivalence are not measured "
             f"(I4 status {st['status']})."])


BUILDERS = {"forecasting": forecasting, "rag-diagnosis": rag_diagnosis, "security-trust": security_trust,
            "cloud": cloud}


def build_track(track: str, include_markdown: bool = False) -> TrackSummary:
    used: list[str] = []
    metrics, caveats = BUILDERS[track](used)
    md = None
    if include_markdown:
        md = {f"results/{p}": (RES / p).read_text(encoding="utf-8") for p in SUMMARIES[track] if (RES / p).is_file()}
    return TrackSummary(track=track, **TRACKS[track], metrics=_clean(metrics), caveats=caveats,
                        sources=list(dict.fromkeys(used + [f"results/{p}" for p in SUMMARIES[track]])), markdown=md)


@router.get("/results/summary", response_model=ResultsSummary, responses=ex("GET /results/summary"))
def results_summary() -> ResultsSummary:
    """Measured numbers for every track, read from results/ (CSV/JSON written by the experiments), with the
    honest caveats. Nothing here is typed in by hand."""
    tracks = [build_track(t) for t in TRACKS]
    headline = [c for t in tracks for c in t.caveats[:1]]
    return ResultsSummary(tracks=tracks, headline_caveats=headline)


@router.get("/results/summary/{track}", response_model=TrackSummary, responses=ex("GET /results/summary/{track}"))
def results_track(track: Track, include_markdown: bool = Query(False, description="Also return the summary "
                                                                                  "markdown files.")) -> TrackSummary:
    """One track's measured numbers (forecasting, rag-diagnosis, security-trust, cloud)."""
    return build_track(track, include_markdown)
