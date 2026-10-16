"""Lookup endpoints the console needs: fault events, forecast assets and day curves, suggested questions and the
text of cited knowledge-base chunks. All values come from the datasets, the deployed models or result files."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from .. import deps
from .examples import ex
from ..api_support import ApiError
from ..forecasting.schemas import AssetId
from ..forecasting.service import ForecastService, ModelNotBuiltError, OriginError
from ..settings import PROJECT_ROOT

TAG_A = "Track A - Forecasting"
TAG_B = "Track B - RAG and fault diagnosis"
router = APIRouter()


# ------------------------------------------------------------------ Track B: events, suggestions, chunks

class FaultEvent(BaseModel):
    event_id: int
    asset_id: int
    label: str = Field(description="anomaly or normal (event_info.csv)")
    start: datetime
    end: datetime
    logged_description: str | None = Field(None, description="Logged root cause (ground truth; hidden from the "
                                                             "diagnosis in 'unlogged' mode).")
    component: str | None = Field(None, description="Component derived from the logged description (B1 rules).")
    detector_score: float | None = Field(None, description="B4 logistic-regression out-of-fold event score.")
    detector_flag: bool | None = None


@lru_cache(maxsize=1)
def _events() -> list[FaultEvent]:
    from ..kb.taxonomy import categorize, load_events

    events = load_events("A")
    path = PROJECT_ROOT / "results" / "faultdiag" / "b4_event_predictions.csv"
    pred = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=["event_id", "model", "seed"])
    lr = pred[(pred.model == "logistic_regression") & (pred.seed == 0)].set_index("event_id")
    out = []
    for e in events.sort_values("event_id").itertuples():
        desc = e.event_description if isinstance(e.event_description, str) else None
        comp = categorize(desc)[0] if desc and e.event_label == "anomaly" else None
        has = e.event_id in lr.index
        out.append(FaultEvent(event_id=int(e.event_id), asset_id=int(e.asset_id), label=e.event_label,
                              start=e.event_start, end=e.event_end, logged_description=desc, component=comp,
                              detector_score=float(lr.loc[e.event_id, "event_prob"]) if has else None,
                              detector_flag=bool(lr.loc[e.event_id, "event_flag"]) if has else None))
    return out


@router.get("/diagnose/events", response_model=list[FaultEvent], tags=[TAG_B], responses=ex("GET /diagnose/events"))
def diagnose_events() -> list[FaultEvent]:
    """Wind Farm A events (event_info.csv) with the B4 detector score that POST /diagnose uses."""
    return _events()


class Suggestion(BaseModel):
    query_id: str
    text: str
    query_type: str


@router.get("/assistant/suggestions", response_model=list[Suggestion], tags=["Integration"], responses=ex("GET /assistant/suggestions"))
def suggestions(per_type: int = Query(1, ge=1, le=5)) -> list[Suggestion]:
    """Suggested questions: the first labelled B2 evaluation queries of each type (real KB questions)."""
    from ..kb.build import build_documents
    from ..kb.queries import build_queries

    seen: dict[str, int] = {}
    out = []
    for q in build_queries(build_documents()):
        if seen.get(q.query_type, 0) < per_type:
            seen[q.query_type] = seen.get(q.query_type, 0) + 1
            out.append(Suggestion(query_id=q.query_id, text=q.text, query_type=q.query_type))
    return out


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str | None
    doc_type: str | None
    synthetic: bool
    text: str


@router.get("/kb/chunks", response_model=list[Chunk], tags=[TAG_B], responses=ex("GET /kb/chunks"))
def kb_chunks(ids: str = Query(..., min_length=3, max_length=2000,
                               description="Comma-separated chunk ids, e.g. 'fault-a-73::0,maint-a-73::1'"),
              sec=Depends(deps.get_secure_retrieval)) -> list[Chunk]:
    """Text of knowledge-base chunks (for the source cards). Loads the secure retrieval index on first use."""
    wanted = [i.strip() for i in ids.split(",") if i.strip()][:20]
    by_id = {c["chunk_id"]: c for c in sec.chunks}
    docs = {d.doc_id: d for d in sec.store_docs}
    out = []
    for cid in wanted:
        c = by_id.get(cid)
        if c is None:
            continue
        d = docs.get(c["metadata"]["doc_id"])
        out.append(Chunk(chunk_id=cid, doc_id=c["metadata"]["doc_id"], title=d.title if d else None,
                         doc_type=d.doc_type if d else None, synthetic=bool(d.synthetic) if d else False,
                         text=c["text"]))
    if not out:
        raise ApiError(404, f"none of the chunk ids {wanted} exist in the knowledge-base index",
                       hint="Use chunk ids from an answer's evidence list.")
    return out


# ------------------------------------------------------------------ Track A: assets and day curves

class AssetInfo(BaseModel):
    asset: str
    kind: str
    plant: int | None
    step_minutes: int
    horizon_minutes: int
    capacity_kw: float
    first_origin: datetime
    last_origin: datetime
    segments: dict[str, list[str]] = Field(description="Days the deployed model was fit on (train / val / calib).")
    levels: list[float]
    method: str
    trained_at_utc: str


def _forecast_errors(fn):
    try:
        return fn()
    except ModelNotBuiltError as exc:
        raise ApiError(503, str(exc), hint="Run python -m experiments.build_forecast_model") from exc
    except OriginError as exc:
        raise ApiError(422, str(exc), hint="Pick a time inside the asset's record (GET /forecast/assets).") from exc


@router.get("/forecast/assets", response_model=list[AssetInfo], tags=[TAG_A], responses=ex("GET /forecast/assets"))
def forecast_assets(service: ForecastService = Depends(deps.get_forecast_service)) -> list[AssetInfo]:
    """The three forecast assets: grid, horizon, valid origin range and the days the deployed model was fit on."""
    return _forecast_errors(lambda: [AssetInfo(**service.asset_info(a))
                                     for a in ("solar_plant1", "solar_plant2", "wind_t1")])


class SeriesPoint(BaseModel):
    origin: datetime
    target_time: datetime
    prediction: float
    lower_bound: float
    upper_bound: float
    observed_kw: float | None
    calibration_group: str
    in_sample: bool


class SeriesResponse(BaseModel):
    asset: str
    confidence: float
    points: list[SeriesPoint]
    coverage_observed: float | None = Field(None, description="Share of points with an observation whose interval "
                                                              "contains it (this window only).")


@router.get("/forecast/series", response_model=SeriesResponse, tags=[TAG_A], responses=ex("GET /forecast/series"))
def forecast_series(asset: AssetId = Query(..., description="solar_plant1, solar_plant2 or wind_t1"),
                    start: datetime = Query(..., description="First origin, e.g. 2020-06-16T00:00"),
                    end: datetime = Query(..., description="Last origin (at most 3 days after start)"),
                    confidence: float = Query(0.9, description="0.8, 0.9 or 0.95"),
                    service: ForecastService = Depends(deps.get_forecast_service)) -> SeriesResponse:
    """1-hour-ahead forecasts for every origin in a window, with the interval and the observed value (same
    computation as POST /forecast; bounds agree to float32 rounding, < 0.01 kW)."""
    if end < start or (end - start).days > 3:
        raise ApiError(422, "end must be after start and at most 3 days later", hint="Request one day at a time.")
    pts = _forecast_errors(lambda: service.series(asset, start, end, confidence))
    obs = [p for p in pts if p["observed_kw"] is not None]
    cov = (sum(p["lower_bound"] <= p["observed_kw"] <= p["upper_bound"] for p in obs) / len(obs)) if obs else None
    return SeriesResponse(asset=asset, confidence=confidence, points=[SeriesPoint(**p) for p in pts],
                          coverage_observed=cov)
