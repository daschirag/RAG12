"""Pydantic schemas for the forecasting API (POST /forecast)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .uncertainty import LEVELS

CONFIDENCE_DESCRIPTION = (
    "Nominal coverage level of the prediction interval, e.g. 0.90 means the interval method "
    "targets containing the observed value in 90% of cases over many forecasts (marginal, "
    "not per forecast). It is NOT the probability that the point prediction is correct. "
    "Achieved coverage can differ from the nominal level because time series are not "
    "exchangeable; measured coverage is reported in results/forecasting/a4_summary.md, "
    "a5_summary.md and wind/wind_summary.md."
)

AssetId = Literal["solar_plant1", "solar_plant2", "wind_t1"]
GRID_MINUTES = {"solar_plant1": 15, "solar_plant2": 15, "wind_t1": 10}


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"asset": "solar_plant1", "origin": "2020-06-16T11:00", "confidence": 0.9},
        {"asset": "wind_t1", "origin": "2018-12-03T14:00", "confidence": 0.8}]})

    plant: Literal[1, 2] | None = Field(
        None, description="Solar plant (1 or 2). Legacy form; equivalent to asset 'solar_plant1'/'solar_plant2'.")
    asset: AssetId | None = Field(
        None, description="Asset id: 'solar_plant1', 'solar_plant2' or 'wind_t1'. Give exactly one of plant/asset.")
    origin: datetime = Field(
        ...,
        description="Forecast origin: the last observed slot (naive local time, e.g. '2020-06-17T12:00'), on the "
        "asset's native grid (15 min for solar, 10 min for wind). Only data at or before this time is used; the "
        "forecast is for origin + 1 hour.",
    )
    confidence: float = Field(
        0.9, description=CONFIDENCE_DESCRIPTION + f" Allowed values: {', '.join(str(v) for v in LEVELS)}."
    )

    @field_validator("origin")
    @classmethod
    def _naive_whole_minutes(cls, v: datetime) -> datetime:
        if v.tzinfo is not None:
            raise ValueError("origin must be naive local time (no timezone), as in the datasets")
        if v.second or v.microsecond:
            raise ValueError("origin must be on the asset's grid (no seconds)")
        return v

    @field_validator("confidence")
    @classmethod
    def _supported_level(cls, v: float) -> float:
        for level in LEVELS:
            if abs(v - level) < 1e-9:
                return level
        raise ValueError(f"confidence must be one of {list(LEVELS)} (the calibrated levels)")

    @model_validator(mode="after")
    def _one_asset_on_grid(self) -> "ForecastRequest":
        if (self.plant is None) == (self.asset is None):
            raise ValueError("give exactly one of 'plant' (1 or 2) or 'asset'")
        step = GRID_MINUTES[self.asset_id]
        if self.origin.minute % step:
            raise ValueError(f"origin must be on the {step}-min grid for {self.asset_id}")
        return self

    @property
    def asset_id(self) -> str:
        return self.asset or f"solar_plant{self.plant}"


class ForecastResponse(BaseModel):
    prediction: float = Field(..., ge=0, description="Point forecast of AC power (kW) at the target time.")
    lower_bound: float = Field(..., ge=0, description="Lower end of the prediction interval (kW).")
    upper_bound: float = Field(..., ge=0, description="Upper end of the prediction interval (kW).")
    confidence: float = Field(..., gt=0, lt=1, description=CONFIDENCE_DESCRIPTION)

    @model_validator(mode="after")
    def _ordered(self) -> "ForecastResponse":
        if not self.lower_bound <= self.prediction <= self.upper_bound:
            raise ValueError("expected lower_bound <= prediction <= upper_bound")
        return self


class ForecastResult(ForecastResponse):
    """Full POST /forecast response: the four plan fields plus context."""

    asset: str
    plant: int | None = Field(None, description="Solar plant number, null for wind.")
    origin: datetime
    target_time: datetime = Field(..., description="origin + horizon_minutes.")
    horizon_minutes: int
    observed_kw: float | None = Field(
        None, description="Measured AC power at target_time if it exists in the dataset (else null)."
    )
    in_sample: bool = Field(
        ...,
        description="True if target_time falls on a day the model was trained, tuned or calibrated on, so "
        "the forecast is not out-of-sample. Only targets after the last day of the record are out-of-sample.",
    )
    calibration_group: str = Field(
        ..., description="Mondrian calibration group used for the interval: 'daylight'/'night' (solar) or "
        "'power regime low/mid/high' (wind, from the point forecast).")
    daylight_group: bool | None = Field(None, description="Solar only: True for a daylight slot; null for wind.")
    method: str = Field(..., description="Interval method (cqr_mondrian = CQR calibrated per group).")
    unit: str = "kW"
    model_version: str
