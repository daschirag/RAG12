"""A6: build the deployed forecast models used by POST /forecast.

    python -m experiments.build_forecast_model

Writes models/forecasting/{solar_plant1,solar_plant2,wind_t1}.joblib
(gitignored) and results/forecasting/a6_model_card.json (splits, chosen
hyperparameters, conformal offsets, environment).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.backend.evaluation.provenance import environment_info
from src.backend.forecasting.service import (
    DEFAULT_MODEL_DIR,
    PLANTS,
    build_plant_model,
    build_wind_model,
    save_plant_model,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "forecasting"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    card = {"environment": environment_info(), "model_dir": str(DEFAULT_MODEL_DIR), "assets": {}}
    builders = [lambda p=p: build_plant_model(p) for p in PLANTS] + [build_wind_model]
    for build in builders:
        model = build()
        path = save_plant_model(model)
        card["assets"][model.asset] = {
            "artifact": path.name, "kind": model.kind, "segments": model.segments,
            "night_threshold": model.night_threshold, "capacity_kw": model.capacity_kw,
            "cqr_offsets_kw": {str(level): {str(g): q for g, q in o.items()} for level, o in model.offsets.items()},
            **model.info,
        }
        print(f"{model.asset}: saved {path.name}; train {model.segments['train']}, "
              f"val {model.segments['val']}, calib {model.segments['calib']}")
    (RESULTS / "a6_model_card.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
