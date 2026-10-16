"""Wind T1: where does persistence beat the learned models? (diagnostic only, no tuning)

    python -m experiments.wind_diagnose

Splits the scored test rows of the rolling-origin folds by (a) the observed
power level at the target and (b) whether power changed over the hour, and
reports MAE per group for persistence and the learned models (seed-averaged
predictions). Output: results/forecasting/wind/wind_diagnosis.md and .csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.wind_evaluate import load
from src.backend.forecasting.wind import RATED_KW

OUT = Path(__file__).resolve().parents[1] / "results" / "forecasting" / "wind"
MODELS = ("persistence", "random_forest", "xgboost", "lstm", "tcn")


def main() -> None:
    base, preds = load()
    s = base[base.scored].copy()
    for m in MODELS:
        s[m] = np.mean([preds[k].loc[s.index, "pred"].to_numpy() for k in preds if k.startswith(m + "__")], axis=0)
    y = s.y_true
    change = (y - s["persistence"]).abs()                       # |y(T) - y(t)|: how much power moved in the hour
    s["level"] = pd.cut(y / RATED_KW, [-0.01, 0.01, 0.10, 0.90, 2.0],
                        labels=["~0 (<1%)", "low (1-10%)", "mid (10-90%)", "rated (>=90%)"])
    s["movement"] = pd.cut(change / RATED_KW, [-0.01, 0.01, 0.10, 2.0],
                           labels=["flat (<1% change)", "moderate (1-10%)", "ramp (>10%)"])
    rows = []
    for col in ("level", "movement"):
        for g, sub in s.groupby(col, observed=True):
            rows.append({"split": col, "group": str(g), "rows": len(sub), "share": len(sub) / len(s),
                         **{m: float((sub.y_true - sub[m]).abs().mean()) for m in MODELS}})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "wind_diagnosis.csv", index=False, encoding="utf-8")
    L = ["# Wind T1: where persistence wins (diagnostic, test rows of the 4 rolling-origin months)", "",
         "MAE (kW) per group; learned models use seed-averaged predictions. 'level' = observed power at the",
         "target as a share of rated power; 'movement' = |power(T) - power(t)|, i.e. how much the hour changed.", "",
         "| split | group | rows | share | " + " | ".join(MODELS) + " |", "|---|---|---|---|" + "---|" * len(MODELS)]
    for _, r in table.iterrows():
        L.append(f"| {r.split} | {r.group} | {r.rows} | {r.share:.1%} | "
                 + " | ".join(f"{r[m]:.1f}" for m in MODELS) + " |")
    (OUT / "wind_diagnosis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[5:]))


if __name__ == "__main__":
    main()
