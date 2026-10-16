"""Wind T1 replication: evaluation report from the rolling-origin month folds.

    python -m experiments.wind_evaluate

Inputs: results/forecasting/wind/folds/<month>/*.csv (experiments.wind_rolling).
Outputs (results/forecasting/wind/): wind_summary.md, preprocessing_report.json,
point_metrics.csv, point_by_day.csv, interval_metrics.csv, interval_by_day.csv,
bootstrap_point.csv, bootstrap_interval.csv, coverage_ci.csv and grayscale
figures wind_monthly_mae.png, wind_coverage_regime.png, wind_error_by_speed.png.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backend.evaluation.provenance import environment_info  # noqa: E402
from src.backend.forecasting.evaluation import day_bootstrap_ratio, paired_day_bootstrap  # noqa: E402
from src.backend.forecasting.metrics import mae, mape, r2, rmse  # noqa: E402
from src.backend.forecasting.uncertainty import LEVELS, METHODS, PROPOSED, interval_metrics  # noqa: E402
from src.backend.forecasting.wind import RATED_KW, REGIMES, preprocess_wind_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "forecasting" / "wind"
FOLDS = OUT / "folds"
MODELS = ("persistence", "daily_persistence", "random_forest", "xgboost", "lstm", "tcn", "xgboost_oracle_weather")
LEARNED = ("random_forest", "xgboost", "lstm", "tcn")
MAPE_MIN_KW = 0.05 * RATED_KW      # 180 kW
N_BOOT = 10_000


def _ms(v, d: int) -> str:
    v = pd.Series(v, dtype=float)
    return f"{v.mean():.{d}f} +/- {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v.iloc[0]:.{d}f}"


def load() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Base rows (all months) and one frame per model-seed file (indexed by origin)."""
    bases, preds = [], {}
    for month_dir in sorted(p for p in FOLDS.iterdir() if p.is_dir()):
        base = pd.read_csv(month_dir / "base.csv", parse_dates=["origin", "target_time"])
        bases.append(base.assign(month=month_dir.name))
        for f in month_dir.glob("*__seed*.csv"):
            df = pd.read_csv(f, parse_dates=["origin"]).set_index("origin")
            preds.setdefault(f.stem, []).append(df)
    base = pd.concat(bases, ignore_index=True).set_index("origin")
    return base, {k: pd.concat(v).reindex(base.index) for k, v in preds.items()}


def point_tables(base: pd.DataFrame, preds: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, day_rows = [], []
    s = base[base.scored]
    day = s.target_time.dt.normalize()
    for key, df in preds.items():
        model, seed = key.split("__seed")
        p = df.loc[s.index, "pred"]
        y = s.y_true
        up = ~s.down
        rows.append({"model": model, "seed": int(seed), "rows": len(s), "mae": mae(y, p), "rmse": rmse(y, p),
                     "r2": r2(y, p), "mape_pct": mape(y, p, MAPE_MIN_KW), "nmae_pct": 100 * mae(y, p) / RATED_KW,
                     "mae_excl_down": mae(y[up], p[up]), "down_rows": int((~up).sum()),
                     **{f"mae_{m}": mae(y[s.month == m], p[s.month == m]) for m in sorted(s.month.unique())}})
        err = (y - p).abs()
        for d, e in err.groupby(day):
            day_rows.append({"model": model, "seed": int(seed), "day": f"{d:%Y-%m-%d}", "mae": float(e.mean())})
    return pd.DataFrame(rows), pd.DataFrame(day_rows)


def interval_tables(base: pd.DataFrame, preds: dict):
    s = base[base.scored]
    day = s.target_time.dt.normalize()
    rows, day_rows = [], []
    for key, df in preds.items():
        if not key.startswith("xgboost__"):
            continue
        seed = int(key.split("__seed")[1])
        d = df.loc[s.index]
        y = s.y_true.to_numpy()
        for method, level in itertools.product(METHODS, LEVELS):
            tag = f"{method}_{int(level * 100)}"
            lo, hi = d[f"{tag}_lo"].to_numpy(), d[f"{tag}_hi"].to_numpy()
            for subset, mask in [("all", np.ones(len(y), bool)), *[(f"regime_{r}", (d.regime == r).to_numpy())
                                                                   for r in REGIMES],
                                 *[(f"month_{m}", (s.month == m).to_numpy()) for m in sorted(s.month.unique())]]:
                if mask.any():
                    rows.append({"seed": seed, "method": method, "level": level, "subset": subset,
                                 **interval_metrics(y[mask], lo[mask], hi[mask], level)})
            for dd, idx in pd.Series(np.arange(len(y)), index=day.to_numpy()).groupby(level=0):
                i = idx.to_numpy()
                m = interval_metrics(y[i], lo[i], hi[i], level)
                day_rows.append({"seed": seed, "method": method, "level": level, "day": f"{dd:%Y-%m-%d}",
                                 "covered": m["coverage"] * m["rows"], **m})
    return pd.DataFrame(rows), pd.DataFrame(day_rows)


def bootstraps(pday: pd.DataFrame, iday: pd.DataFrame):
    loss = pday.groupby(["model", "day"])["mae"].mean()
    bp = []
    for a, b in itertools.combinations([m for m in MODELS if m != "xgboost_oracle_weather"], 2):
        r = paired_day_bootstrap(loss.loc[a], loss.loc[b], N_BOOT, seed=0)
        bp.append({"model_a": a, "model_b": b, **r.as_dict()})
    wl = iday.groupby(["level", "method", "day"])["winkler"].mean()
    bi = []
    for level in LEVELS:
        for other in METHODS:
            if other != PROPOSED:
                r = paired_day_bootstrap(wl.loc[(level, PROPOSED)], wl.loc[(level, other)], N_BOOT, seed=0)
                bi.append({"level": level, "method_a": PROPOSED, "method_b": other, **r.as_dict()})
    g = iday.groupby(["level", "method", "day"])[["covered", "rows"]].mean()
    cov = []
    for (level, method), sub in g.groupby(level=[0, 1]):
        sub = sub.droplevel([0, 1])
        point, lo, hi = day_bootstrap_ratio(sub["covered"], sub["rows"], N_BOOT, seed=0)
        cov.append({"level": level, "method": method, "coverage": point, "ci_low": lo, "ci_high": hi,
                    "shortfall_pts": 100 * (level - point), "days": len(sub)})
    return pd.DataFrame(bp), pd.DataFrame(bi), pd.DataFrame(cov)


def plots(point: pd.DataFrame, imet: pd.DataFrame, base: pd.DataFrame, preds: dict, frame: pd.DataFrame) -> None:
    months = sorted(base.month.unique())
    styles = {"persistence": ("black", ":", "o"), "daily_persistence": ("0.5", ":", "s"),
              "random_forest": ("black", "-", "^"), "xgboost": ("0.35", "-", "v"),
              "lstm": ("black", "--", "D"), "tcn": ("0.35", "--", "x")}
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, (c, ls, mk) in styles.items():
        v = point[point.model == model][[f"mae_{m}" for m in months]].mean()
        ax.plot(months, v.values, color=c, linestyle=ls, marker=mk, label=model)
    ax.set_ylabel("MAE (kW), mean over seeds")
    ax.set_title("Wind T1: test-month MAE, rolling origin by month (1 h ahead)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "wind_monthly_mae.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    hatches = ["", "//", "..", "xx", "\\\\"]
    x = np.arange(len(REGIMES))
    w = 0.16
    sel = imet[imet.level == 0.9]
    for i, method in enumerate(METHODS):
        v = [sel[(sel.method == method) & (sel.subset == f"regime_{r}")].coverage.mean() for r in REGIMES]
        ax.bar(x + (i - 2) * w, v, w, color="white", edgecolor="black", hatch=hatches[i], label=method)
    ax.axhline(0.9, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [f"{r} (predicted)" for r in REGIMES])
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("coverage (y axis starts at 0.5)")
    ax.set_title("Wind T1: test coverage by power regime, nominal 90% (dashed)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=5, loc="lower center")
    fig.tight_layout()
    fig.savefig(OUT / "wind_coverage_regime.png", dpi=130)
    plt.close(fig)

    s = base[base.scored]
    ws = frame["wind_speed"].reindex(s.target_time).to_numpy()
    bins = np.arange(0, 22, 2)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for model, (c, ls, mk) in styles.items():
        p = np.mean([preds[k].loc[s.index, "pred"].to_numpy() for k in preds if k.startswith(model + "__")], axis=0)
        err = np.abs(s.y_true.to_numpy() - p)
        idx = np.digitize(ws, bins)
        v = [err[idx == i].mean() if (idx == i).sum() > 50 else np.nan for i in range(1, len(bins))]
        ax.plot(bins[:-1] + 1, v, color=c, linestyle=ls, marker=mk, markersize=3, label=model)
    ax.set_xlabel("measured wind speed at the target time (m/s), 2 m/s bins")
    ax.set_ylabel("MAE (kW)")
    ax.set_title("Wind T1: error by wind speed (bins with > 50 rows)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "wind_error_by_speed.png", dpi=130)
    plt.close(fig)


def summary(env, rep, point, pday, imet, bp, bi, cov, base) -> str:
    months = sorted(base.month.unique())
    s = base[base.scored]
    L = ["# Wind T1 replication of Track A (1 hour ahead, 10-min resolution)", "",
         f"Generated by `python -m experiments.wind_evaluate` at base commit {env['git_commit']}.",
         f"Rolling origin by month: test months {', '.join(months)}; for test month m, train = months before m-2,",
         "val = m-2, calib = m-1. Seeds 0-2 (persistence baselines deterministic). "
         f"Scored rows: {len(s)} over {s.target_time.dt.normalize().nunique()} days.",
         f"MAPE only where y >= {MAPE_MIN_KW:.0f} kW (5% of rated {RATED_KW:.0f} kW). nMAE = MAE / rated power.", "",
         "## Preprocessing", "", "```", json.dumps(rep, indent=1), "```", "",
         "## Point forecasts (pooled over the 4 test months)", "",
         "| model | MAE (kW) | nMAE (%) | RMSE (kW) | MAPE (%) | R2 | MAE excl. downtime targets | "
         + " | ".join(f"MAE {m}" for m in months) + " |",
         "|---|---|---|---|---|---|---|" + "---|" * len(months)]
    for model in MODELS:
        p = point[point.model == model]
        note = " (upper bound, not deployable)" if model == "xgboost_oracle_weather" else ""
        L.append(f"| {model}{note} | {_ms(p.mae, 1)} | {_ms(p.nmae_pct, 2)} | {_ms(p.rmse, 1)} | {_ms(p.mape_pct, 1)} | "
                 f"{_ms(p.r2, 4)} | {_ms(p.mae_excl_down, 1)} | "
                 + " | ".join(f"{p[f'mae_{m}'].mean():.1f}" for m in months) + " |")
    L += ["", f"Downtime targets (power <= 10 kW while the power curve says >= 500 kW) in the scored rows: "
          f"{int(point.down_rows.iloc[0])}. They are kept in every metric except the 'excl. downtime' column.", "",
          "### Paired day-level bootstrap (per-day MAE, seed-averaged; 10,000 resamples)", "",
          "| A | B | mean diff (kW) | 95% CI | share A better | days A better |", "|---|---|---|---|---|---|"]
    for _, r in bp.iterrows():
        L.append(f"| {r.model_a} | {r.model_b} | {r.mean_diff:.1f} | [{r.ci_low:.1f}, {r.ci_high:.1f}] | "
                 f"{r.share_a_better:.3f} | {r.days_a_better}/{r.n_days} |")
    L += ["", "## Intervals (point model XGBoost; groups = predicted power regime)", "",
          "| level | method | coverage | 95% day-bootstrap CI | shortfall (points) | mean width (kW) | Winkler | pinball |",
          "|---|---|---|---|---|---|---|---|"]
    allrows = imet[imet.subset == "all"].groupby(["level", "method"])[["mean_width", "winkler", "pinball"]].mean()
    for _, r in cov.sort_values(["level", "method"]).iterrows():
        a = allrows.loc[(r.level, r.method)]
        L.append(f"| {r.level:.0%} | {r.method} | {r.coverage:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] | "
                 f"{r.shortfall_pts:+.1f} | {a.mean_width:.0f} | {a.winkler:.0f} | {a.pinball:.1f} |")
    L += ["", "Coverage at 90% by predicted regime and by test month (mean over seeds):", "",
          "| method | " + " | ".join([*REGIMES, *months]) + " |", "|---|" + "---|" * (len(REGIMES) + len(months))]
    sel = imet[imet.level == 0.9]
    for method in METHODS:
        vals = [sel[(sel.method == method) & (sel.subset == f"regime_{r}")].coverage.mean() for r in REGIMES]
        vals += [sel[(sel.method == method) & (sel.subset == f"month_{m}")].coverage.mean() for m in months]
        L.append(f"| {method} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    rows_regime = sel[(sel.method == PROPOSED) & (sel.seed == 0) & sel.subset.str.startswith("regime_")]
    L += ["", "Rows per predicted regime (seed 0): " + ", ".join(
        f"{r.subset[7:]} {r.rows}" for _, r in rows_regime.iterrows()), "",
        "### Paired day-level bootstrap, intervals (per-day Winkler)", "",
        "| level | A | B | mean diff (kW) | 95% CI | share A better | days A better |", "|---|---|---|---|---|---|---|"]
    for _, r in bi.iterrows():
        L.append(f"| {r.level:.0%} | {r.method_a} | {r.method_b} | {r.mean_diff:.0f} | [{r.ci_low:.0f}, {r.ci_high:.0f}] | "
                 f"{r.share_a_better:.3f} | {r.days_a_better}/{r.n_days} |")
    return "\n".join(L) + "\n"


def main() -> None:
    env = environment_info()
    frame, rep = preprocess_wind_file()
    (OUT / "preprocessing_report.json").write_text(json.dumps({"report": rep.to_dict(), "environment": env}, indent=2),
                                                   encoding="utf-8")
    base, preds = load()
    missing = [f"{m}__seed{s}" for m in MODELS for s in ((0,) if "persistence" in m else (0, 1, 2))
               if f"{m}__seed{s}" not in preds]
    if missing:
        raise SystemExit(f"incomplete folds, missing model files: {missing}")
    point, pday = point_tables(base, preds)
    imet, iday = interval_tables(base, preds)
    bp, bi, cov = bootstraps(pday, iday)
    for name, df in {"point_metrics": point, "point_by_day": pday, "interval_metrics": imet,
                     "interval_by_day": iday, "bootstrap_point": bp, "bootstrap_interval": bi,
                     "coverage_ci": cov}.items():
        df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    plots(point, imet, base, preds, frame)
    (OUT / "wind_summary.md").write_text(summary(env, rep.to_dict(), point, pday, imet, bp, bi, cov, base),
                                         encoding="utf-8")
    c = cov[(cov.level == 0.9) & (cov.method == PROPOSED)].iloc[0]
    print(f"wind {PROPOSED} 90% coverage: {c.coverage:.3f} [{c.ci_low:.3f}, {c.ci_high:.3f}]")


if __name__ == "__main__":
    main()
