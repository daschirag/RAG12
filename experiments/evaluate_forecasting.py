"""A5: evaluation report for Track A (fixed split + rolling origin + bootstrap).

    python -m experiments.evaluate_forecasting

Inputs (produced earlier, strict night rule):
- results/forecasting/a3_predictions_plant{1,2}.csv, a4_metrics.csv   (fixed 5-day test split)
- results/forecasting/a5_folds/plant{p}_{day}.csv                      (rolling origin, 12 days)

Outputs (results/forecasting/): a5_summary.md, a5_fixed_split.csv,
a5_rolling_point_by_day.csv, a5_rolling_interval_by_day.csv,
a5_bootstrap_point.csv, a5_bootstrap_interval.csv, a5_coverage_ci.csv,
a5_error_by_hour.csv, a5_error_vs_clearness.csv, and grayscale figures
a5_rolling_mae.png, a5_rolling_coverage.png, a5_bootstrap_interval.png,
a5_error_by_hour.png, a5_error_vs_clearness.png.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backend.evaluation.provenance import environment_info  # noqa: E402
from src.backend.forecasting.evaluation import (  # noqa: E402
    daily_clearness,
    day_bootstrap_ratio,
    paired_day_bootstrap,
)
from src.backend.forecasting.features import irradiation_envelope  # noqa: E402
from src.backend.forecasting.metrics import mae, mape, r2, rmse  # noqa: E402
from src.backend.forecasting.models import PLANT_CAPACITY_KW  # noqa: E402
from src.backend.forecasting.preprocess import preprocess_plant  # noqa: E402
from src.backend.forecasting.uncertainty import (  # noqa: E402
    LEVELS,
    METHODS,
    PROPOSED,
    interval_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "forecasting"
FOLD_DIR = RESULTS / "a5_folds"
PLANTS = (1, 2)
MAPE_MIN_KW = 0.05 * PLANT_CAPACITY_KW          # 1650 kW: MAPE only where y >= 5% of capacity
POINT_ORDER = ("persistence", "daily_persistence", "random_forest", "xgboost",
               "xgboost_no_day_month", "lstm", "tcn", "xgboost_oracle_weather")
LEARNED = ("random_forest", "xgboost", "lstm", "tcn")
N_BOOT = 10_000


def _seed_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if c.startswith(prefix + "__seed")]


def _ms(values, d: int) -> str:
    v = pd.Series(values, dtype=float)
    return f"{v.mean():.{d}f} +/- {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v.iloc[0]:.{d}f}"


# ----------------------------------------------------------------- fixed split

def fixed_split_table() -> pd.DataFrame:
    rows = []
    for plant in PLANTS:
        df = pd.read_csv(RESULTS / f"a3_predictions_plant{plant}.csv")
        test = df[(df.split == "test") & df.scored]
        y = test.y_true.to_numpy()
        for model in POINT_ORDER:
            for col in _seed_cols(test, model):
                p = test[col].to_numpy()
                rows.append({"plant": plant, "model": model, "seed": col.split("seed")[-1],
                             "mae": mae(y, p), "rmse": rmse(y, p), "r2": r2(y, p),
                             "mape_pct": mape(y, p, MAPE_MIN_KW), "mape_rows": int((y >= MAPE_MIN_KW).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- rolling origin

def load_folds(plant: int) -> pd.DataFrame:
    files = sorted(FOLD_DIR.glob(f"plant{plant}_*.csv"))
    if not files:
        raise FileNotFoundError(f"no rolling-origin folds for plant {plant}; run experiments.rolling_origin")
    frames = [pd.read_csv(f, parse_dates=["origin", "target_time"]) for f in files]
    days = np.repeat([f.stem.split("_", 1)[1] for f in files], [len(df) for df in frames])
    # Label after concatenation: inserting into each wide per-fold frame fragments it.
    out = pd.concat(frames, ignore_index=True)
    return pd.concat([out, pd.Series(days, name="test_day", index=out.index)], axis=1)


def rolling_point_by_day(folds: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for plant, df in folds.items():
        for day, g in df.groupby("test_day"):
            s = g[g.scored]
            dl = s[s.daylight]
            for model in POINT_ORDER:
                cols = _seed_cols(s, model)
                for col in cols:
                    rows.append({"plant": plant, "test_day": day, "model": model,
                                 "seed": int(col.split("seed")[-1]), "rows": len(s),
                                 "mae": mae(s.y_true, s[col]), "daylight_mae": mae(dl.y_true, dl[col]),
                                 "sq_err_sum": float(((s.y_true - s[col]) ** 2).sum())})
    return pd.DataFrame(rows)


def rolling_interval_by_day(folds: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for plant, df in folds.items():
        for day, g in df.groupby("test_day"):
            s = g[g.scored & g.daylight]
            for method, level in itertools.product(METHODS, LEVELS):
                tag = f"{method}_{int(level * 100)}"
                for lo_col in _seed_cols(s, tag + "_lo"):
                    seed = int(lo_col.split("seed")[-1])
                    hi_col = f"{tag}_hi__seed{seed}"
                    m = interval_metrics(s.y_true.to_numpy(), s[lo_col].to_numpy(), s[hi_col].to_numpy(), level)
                    rows.append({"plant": plant, "test_day": day, "method": method, "level": level,
                                 "seed": seed, "covered": m["coverage"] * m["rows"], **m})
    return pd.DataFrame(rows)


def bootstrap_point(by_day: pd.DataFrame) -> pd.DataFrame:
    loss = by_day.groupby(["plant", "model", "test_day"])["mae"].mean()
    rows = []
    for plant in PLANTS:
        models = [m for m in POINT_ORDER if (plant, m) in loss.index.droplevel(2)]
        for a, b in itertools.combinations(models, 2):
            res = paired_day_bootstrap(loss.loc[(plant, a)], loss.loc[(plant, b)], N_BOOT, seed=0)
            rows.append({"plant": plant, "model_a": a, "model_b": b, **res.as_dict()})
    return pd.DataFrame(rows)


def bootstrap_interval(by_day: pd.DataFrame) -> pd.DataFrame:
    loss = by_day.groupby(["plant", "level", "method", "test_day"])["winkler"].mean()
    rows = []
    for plant, level in itertools.product(PLANTS, LEVELS):
        for other in METHODS:
            if other == PROPOSED:
                continue
            res = paired_day_bootstrap(loss.loc[(plant, level, PROPOSED)], loss.loc[(plant, level, other)],
                                       N_BOOT, seed=0)
            rows.append({"plant": plant, "level": level, "method_a": PROPOSED, "method_b": other,
                         "statistic": "daylight Winkler per day", **res.as_dict()})
    return pd.DataFrame(rows)


def coverage_ci(by_day: pd.DataFrame) -> pd.DataFrame:
    g = by_day.groupby(["plant", "level", "method", "test_day"])[["covered", "rows"]].mean()
    rows = []
    for (plant, level, method), sub in g.groupby(level=[0, 1, 2]):
        sub = sub.droplevel([0, 1, 2])
        point, lo, hi = day_bootstrap_ratio(sub["covered"], sub["rows"], N_BOOT, seed=0)
        rows.append({"plant": plant, "level": level, "method": method, "coverage": point,
                     "ci_low": lo, "ci_high": hi, "shortfall_pts": 100 * (level - point),
                     "days": len(sub), "rows": int(sub["rows"].sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- error analysis

def error_by_hour(folds: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for plant, df in folds.items():
        s = df[df.scored].copy()
        s["hour"] = s.target_time.dt.hour
        for model in ("persistence", "daily_persistence", *LEARNED):
            pred = s[_seed_cols(s, model)].mean(axis=1)
            err = (s.y_true - pred).abs()
            for hour, e in err.groupby(s.hour):
                rows.append({"plant": plant, "model": model, "hour": hour, "mae": float(e.mean()),
                             "rows": int(len(e))})
    return pd.DataFrame(rows)


def error_vs_clearness(by_day: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for plant in PLANTS:
        frame = preprocess_plant(plant).frame
        clear = daily_clearness(frame, irradiation_envelope(frame["irradiation"]))
        clear.index = clear.index.strftime("%Y-%m-%d")
        daymae = by_day[by_day.plant == plant].groupby(["model", "test_day"])["daylight_mae"].mean()
        for model in ("persistence", *LEARNED):
            for day, v in daymae.loc[model].items():
                rows.append({"plant": plant, "model": model, "test_day": day,
                             "daylight_mae": v, "clearness": float(clear.get(day, np.nan))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- figures

STYLES = {"persistence": ("black", ":", "o"), "daily_persistence": ("0.5", ":", "s"),
          "random_forest": ("black", "-", "^"), "xgboost": ("0.35", "-", "v"),
          "lstm": ("black", "--", "D"), "tcn": ("0.35", "--", "x")}


def plot_rolling_mae(by_day: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, plant in zip(axes, PLANTS):
        d = by_day[by_day.plant == plant].groupby(["model", "test_day"])["mae"].mean()
        for model, (c, ls, mk) in STYLES.items():
            s = d.loc[model]
            ax.plot(pd.to_datetime(s.index), s.values, color=c, linestyle=ls, marker=mk, markersize=4,
                    linewidth=1, label=model)
        ax.set_title(f"Plant {plant}: rolling-origin test-day MAE (mean over seeds)", fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0].set_ylabel("MAE (kW), scored rows of the day")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rolling_coverage(by_day: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    styles = {"qr_raw": ("0.55", "-", "o"), "gaussian": ("0.7", ":", "s"), "split_conformal": ("0.7", "--", "^"),
              "cqr_pooled": ("0.4", ":", "v"), "cqr_mondrian": ("black", "-", "D")}
    for ax, plant in zip(axes, PLANTS):
        d = by_day[(by_day.plant == plant) & (by_day.level == 0.9)].groupby(["method", "test_day"])["coverage"].mean()
        for method, (c, ls, mk) in styles.items():
            s = d.loc[method]
            ax.plot(pd.to_datetime(s.index), s.values, color=c, linestyle=ls, marker=mk, markersize=4,
                    linewidth=1.4 if method == PROPOSED else 0.9, label=method)
        ax.axhline(0.9, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"Plant {plant}: daylight coverage per test day, nominal 90% (dashed)", fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    axes[0].set_ylabel("empirical coverage")
    axes[1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bootstrap_interval(boot: pd.DataFrame, path: Path) -> None:
    b = boot[boot.level == 0.9]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, plant in zip(axes, PLANTS):
        s = b[b.plant == plant].reset_index(drop=True)
        yy = np.arange(len(s))
        ax.errorbar(s.mean_diff, yy, xerr=[s.mean_diff - s.ci_low, s.ci_high - s.mean_diff], fmt="o",
                    color="black", capsize=3)
        ax.axvline(0, color="0.5", linewidth=0.8)
        ax.set_yticks(yy, [f"{PROPOSED} - {m}" for m in s.method_b], fontsize=7)
        ax.set_title(f"Plant {plant}: daily daylight Winkler difference, 90% (95% CI)", fontsize=8)
        ax.set_xlabel("kW (negative = cqr_mondrian better)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_error_by_hour(eh: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, plant in zip(axes, PLANTS):
        d = eh[eh.plant == plant]
        for model, (c, ls, mk) in STYLES.items():
            s = d[d.model == model]
            ax.plot(s.hour, s.mae, color=c, linestyle=ls, marker=mk, markersize=3, linewidth=1, label=model)
        ax.set_xlabel("hour of target time")
        ax.set_title(f"Plant {plant}: MAE by hour, 12 rolling-origin test days", fontsize=9)
    axes[0].set_ylabel("MAE (kW)")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_error_vs_clearness(ec: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, plant in zip(axes, PLANTS):
        d = ec[(ec.plant == plant) & (ec.model == "xgboost")]
        ax.scatter(d.clearness, d.daylight_mae, color="black", s=18)
        rho = d[["clearness", "daylight_mae"]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"Plant {plant}: XGBoost daylight MAE per test day (Spearman {rho:.2f})", fontsize=8)
        ax.set_xlabel("daily clearness (irradiation / prior-days envelope)")
    axes[0].set_ylabel("daylight MAE (kW)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- summary

def write_summary(env, fixed, a4, pday, iday, bpoint, bint, cov, eh, ec) -> None:
    L = [
        "# A5 forecasting evaluation (1 hour ahead, strict night rule everywhere)",
        "",
        f"Generated by `python -m experiments.evaluate_forecasting` at base commit {env['git_commit']}.",
        "Point models: mean +/- sample std over seeds 0-2. MAPE only on rows with y >= "
        f"{MAPE_MIN_KW:.0f} kW (5% of capacity), because MAPE is undefined near zero.",
        "",
        "## 1. Fixed split (test 2020-06-13 .. 06-17, A3 re-run with the strict night rule)",
        "",
    ]
    for plant in PLANTS:
        f = fixed[fixed.plant == plant]
        L += [f"### Plant {plant} (MAPE rows: {int(f.mape_rows.iloc[0])})", "",
              "| model | MAE (kW) | RMSE (kW) | MAPE (%) | R2 |", "|---|---|---|---|---|"]
        for model in POINT_ORDER:
            s = f[f.model == model]
            note = " (upper bound, not deployable)" if model == "xgboost_oracle_weather" else ""
            L.append(f"| {model}{note} | {_ms(s.mae, 1)} | {_ms(s.rmse, 1)} | {_ms(s.mape_pct, 1)} | {_ms(s.r2, 4)} |")
        L.append("")
    L += ["Intervals on the fixed split (A4 re-run), nominal 90%, daylight rows:", "",
          "| plant | method | coverage | width (kW) | Winkler |", "|---|---|---|---|---|"]
    for plant in PLANTS:
        for method in METHODS:
            s = a4[(a4.plant == plant) & (a4.method == method) & (a4.level == 0.9) & (a4.subset == "daylight")]
            L.append(f"| {plant} | {method} | {_ms(s.coverage, 3)} | {_ms(s.mean_width, 0)} | {_ms(s.winkler, 0)} |")
    L.append("")

    L += ["## 2. Rolling origin (12 test days, 2020-06-06 .. 06-17; each model refit per day)", "",
          "Per-model MAE over the 12 test days: pooled over all scored rows (mean +/- std over seeds), and the",
          "mean / min / max of the per-day MAE (seed-averaged).", ""]
    for plant in PLANTS:
        p = pday[pday.plant == plant]
        L += [f"### Plant {plant}", "",
              "| model | pooled MAE (kW) | pooled RMSE (kW) | per-day MAE mean | min | max | days best |",
              "|---|---|---|---|---|---|---|"]
        perday = p.groupby(["model", "test_day"])["mae"].mean().unstack(0)
        comparable = [m for m in POINT_ORDER if m in perday.columns and m != "xgboost_no_day_month"]
        best_counts = perday[comparable].idxmin(axis=1).value_counts()
        for model in POINT_ORDER:
            s = p[p.model == model]
            if s.empty:
                continue
            per_seed = s.groupby("seed").apply(
                lambda g: pd.Series({"mae": (g.mae * g.rows).sum() / g.rows.sum(),
                                     "rmse": np.sqrt(g.sq_err_sum.sum() / g.rows.sum())}), include_groups=False)
            d = perday[model]
            L.append(f"| {model} | {_ms(per_seed.mae, 1)} | {_ms(per_seed.rmse, 1)} | {d.mean():.1f} | "
                     f"{d.min():.1f} | {d.max():.1f} | {int(best_counts.get(model, 0))} |")
        L.append("")
    L += ["`days best` counts days on which the model had the lowest seed-averaged MAE among the six main",
          "models (the no-day/month ablation is excluded from the count).", ""]

    L += ["### Paired day-level bootstrap, point models (per-day MAE, 10,000 resamples, seed 0)", "",
          "mean diff = MAE(A) - MAE(B) averaged over days; negative means A is better.", "",
          "| plant | A | B | mean diff (kW) | 95% CI | share of resamples A better | days A better |",
          "|---|---|---|---|---|---|---|"]
    shown = {"persistence", "daily_persistence", *LEARNED}
    baselines = {"persistence", "daily_persistence"}
    for _, r in bpoint.iterrows():
        if r.model_a in shown and r.model_b in shown:
            # Show the learned model as A against a baseline (negative diff = learned model better).
            flip = r.model_a in baselines and r.model_b not in baselines
            a, b = (r.model_b, r.model_a) if flip else (r.model_a, r.model_b)
            sign = -1 if flip else 1
            md, lo, hi = sign * r.mean_diff, *(sorted([sign * r.ci_low, sign * r.ci_high]))
            share = r.share_a_better if sign == 1 else 1 - r.share_a_better
            days = r.days_a_better if sign == 1 else r.n_days - r.days_a_better
            L.append(f"| {r.plant} | {a} | {b} | {md:.1f} | [{lo:.1f}, {hi:.1f}] | {share:.3f} | {days}/{r.n_days} |")
    L.append("")

    L += ["### Intervals, rolling origin: daylight coverage pooled over 12 days (95% day-bootstrap CI)", "",
          "| plant | level | method | coverage | 95% CI | shortfall vs nominal (points) | mean width (kW) | mean Winkler |",
          "|---|---|---|---|---|---|---|---|"]
    agg = iday.groupby(["plant", "level", "method"])[["mean_width", "winkler"]].mean()
    for _, r in cov.sort_values(["plant", "level"]).iterrows():
        a = agg.loc[(r.plant, r.level, r.method)]
        L.append(f"| {r.plant} | {r.level:.0%} | {r.method} | {r.coverage:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] | "
                 f"{r.shortfall_pts:+.1f} | {a.mean_width:.0f} | {a.winkler:.0f} |")
    L += ["", "Shortfall = nominal minus achieved coverage, in percentage points (positive = under-coverage).", ""]

    L += ["### Per-fold daylight coverage and Winkler at 90% (seed-averaged)", ""]
    for plant in PLANTS:
        d = iday[(iday.plant == plant) & (iday.level == 0.9)].groupby(["method", "test_day"])[["coverage", "winkler"]].mean()
        days = sorted(d.index.get_level_values(1).unique())
        L += [f"#### Plant {plant}: coverage", "", "| method | " + " | ".join(x[5:] for x in days) + " |",
              "|---|" + "---|" * len(days)]
        for method in METHODS:
            L.append(f"| {method} | " + " | ".join(f"{d.loc[(method, x), 'coverage']:.2f}" for x in days) + " |")
        L += ["", f"#### Plant {plant}: Winkler (kW)", "", "| method | " + " | ".join(x[5:] for x in days) + " |",
              "|---|" + "---|" * len(days)]
        for method in METHODS:
            L.append(f"| {method} | " + " | ".join(f"{d.loc[(method, x), 'winkler']:.0f}" for x in days) + " |")
        L.append("")

    L += ["### Paired day-level bootstrap, interval methods (per-day daylight Winkler)", "",
          "| plant | level | A | B | mean diff (kW) | 95% CI | share A better | days A better |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in bint.iterrows():
        L.append(f"| {r.plant} | {r.level:.0%} | {r.method_a} | {r.method_b} | {r.mean_diff:.0f} | "
                 f"[{r.ci_low:.0f}, {r.ci_high:.0f}] | {r.share_a_better:.3f} | {r.days_a_better}/{r.n_days} |")
    L.append("")

    L += ["## 3. Error analysis (rolling origin)", "", "MAE by hour of the target time (seed-averaged predictions):", ""]
    for plant in PLANTS:
        e = eh[eh.plant == plant].pivot(index="hour", columns="model", values="mae")
        e = e[[m for m in ("persistence", "daily_persistence", *LEARNED) if m in e.columns]]
        e = e[(e > 0).any(axis=1)]
        L += [f"### Plant {plant}", "", "| hour | " + " | ".join(e.columns) + " |", "|---|" + "---|" * len(e.columns)]
        for hour, r in e.iterrows():
            L.append(f"| {hour:02d} | " + " | ".join(f"{v:.0f}" for v in r) + " |")
        L.append("")
    L += ["Daylight MAE per test day vs daily clearness (Spearman correlation over the 12 days):", "",
          "| plant | model | Spearman rho | cloudiest day (clearness) | its daylight MAE | clearest day | its daylight MAE |",
          "|---|---|---|---|---|---|---|"]
    for plant in PLANTS:
        for model in ("persistence", *LEARNED):
            d = ec[(ec.plant == plant) & (ec.model == model)].dropna()
            rho = d[["clearness", "daylight_mae"]].corr(method="spearman").iloc[0, 1]
            lo, hi = d.loc[d.clearness.idxmin()], d.loc[d.clearness.idxmax()]
            L.append(f"| {plant} | {model} | {rho:.2f} | {lo.test_day} ({lo.clearness:.2f}) | {lo.daylight_mae:.0f} | "
                     f"{hi.test_day} ({hi.clearness:.2f}) | {hi.daylight_mae:.0f} |")
    L.append("")
    (RESULTS / "a5_summary.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    env = environment_info()
    fixed = fixed_split_table()
    fixed.to_csv(RESULTS / "a5_fixed_split.csv", index=False, encoding="utf-8")
    a4 = pd.read_csv(RESULTS / "a4_metrics.csv")
    folds = {p: load_folds(p) for p in PLANTS}
    pday = rolling_point_by_day(folds)
    iday = rolling_interval_by_day(folds)
    bpoint, bint, cov = bootstrap_point(pday), bootstrap_interval(iday), coverage_ci(iday)
    eh, ec = error_by_hour(folds), error_vs_clearness(pday)
    for name, df in {"a5_rolling_point_by_day": pday, "a5_rolling_interval_by_day": iday,
                     "a5_bootstrap_point": bpoint, "a5_bootstrap_interval": bint, "a5_coverage_ci": cov,
                     "a5_error_by_hour": eh, "a5_error_vs_clearness": ec}.items():
        df.to_csv(RESULTS / f"{name}.csv", index=False, encoding="utf-8")
    plot_rolling_mae(pday, RESULTS / "a5_rolling_mae.png")
    plot_rolling_coverage(iday, RESULTS / "a5_rolling_coverage.png")
    plot_bootstrap_interval(bint, RESULTS / "a5_bootstrap_interval.png")
    plot_error_by_hour(eh, RESULTS / "a5_error_by_hour.png")
    plot_error_vs_clearness(ec, RESULTS / "a5_error_vs_clearness.png")
    write_summary(env, fixed, a4, pday, iday, bpoint, bint, cov, eh, ec)
    c = cov[(cov.plant == 1) & (cov.level == 0.9) & (cov.method == PROPOSED)].iloc[0]
    print(f"plant 1 {PROPOSED} 90% daylight coverage (rolling): {c.coverage:.3f} "
          f"[{c.ci_low:.3f}, {c.ci_high:.3f}], shortfall {c.shortfall_pts:+.1f} points")


if __name__ == "__main__":
    main()
