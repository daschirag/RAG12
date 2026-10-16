"""A4: night-rule decision (val only) and prediction intervals (calib only) for both plants.

    python -m experiments.uncertainty_solar

Step 1 - night-zeroing rule, decided on validation data only. Candidates are
fixed in advance; XGBoost and Random Forest (seeds 0-2, both plants) are
refit under each rule, and the rule with the lowest mean val MAE is used.

Step 2 - intervals. Point model: XGBoost (A3 grid, chosen on val). Quantile
model: multi-quantile XGBoost (grid chosen on val pinball loss). All interval
methods are calibrated on the calib days only and scored once on test.

Outputs (results/forecasting/): a4_night_rule.{csv,md}, a4_metrics.csv,
a4_coverage_by_tod.csv, a4_coverage_by_day.csv, a4_intervals_plant{1,2}.csv,
a4_summary.md, a4_run.json, a4_interval_plant1.png, a4_coverage_tod.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backend.evaluation.provenance import environment_info  # noqa: E402
from src.backend.forecasting.features import build_features, split_by_target_day  # noqa: E402
from src.backend.forecasting.models import (  # noqa: E402
    RandomForest,
    XGBoost,
    _val_mae,
    eval_mask,
    postprocess,
)
from src.backend.forecasting.preprocess import preprocess_plant  # noqa: E402
from src.backend.forecasting.uncertainty import (  # noqa: E402
    LEVELS,
    METHODS,
    PROPOSED,
    QuantileXGBoost,
    build_intervals,
    interval_metrics,
    quantiles_for,
    time_of_day_bin,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "forecasting"
SEEDS = (0, 1, 2)
PLANTS = (1, 2)
NIGHT_RULES: dict[str, float | None] = {
    "envelope<=0.05 (A3 default)": 0.05,
    "envelope<=0 (strict)": 0.0,
    "no night rule": None,
}


def load(plant: int):
    frame = preprocess_plant(plant).frame
    return frame, split_by_target_day(build_features(frame))


# ------------------------------------------------------------ step 1: night rule

def decide_night_rule(data: dict) -> tuple[str, pd.DataFrame]:
    rows = []
    for plant, (frame, parts) in data.items():
        for rule, thr in NIGHT_RULES.items():
            for cls in (XGBoost, RandomForest):
                for seed in SEEDS:
                    model = cls(seed=seed, night_threshold=thr).fit(parts["train"], parts["val"], frame)
                    raw = model.predict(parts["val"], frame)
                    rows.append({"plant": plant, "rule": rule, "model": cls.name, "seed": seed,
                                 "val_mae": _val_mae(raw, parts["val"], thr)})
                    print(f"night rule | plant {plant} {rule:<28} {cls.name:<14} seed {seed} "
                          f"val MAE {rows[-1]['val_mae']:8.1f}")
    table = pd.DataFrame(rows)
    means = table.groupby("rule")["val_mae"].mean()
    return str(means.idxmin()), table


# --------------------------------------------------------------- step 2: intervals

def run_intervals(plant: int, frame, parts, seed: int, thr: float | None):
    train, val, calib, test = (parts[k] for k in ("train", "val", "calib", "test"))
    point = XGBoost(seed=seed, night_threshold=thr).fit(train, val, frame)
    qr = QuantileXGBoost(quantiles_for(LEVELS), seed=seed).fit(train, val)
    pc = postprocess(point.predict(calib, frame), calib, thr)
    pt = postprocess(point.predict(test, frame), test, thr)
    ints = build_intervals(pc, qr.predict(calib), calib, pt, qr.predict(test), test, LEVELS, thr)
    return ints, {"point": point.params_, "quantile": qr.params_, "quantile_val_pinball": qr.val_pinball_}


def score_rows(plant, seed, test, ints):
    m = eval_mask(test)
    y = test.y.to_numpy()
    day = test.meta["target_daytime"].to_numpy()
    tod = time_of_day_bin(test.meta["target_time"])
    date = test.meta["target_time"].dt.strftime("%Y-%m-%d").to_numpy()
    metrics, by_tod, by_day = [], [], []
    for (method, level), (lo, hi) in ints.bounds.items():
        for subset, mask in (("all", m), ("daylight", m & day)):
            metrics.append({"plant": plant, "seed": seed, "method": method, "level": level,
                            "subset": subset, **interval_metrics(y[mask], lo[mask], hi[mask], level)})
        for name in ("dawn", "midday", "dusk", "night"):
            mask = m & (tod == name)
            if mask.any():
                by_tod.append({"plant": plant, "seed": seed, "method": method, "level": level, "tod": name,
                               **interval_metrics(y[mask], lo[mask], hi[mask], level)})
        for d in sorted(set(date)):
            mask = m & day & (date == d)
            by_day.append({"plant": plant, "seed": seed, "method": method, "level": level, "day": d,
                           **interval_metrics(y[mask], lo[mask], hi[mask], level)})
    return metrics, by_tod, by_day


def _ms(s: pd.Series, d: int) -> str:
    return f"{s.mean():.{d}f} +/- {s.std(ddof=1):.{d}f}" if len(s) > 1 else f"{s.iloc[0]:.{d}f}"


def write_summary(env, chosen, rule_table, metrics, by_tod, by_day, calib_sizes) -> None:
    rule_means = rule_table.groupby(["rule"])["val_mae"].agg(["mean", "std", "count"])
    per_plant = rule_table.groupby(["plant", "rule"])["val_mae"].mean().unstack(0)
    L = [
        "# A4 uncertainty-aware forecasting (1 hour ahead)",
        "",
        f"Generated by `python -m experiments.uncertainty_solar` at base commit {env['git_commit']}.",
        "Point model XGBoost; quantile model multi-quantile XGBoost; both chosen on val. All interval methods",
        "calibrated on the calib days (06-09 to 06-12) only; test (06-13 to 06-17) scored once.",
        "Mean +/- sample std over seeds 0, 1, 2. Scored rows = observed targets (same as A3).",
        "`daylight` = target slot is day by the prior-days irradiation envelope (> 0.05).",
        "**Confidence = nominal coverage level of the interval**, not the probability that the point",
        "forecast is correct.",
        "",
        "## Night-zeroing rule (decided on val only)",
        "",
        "Mean val MAE (kW) over XGBoost + Random Forest, seeds 0-2:",
        "",
        "| rule | mean val MAE (both plants) | Plant 1 | Plant 2 |",
        "|---|---|---|---|",
    ]
    for rule in NIGHT_RULES:
        L.append(f"| {rule} | {rule_means.loc[rule, 'mean']:.1f} | {per_plant.loc[rule, 1]:.1f} | "
                 f"{per_plant.loc[rule, 2]:.1f} |")
    L += ["", f"**Chosen: {chosen}** (lowest mean val MAE; criterion fixed before the run).", ""]
    L += ["## Calibration set sizes (scored rows)", "", "| plant | daylight | night |", "|---|---|---|"]
    for plant, (nd, nn) in calib_sizes.items():
        L.append(f"| {plant} | {nd} | {nn} |")
    L.append("")
    for plant in PLANTS:
        for level in LEVELS:
            L += [f"## Plant {plant}, nominal {level:.0%}", "",
                  "| method | coverage all | coverage daylight | width all (kW) | width daylight (kW) | "
                  "Winkler all | Winkler daylight | pinball all | pinball daylight |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for method in METHODS:
                sel = metrics[(metrics.plant == plant) & (metrics.level == level) & (metrics.method == method)]
                a, d = sel[sel.subset == "all"], sel[sel.subset == "daylight"]
                tag = " (proposed)" if method == PROPOSED else ""
                L.append(f"| {method}{tag} | {_ms(a.coverage, 3)} | {_ms(d.coverage, 3)} | "
                         f"{_ms(a.mean_width, 0)} | {_ms(d.mean_width, 0)} | {_ms(a.winkler, 0)} | "
                         f"{_ms(d.winkler, 0)} | {_ms(a.pinball, 1)} | {_ms(d.pinball, 1)} |")
            L.append("")
    L += ["## Coverage by time of day of the target, nominal 90% (mean over seeds)", "",
          "Bins: dawn 05:00-08:59, midday 09:00-15:59, dusk 16:00-19:59, night otherwise.", ""]
    for plant in PLANTS:
        t = by_tod[(by_tod.plant == plant) & (by_tod.level == 0.9)]
        piv = t.groupby(["method", "tod"])["coverage"].mean().unstack()[["dawn", "midday", "dusk", "night"]]
        rows = t[t.seed == 0].groupby("tod")["rows"].first()
        L += [f"### Plant {plant} (rows: " + ", ".join(f"{k} {int(rows[k])}" for k in piv.columns) + ")", "",
              "| method | dawn | midday | dusk | night |", "|---|---|---|---|---|"]
        for method in METHODS:
            L.append(f"| {method} | " + " | ".join(f"{piv.loc[method, c]:.3f}" for c in piv.columns) + " |")
        L.append("")
    L += ["## Daylight coverage per test day, nominal 90% (mean over seeds)", ""]
    for plant in PLANTS:
        t = by_day[(by_day.plant == plant) & (by_day.level == 0.9)]
        piv = t.groupby(["method", "day"])["coverage"].mean().unstack()
        L += [f"### Plant {plant}", "", "| method | " + " | ".join(piv.columns) + " |",
              "|---|" + "---|" * len(piv.columns)]
        for method in METHODS:
            L.append(f"| {method} | " + " | ".join(f"{v:.3f}" for v in piv.loc[method]) + " |")
        L.append("")
    (RESULTS / "a4_summary.md").write_text("\n".join(L), encoding="utf-8")


def plot_interval(test, ints, path: Path) -> None:
    t = test.meta["target_time"].to_numpy()
    y = test.y.to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, method in zip(axes, ("qr_raw", PROPOSED)):
        lo, hi = ints.bounds[(method, 0.9)]
        ax.fill_between(t, lo, hi, color="0.8", step="mid", label="90% interval")
        ax.plot(t, ints.point, color="0.3", linewidth=0.8, label="point forecast")
        ax.plot(t, y, color="black", linewidth=0.6, linestyle="--", label="observed")
        miss = (y < lo) | (y > hi)
        ax.scatter(t[miss], y[miss], s=8, marker="x", color="black", label=f"outside interval (n={miss.sum()})")
        ax.set_ylabel("AC power (kW)")
        ax.set_title(f"Plant 1 test days, {method}, seed 0", fontsize=9)
        ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_tod(by_tod: pd.DataFrame, path: Path) -> None:
    bins = ["dawn", "midday", "dusk"]
    hatches = ["", "//", "..", "xx", "\\\\"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, plant in zip(axes, PLANTS):
        t = by_tod[(by_tod.plant == plant) & (by_tod.level == 0.9)]
        piv = t.groupby(["method", "tod"])["coverage"].mean().unstack()
        x = np.arange(len(bins))
        w = 0.16
        for i, method in enumerate(METHODS):
            ax.bar(x + (i - 2) * w, piv.loc[method, bins], w, color="white", edgecolor="black",
                   hatch=hatches[i], label=method)
        ax.axhline(0.9, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x, bins)
        ax.set_ylim(0.5, 1.02)
        ax.set_title(f"Plant {plant}: test coverage at nominal 90% (dashed)", fontsize=9)
    axes[0].set_ylabel("empirical coverage (y axis starts at 0.5)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHODS), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = environment_info()
    data = {p: load(p) for p in PLANTS}

    chosen, rule_table = decide_night_rule(data)
    rule_table.to_csv(RESULTS / "a4_night_rule.csv", index=False, encoding="utf-8")
    thr = NIGHT_RULES[chosen]
    print(f"chosen night rule: {chosen}")

    metrics, by_tod, by_day, params, calib_sizes = [], [], [], [], {}
    for plant, (frame, parts) in data.items():
        test = parts["test"]
        cal = parts["calib"]
        mc = eval_mask(cal)
        day_c = cal.meta["target_daytime"].to_numpy()
        calib_sizes[plant] = (int((mc & day_c).sum()), int((mc & ~day_c).sum()))
        out = pd.DataFrame({"origin": test.X.index, "target_time": test.meta["target_time"].to_numpy(),
                            "y_true": test.y.to_numpy(), "scored": eval_mask(test),
                            "daylight": test.meta["target_daytime"].to_numpy()})
        for seed in SEEDS:
            ints, p = run_intervals(plant, frame, parts, seed, thr)
            params.append({"plant": plant, "seed": seed, **p})
            mrows, trows, drows = score_rows(plant, seed, test, ints)
            metrics += mrows
            by_tod += trows
            by_day += drows
            out[f"point__seed{seed}"] = ints.point
            for (method, level), (lo, hi) in ints.bounds.items():
                out[f"{method}_{int(level * 100)}_lo__seed{seed}"] = lo
                out[f"{method}_{int(level * 100)}_hi__seed{seed}"] = hi
            if plant == 1 and seed == 0:
                plot_interval(test, ints, RESULTS / "a4_interval_plant1.png")
            cov = [r for r in mrows if r["method"] == PROPOSED and r["level"] == 0.9]
            print(f"intervals | plant {plant} seed {seed} {PROPOSED} 90%: coverage all "
                  f"{cov[0]['coverage']:.3f} daylight {cov[1]['coverage']:.3f}")
        out.to_csv(RESULTS / f"a4_intervals_plant{plant}.csv", index=False, encoding="utf-8")

    metrics, by_tod, by_day = map(pd.DataFrame, (metrics, by_tod, by_day))
    metrics.to_csv(RESULTS / "a4_metrics.csv", index=False, encoding="utf-8")
    by_tod.to_csv(RESULTS / "a4_coverage_by_tod.csv", index=False, encoding="utf-8")
    by_day.to_csv(RESULTS / "a4_coverage_by_day.csv", index=False, encoding="utf-8")
    plot_tod(by_tod, RESULTS / "a4_coverage_tod.png")
    write_summary(env, chosen, rule_table, metrics, by_tod, by_day, calib_sizes)
    (RESULTS / "a4_run.json").write_text(json.dumps(
        {"night_rule": chosen, "night_threshold": thr, "levels": LEVELS, "seeds": SEEDS,
         "calib_scored_rows": calib_sizes, "params": params, "environment": env},
        indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
