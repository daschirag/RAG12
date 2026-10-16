// Forecast: actual vs 1-h-ahead forecast with the interval band (GET /forecast/series), a single-origin trust
// check (POST /decision), the model comparison against persistence and the coverage numbers (GET /results/summary/forecasting).
import { api } from "../api.js";
import { h, fmt, loading, errorBox, setContent, trustPanel, field, select, table } from "../ui.js";
import { intervalChart, barChart } from "../charts.js";

const MODEL_NAMES = { persistence: "Persistence", daily_persistence: "Daily persistence", random_forest: "Random forest",
  xgboost: "XGBoost", xgboost_no_day_month: "XGBoost (no day/month)", lstm: "LSTM", tcn: "TCN",
  xgboost_oracle_weather: "XGBoost + measured weather (upper bound)" };

export default function mount(root) {
  const assetSel = select([], "", { id: "fc-asset" });
  const day = h("input", { type: "date", id: "fc-day" });
  const conf = select([["0.8", "80%"], ["0.9", "90%"], ["0.95", "95%"]], "0.9", { id: "fc-conf" });
  const go = h("button", { class: "btn" }, "Show day");
  const chartCard = h("div", { class: "card" }, loading("Loading assets..."));
  const sampleCard = h("div", { class: "card" });
  const compareCard = h("div", { class: "card" }, loading("Loading measured results..."));
  const covCard = h("div", { class: "card" });
  let assets = [], summary = null, lastSeries = null;

  root.append(
    h("div", { class: "page-head" }, h("div", {}, h("h2", {}, "Forecast"),
      h("p", {}, "1-hour-ahead AC power with a conformal (CQR, calibrated per group) prediction interval. The interval's level is its nominal coverage, not the probability that the forecast is right."))),
    h("div", { class: "controls" }, field("Asset", assetSel), field("Day", day), field("Interval level", conf), go),
    h("div", { class: "split" }, h("div", { class: "stack" }, chartCard, sampleCard), h("div", { class: "stack" }, compareCard, covCard)));

  go.addEventListener("click", loadSeries);
  assetSel.addEventListener("change", () => { setDayRange(); renderSide(); loadSeries(); });
  conf.addEventListener("change", () => { renderSide(); loadSeries(); });

  api.get("/forecast/assets").then((a) => {
    assets = a;
    setContent(assetSel, a.map((x) => h("option", { value: x.asset }, `${x.asset} (${x.kind}, ${x.step_minutes}-min grid)`)));
    setDayRange(true);
    loadSeries();
  }).catch((e) => setContent(chartCard, errorBox(e, () => location.reload())));
  api.get("/results/summary/forecasting").then((s) => { summary = s; renderSide(); })
    .catch((e) => setContent(compareCard, errorBox(e)));

  const cur = () => assets.find((x) => x.asset === assetSel.value);
  function setDayRange(init = false) {
    const a = cur();
    if (!a) return;
    day.min = a.first_origin.slice(0, 10);
    day.max = a.last_origin.slice(0, 10);
    if (init || day.value < day.min || day.value > day.max) {
      const d = new Date(`${a.last_origin.slice(0, 10)}T00:00:00`);
      d.setDate(d.getDate() - 1);
      day.value = d.toISOString().slice(0, 10);
    }
  }

  async function loadSeries() {
    const a = cur();
    if (!a || !day.value) return;
    setContent(chartCard, loading(`Forecasting every ${a.step_minutes}-min origin of ${day.value}...`));
    go.disabled = true;
    try {
      const r = await api.get(`/forecast/series?asset=${a.asset}&start=${day.value}T00:00&end=${day.value}T23:59&confidence=${conf.value}`);
      lastSeries = r;
      const pts = r.points;
      const inS = pts.filter((p) => p.in_sample).length;
      const obs = pts.filter((p) => p.observed_kw !== null);
      const mae = obs.length ? obs.reduce((acc, p) => acc + Math.abs(p.prediction - p.observed_kw), 0) / obs.length : null;
      setContent(chartCard,
        h("h3", {}, `${a.asset}: ${day.value}`),
        intervalChart(pts, { level: Number(conf.value), label: `${a.asset} ${day.value}` }),
        h("div", { class: "kpis", style: "margin-top:.75rem" },
          kpi(fmt.p(r.coverage_observed), `observed inside the ${Math.round(conf.value * 100)}% interval (this day, ${obs.length} points)`),
          kpi(mae === null ? "-" : `${fmt.n(mae, 0)} kW`, "mean absolute error (this day)"),
          kpi(`${inS}/${pts.length}`, "points in-sample")),
        inS ? h("div", { class: "callout", style: "margin-top:.75rem" }, h("strong", {}, "in_sample = true. "),
          `The deployed ${a.asset} model was fit on ${a.segments.train[0]} to ${a.segments.calib[1]} (train ${a.segments.train.join(" to ")}, ` +
          `validation ${a.segments.val.join(" to ")}, calibration ${a.segments.calib.join(" to ")}), so this day was seen during fitting and ` +
          "the accuracy above is optimistic. Out-of-sample accuracy is the rolling-origin result on the right.") : null);
      renderSample(pts);
    } catch (e) {
      setContent(chartCard, errorBox(e, loadSeries));
    } finally {
      go.disabled = false;
    }
  }

  function renderSample(pts) {
    const times = pts.map((p) => p.origin);
    const noon = times.find((t) => t.slice(11, 16) === "11:00") || times[Math.floor(times.length / 2)];
    const tsel = select(times.map((t) => [t, `origin ${fmt.dt(t).slice(11)} -> target +1 h`]), noon);
    const out = h("div");
    const btn = h("button", { class: "btn btn-ghost" }, "Check with the decision engine");
    btn.addEventListener("click", async () => {
      setContent(out, loading("POST /decision ..."));
      try {
        const d = await api.post("/decision", { kind: "forecast", forecast: { asset: assetSel.value, origin: tsel.value.slice(0, 16), confidence: Number(conf.value) } });
        const o = d.output;
        setContent(out, h("p", { class: "small" }, `Forecast ${fmt.n(o.prediction, 0)} kW for ${fmt.dt(o.target_time)}, interval ${fmt.n(o.lower_bound, 0)} - ${fmt.n(o.upper_bound, 0)} kW `,
          `(${o.calibration_group}), observed ${o.observed_kw === null ? "n/a" : `${fmt.n(o.observed_kw, 0)} kW`}, in_sample ${o.in_sample}.`),
          trustPanel(d),
          h("p", { class: "small muted" }, "Served forecasts carry no live telemetry check (the C3 detector decides per whole day), so S = 0 and the best possible decision is REVIEW. The Security lab shows the telemetry gate."));
      } catch (e) {
        setContent(out, errorBox(e));
      }
    });
    setContent(sampleCard, h("h3", {}, "One forecast through the trust engine"), h("div", { class: "controls" }, field("Origin", tsel), btn), out);
  }

  function renderSide() {
    if (!summary || !cur()) return;
    const a = cur(), m = summary.metrics;
    let rows, title, note;
    if (a.kind === "wind") {
      rows = m.wind_rolling_origin;
      title = "Wind T1: rolling origin, Sep-Dec 2018 (out-of-sample)";
    } else {
      rows = m.solar_fixed_split.filter((r) => r.plant === a.plant);
      title = `Solar plant ${a.plant}: A5 fixed test split (out-of-sample)`;
    }
    const pers = rows.find((r) => r.model === "persistence");
    const bars = rows.filter((r) => !r.model.includes("oracle")).sort((x, y) => x.mae - y.mae)
      .map((r) => ({ label: MODEL_NAMES[r.model] || r.model, value: r.mae, extra: [["RMSE kW", fmt.n(r.rmse, 1)], ["R2", fmt.n(r.r2, 3)], ["seeds", r.seeds]] }));
    const beats = rows.filter((r) => !["persistence", "daily_persistence"].includes(r.model) && !r.model.includes("oracle") && r.mae < pers.mae).length;
    note = beats === 0 ? h("div", { class: "callout bad" }, h("strong", {}, "Persistence wins on MAE here. "), `No learned model beats persistence (${fmt.n(pers.mae, 1)} kW).`)
      : h("p", { class: "small muted" }, `${beats} learned models beat persistence (MAE ${fmt.n(pers.mae, 1)} kW) on this split.`);
    setContent(compareCard, h("h3", {}, "Model comparison vs persistence"), h("p", { class: "small muted" }, `${title}; MAE in kW, mean over seeds.`),
      barChart(bars, { valueLabel: "MAE kW", baseline: "Persistence" }), note,
      h("p", { class: "small muted" }, "Source: ", summary.sources.filter((x) => x.includes("a5_fixed") || x.includes("point_metrics")).join(", ")));
    const cov = m.coverage.filter((c) => c.asset === a.asset && Math.abs(c.level - Number(conf.value)) < 1e-9);
    setContent(covCard, h("h3", {}, `Measured coverage at ${Math.round(conf.value * 100)}%`),
      h("p", { class: "small muted" }, `Rolling origin, ${cov[0]?.days ?? "?"} test days${a.kind === "wind" ? "" : ", daylight rows"} (out-of-sample).`),
      table([{ label: "method", key: "method" }, { label: "coverage", num: true, render: (c) => fmt.n(c.coverage, 3) },
        { label: "95% CI", num: true, render: (c) => fmt.ci(c.ci_low, c.ci_high) }], cov, { caption: "Coverage by interval method" }),
      h("p", { class: "small muted" }, "Deployed method: cqr_mondrian. Sources: results/forecasting/a5_coverage_ci.csv, results/forecasting/wind/coverage_ci.csv."));
  }
}

function kpi(v, l) {
  return h("div", { class: "kpi" }, h("div", { class: "v" }, v), h("div", { class: "l" }, l));
}
