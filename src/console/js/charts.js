// Small SVG charts: one y-axis, recessive grid, 2px lines, crosshair tooltip, and a data-table toggle.
import { h, s, fmt } from "./ui.js";

const tip = () => document.getElementById("tooltip");
function showTip(evt, lines) {
  const t = tip();
  t.replaceChildren(...lines.map(([label, value]) => h("div", {}, value !== undefined ? h("strong", {}, value) : null, value !== undefined ? " " : null, label)));
  t.hidden = false;
  const x = Math.min(evt.clientX + 14, window.innerWidth - t.offsetWidth - 8);
  const y = Math.min(evt.clientY + 14, window.innerHeight - t.offsetHeight - 8);
  t.style.left = `${x}px`;
  t.style.top = `${y}px`;
}
export const hideTip = () => { tip().hidden = true; };

function niceMax(v) {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].map((m) => m * p).find((m) => m >= v);
}

function dataTable(columns, rows) {
  return h("details", { class: "small" }, h("summary", {}, "Show data table"),
    h("div", { class: "table-wrap", style: "max-height:260px;overflow:auto" }, h("table", {},
      h("thead", {}, h("tr", {}, columns.map((c) => h("th", { scope: "col" }, c[0])))),
      h("tbody", {}, rows.map((r) => h("tr", {}, columns.map((c) => h("td", { class: "num" }, c[1](r)))))))));
}

// Forecast vs observed with the prediction-interval band. points: {target_time, prediction, lower_bound, upper_bound, observed_kw}
export function intervalChart(points, { unit = "kW", label = "Forecast", level = 0.9 } = {}) {
  const W = 760, H = 310, m = { l: 56, r: 16, t: 24, b: 30 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const n = points.length;
  const ymax = niceMax(Math.max(1, ...points.flatMap((p) => [p.upper_bound, p.observed_kw ?? 0])));
  const x = (i) => m.l + (n <= 1 ? iw / 2 : (i * iw) / (n - 1));
  const y = (v) => m.t + ih - (v / ymax) * ih;
  const band = points.map((p, i) => `${x(i)},${y(p.upper_bound)}`).join(" ") + " " +
    points.map((p, i) => `${x(n - 1 - i)},${y(points[n - 1 - i].lower_bound)}`).join(" ");
  const pred = points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.prediction)}`).join("");
  let obsPath = "";
  points.forEach((p, i) => {
    if (p.observed_kw === null || p.observed_kw === undefined) return;
    const prev = i > 0 && points[i - 1].observed_kw !== null && points[i - 1].observed_kw !== undefined;
    obsPath += `${prev ? "L" : "M"}${x(i)},${y(p.observed_kw)}`;
  });
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * ymax);
  const xt = [0, Math.floor(n / 4), Math.floor(n / 2), Math.floor((3 * n) / 4), n - 1].filter((v, i, a) => v >= 0 && a.indexOf(v) === i);
  const cross = s("line", { class: "crosshair", y1: m.t, y2: m.t + ih, x1: -10, x2: -10, visibility: "hidden" });
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `${label}: ${n} points; prediction line, ${Math.round(level * 100)}% interval band and observed values. Use the data table for exact values.` },
    ticks.map((v) => s("line", { class: "gridline", x1: m.l, x2: m.l + iw, y1: y(v), y2: y(v) })),
    ticks.map((v) => s("text", { class: "tick", x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, fmt.n(v, 0))),
    xt.map((i) => s("text", { class: "tick", x: x(i), y: H - 8, "text-anchor": "middle" }, fmt.dt(points[i].target_time).slice(11))),
    s("line", { class: "axis", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }),
    s("text", { class: "tick", x: m.l - 6, y: 12, "text-anchor": "end" }, unit),
    s("polygon", { class: "band", points: band }),
    s("path", { class: "line-1", d: pred }),
    obsPath ? s("path", { class: "line-2", d: obsPath, "stroke-dasharray": "1 0" }) : null,
    cross);
  const hit = s("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
  hit.addEventListener("pointermove", (evt) => {
    const r = svg.getBoundingClientRect();
    const px = ((evt.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / iw) * (n - 1))));
    const p = points[i];
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    const inside = p.observed_kw !== null && p.observed_kw !== undefined ? (p.observed_kw >= p.lower_bound && p.observed_kw <= p.upper_bound ? "inside" : "outside") : "";
    showTip(evt, [[`target ${fmt.dt(p.target_time)}`], [`${unit} forecast`, fmt.n(p.prediction, 0)],
      [`${unit} interval`, `${fmt.n(p.lower_bound, 0)} - ${fmt.n(p.upper_bound, 0)}`],
      [`${unit} observed ${inside ? `(${inside})` : ""}`, p.observed_kw === null || p.observed_kw === undefined ? "n/a" : fmt.n(p.observed_kw, 0)]]);
  });
  hit.addEventListener("pointerleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); });
  svg.append(hit);
  return h("div", { class: "chart" },
    h("div", { class: "legend" },
      h("span", {}, h("span", { class: "sw", style: "background:var(--series-1);height:3px" }), "Forecast (1 h ahead)"),
      h("span", {}, h("span", { class: "sw", style: "background:var(--band)" }), `${Math.round(level * 100)}% prediction interval`),
      h("span", {}, h("span", { class: "sw", style: "background:var(--series-2);height:3px" }), "Observed")),
    svg,
    dataTable([["target", (p) => fmt.dt(p.target_time)], ["forecast", (p) => fmt.n(p.prediction, 0)],
      ["lower", (p) => fmt.n(p.lower_bound, 0)], ["upper", (p) => fmt.n(p.upper_bound, 0)],
      ["observed", (p) => (p.observed_kw === null ? "-" : fmt.n(p.observed_kw, 0))]], points));
}

// Two series over one day (clean vs attacked telemetry).
export function twoLineChart(times, a, b, { labelA = "clean", labelB = "attacked", unit = "" } = {}) {
  const pts = times.map((t, i) => ({ t, a: a[i], b: b[i] }));
  const W = 760, H = 220, m = { l: 56, r: 16, t: 10, b: 26 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b, n = pts.length;
  const vals = pts.flatMap((p) => [p.a, p.b]).filter((v) => v !== null && v !== undefined);
  const lo = Math.min(0, ...vals), hi = niceMax(Math.max(...vals, 1));
  const x = (i) => m.l + (n <= 1 ? 0 : (i * iw) / (n - 1));
  const y = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  const path = (k) => pts.reduce((acc, p, i) => (p[k] === null || p[k] === undefined ? acc : acc + `${acc && pts[i - 1]?.[k] !== null && pts[i - 1]?.[k] !== undefined ? "L" : "M"}${x(i)},${y(p[k])}`), "");
  const ticks = [0, 0.5, 1].map((f) => lo + f * (hi - lo));
  const cross = s("line", { class: "crosshair", y1: m.t, y2: m.t + ih, x1: -10, x2: -10, visibility: "hidden" });
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${labelA} vs ${labelB} ${unit}; see the data table` },
    ticks.map((v) => s("line", { class: "gridline", x1: m.l, x2: m.l + iw, y1: y(v), y2: y(v) })),
    ticks.map((v) => s("text", { class: "tick", x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, v === 0 ? "0" : fmt.n(v, Math.abs(v) < 10 ? 1 : 0))),
    [0, Math.floor(n / 2), n - 1].map((i) => s("text", { class: "tick", x: x(i), y: H - 6, "text-anchor": "middle" }, pts[i]?.t || "")),
    s("path", { class: "line-1", d: path("a") }), s("path", { class: "line-2", d: path("b"), "stroke-dasharray": "5 3" }), cross);
  const hit = s("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" });
  hit.addEventListener("pointermove", (evt) => {
    const r = svg.getBoundingClientRect();
    const i = Math.max(0, Math.min(n - 1, Math.round(((((evt.clientX - r.left) / r.width) * W - m.l) / iw) * (n - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    showTip(evt, [[pts[i].t], [labelA, fmt.n(pts[i].a, 2)], [labelB, fmt.n(pts[i].b, 2)]]);
  });
  hit.addEventListener("pointerleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); });
  svg.append(hit);
  return h("div", { class: "chart" },
    h("div", { class: "legend" }, h("span", {}, h("span", { class: "sw", style: "background:var(--series-1);height:3px" }), labelA),
      h("span", {}, h("span", { class: "sw", style: "background:var(--series-2);height:3px" }), `${labelB} (dashed)`)),
    svg, dataTable([["time", (p) => p.t], [labelA, (p) => fmt.n(p.a, 2)], [labelB, (p) => fmt.n(p.b, 2)]], pts));
}

// Horizontal bars (one measure) as HTML so the text stays readable at any width; `baseline` is drawn neutral.
export function barChart(rows, { valueLabel = "", digits = 1, baseline = null } = {}) {
  const max = niceMax(Math.max(...rows.map((r) => r.value), 1));
  const list = h("div", { class: "hbars", role: "list", "aria-label": `${valueLabel} by model` }, rows.map((r) => {
    const bar = h("div", { class: `hbar-fill${r.label === baseline ? " base" : ""}`, style: `width:${((100 * r.value) / max).toFixed(1)}%` });
    const row = h("div", { class: "hbar-row", role: "listitem", tabindex: 0, "aria-label": `${r.label}: ${fmt.n(r.value, digits)} ${valueLabel}` },
      h("span", { class: "hbar-label" }, r.label), h("div", { class: "hbar-track" }, bar), h("span", { class: "hbar-v" }, fmt.n(r.value, digits)));
    row.addEventListener("pointermove", (evt) => showTip(evt, [[r.label], [valueLabel, fmt.n(r.value, digits)], ...(r.extra || [])]));
    row.addEventListener("pointerleave", hideTip);
    return row;
  }));
  return h("div", { class: "chart" }, list,
    dataTable([["model", (r) => r.label], [valueLabel, (r) => fmt.n(r.value, digits)]], rows));
}
