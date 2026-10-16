// DOM helpers and shared components (states, badges, trust panel, source cards). Text is always set via
// textContent / text nodes, never innerHTML with API data.
import { api } from "./api.js";

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const SVG_NS = "http://www.w3.org/2000/svg";
export function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, v);
  for (const c of children.flat()) if (c !== null && c !== undefined) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}

export const fmt = {
  n: (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d })),
  p: (v, d = 1) => (v === null || v === undefined ? "-" : `${(v * 100).toFixed(d)}%`),
  rate: (r) => (!r || !r.n ? "n/a" : `${r.k}/${r.n} (${(r.rate * 100).toFixed(1)}%)`),
  ci: (lo, hi, d = 3) => (lo === null || lo === undefined ? "" : `[${Number(lo).toFixed(d)}, ${Number(hi).toFixed(d)}]`),
  dt: (v) => String(v || "").replace("T", " ").slice(0, 16),
};

export function loading(text = "Loading...") {
  return h("div", { class: "state", role: "status" }, h("span", { class: "spinner", "aria-hidden": "true" }), text);
}
export function empty(text) {
  return h("div", { class: "state" }, text);
}
export function errorBox(err, retry) {
  return h("div", { class: "error-box", role: "alert" },
    h("strong", {}, `${err.code || "error"}${err.status ? ` (HTTP ${err.status})` : ""}: `), err.message || String(err),
    err.hint ? h("div", { class: "hint" }, "Hint: ", err.hint) : null,
    err.requestId ? h("div", { class: "hint mono" }, `request_id ${err.requestId}`) : null,
    retry ? h("div", { style: "margin-top:.5rem" }, h("button", { class: "btn btn-ghost", onclick: retry }, "Retry")) : null);
}
export function setContent(el, ...nodes) {
  el.replaceChildren(...nodes.flat().filter(Boolean));
}

const ICONS = {
  accept: "M5 12.5l4.2 4.2L19 7",
  review: "M12 7v6M12 16.5v.5",
  reject: "M7 7l10 10M17 7L7 17",
};
export function badge(decision) {
  const d = decision || "neutral";
  const icon = ICONS[d] ? s("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" },
    s("path", { d: ICONS[d], fill: "none", stroke: "currentColor", "stroke-width": 2.6, "stroke-linecap": "round", "stroke-linejoin": "round" })) : null;
  return h("span", { class: `badge ${d}` }, icon, d);
}

const COMPONENT_NAMES = { E: "Evidence quality (E)", R: "Retrieval relevance (R)", C: "Model confidence (C)", S: "Security (S)" };
const FLOORS = { E: 0.5, R: 0.3, C: 0.5, S: 0.5 }; // DecisionConfig defaults (src/backend/trust/decision.py)

export function trustPanel(t) {
  if (!t) return h("div", { class: "trust" }, h("span", { class: "muted small" }, "No trust decision on this path."));
  const bars = h("div", { class: "bars", role: "list", "aria-label": "Trust score components" },
    ["E", "R", "C", "S"].map((k) => {
      const v = Math.max(0, Math.min(1, t.components?.[k] ?? 0));
      return h("div", { class: "bar-row", role: "listitem" },
        h("span", {}, COMPONENT_NAMES[k], h("span", { class: "muted small" }, ` w ${fmt.n(t.weights?.[k], 2)}`)),
        h("div", { class: "bar-track", title: `floor ${FLOORS[k]}` },
          h("div", { class: "bar-fill", style: `width:${(v * 100).toFixed(1)}%` }),
          h("div", { class: "bar-floor", style: `left:${FLOORS[k] * 100}%`, "aria-hidden": "true" })),
        h("span", { class: "v" }, fmt.n(v, 3)));
    }));
  const banner = t.decision === "review"
    ? h("div", { class: "review-banner", role: "note" }, h("strong", {}, "Needs human review. "), "Do not act on this output until a person has checked it.")
    : t.decision === "reject"
      ? h("div", { class: "review-banner reject", role: "note" }, h("strong", {}, "Rejected. "), "This output must not be used.")
      : null;
  return h("div", { class: `trust ${t.decision}` },
    banner,
    h("div", { class: "trust-head" },
      h("div", {}, h("div", { class: "muted small" }, "Decision"), badge(t.decision)),
      h("div", { style: "text-align:right" }, h("div", { class: "muted small" }, "Trust score T"), h("div", { class: "trust-score" }, fmt.n(t.trust_score, 3)))),
    bars,
    h("div", { class: "muted small" }, "T = 0.35E + 0.15R + 0.30C + 0.20S; the tick on each bar is its floor. T ranks outputs for accept / review / reject; it is not a probability."),
    t.reasons?.length ? h("ul", { class: "reasons" }, t.reasons.map((r) => h("li", {}, r))) : null);
}

// Expandable source cards; chunk text is fetched on first expand (GET /kb/chunks).
export function sourceCards(evidence, cited = []) {
  if (!evidence?.length) return h("p", { class: "muted small" }, "No evidence passed to the model.");
  const citedSet = new Set(cited);
  return h("div", { class: "sources" }, evidence.map((e) => {
    const body = h("div", { class: "body" }, "Loading chunk text...");
    const det = h("details", { class: "source" },
      h("summary", {}, h("span", { class: "mono" }, e.chunk_id),
        e.doc_type ? h("span", { class: "tag" }, e.doc_type) : null,
        e.synthetic ? h("span", { class: "tag synthetic" }, "synthetic") : null,
        citedSet.has(e.chunk_id) ? h("span", { class: "tag" }, "cited") : null,
        e.rerank_score !== undefined ? h("span", { class: "muted small" }, `rerank ${fmt.n(e.rerank_score, 2)}`) : null),
      body);
    det.addEventListener("toggle", async () => {
      if (!det.open || det.dataset.loaded) return;
      det.dataset.loaded = "1";
      try {
        const [c] = await api.get(`/kb/chunks?ids=${encodeURIComponent(e.chunk_id)}`);
        setContent(body, c.title ? h("div", { style: "font-weight:600;color:var(--ink)" }, c.title) : null, c.text);
      } catch (err) {
        setContent(body, errorBox(err));
      }
    });
    return det;
  }));
}

export function table(columns, rows, { caption, rowAttrs } = {}) {
  return h("div", { class: "table-wrap" }, h("table", {},
    caption ? h("caption", { class: "sr-only" }, caption) : null,
    h("thead", {}, h("tr", {}, columns.map((c) => h("th", { class: c.num ? "num" : "", scope: "col" }, c.label)))),
    h("tbody", {}, rows.map((r, i) => h("tr", rowAttrs ? rowAttrs(r, i) : {},
      columns.map((c) => h("td", { class: c.num ? "num" : "" }, c.render ? c.render(r) : r[c.key])))))));
}

export function field(label, control) {
  return h("label", { class: "field" }, label, control);
}
export function select(options, value, attrs = {}) {
  return h("select", attrs, options.map(([v, l]) => h("option", { value: v, selected: String(v) === String(value) }, l)));
}
