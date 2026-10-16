// Fault diagnosis: Wind Farm A event list (GET /diagnose/events), then POST /decision {kind: diagnosis} which runs the
// evidence-grounded diagnosis (logged / unlogged mode) and the trust engine.
import { api } from "../api.js";
import { h, fmt, loading, errorBox, setContent, trustPanel, empty, field, badge } from "../ui.js";

export default function mount(root) {
  const listCard = h("div", { class: "card" }, loading("Loading Wind Farm A events..."));
  const report = h("div", { class: "card" }, empty("Select an event on the left to diagnose it."));
  let filter = "anomaly", events = [], selected = null, mode = "logged";
  const seg = (vals, cur, on) => h("div", { class: "seg", role: "group" }, vals.map(([v, l]) =>
    h("button", { type: "button", "aria-pressed": String(v === cur), onclick: () => on(v) }, l)));

  root.append(
    h("div", { class: "page-head" }, h("div", {}, h("h2", {}, "Fault diagnosis"),
      h("p", {}, "Wind Farm A SCADA events. The detector score comes from the B4 logistic-regression model; the cause comes only from retrieved, trust-verified evidence with cited chunks."))),
    h("div", { class: "split", style: "grid-template-columns:minmax(0,1fr) minmax(0,1.25fr)" }, listCard, report));

  api.get("/diagnose/events").then((e) => { events = e; renderList(); })
    .catch((e) => setContent(listCard, errorBox(e, () => location.reload())));

  function renderList() {
    const rows = events.filter((e) => filter === "all" || e.label === filter);
    const tbody = h("tbody", {}, rows.map((e) => {
      const tr = h("tr", { class: `clickable${selected?.event_id === e.event_id ? " selected" : ""}`, tabindex: 0,
        "aria-label": `Event ${e.event_id}, turbine ${e.asset_id}, ${e.label}` },
        h("td", { class: "num" }, e.event_id), h("td", { class: "num" }, e.asset_id), h("td", {}, e.label),
        h("td", { class: "nowrap" }, fmt.dt(e.start).slice(0, 10)), h("td", { class: "num" }, e.detector_score === null ? "-" : fmt.n(e.detector_score, 3)),
        h("td", {}, e.component || "-"));
      const pick = () => { selected = e; renderList(); renderControls(); };
      tr.addEventListener("click", pick);
      tr.addEventListener("keydown", (k) => { if (k.key === "Enter" || k.key === " ") { k.preventDefault(); pick(); } });
      return tr;
    }));
    setContent(listCard, h("h3", {}, "Events"),
      h("div", { class: "controls" }, seg([["anomaly", "Anomalies"], ["normal", "Normal"], ["all", "All"]], filter, (v) => { filter = v; renderList(); }),
        h("span", { class: "small muted" }, `${rows.length} events`)),
      h("div", { class: "list-scroll" }, h("table", {}, h("thead", {}, h("tr", {},
        ["event", "turbine", "label", "start", "detector score", "logged component"].map((c, i) => h("th", { class: [0, 1, 4].includes(i) ? "num" : "", scope: "col" }, c)))), tbody)),
      h("p", { class: "small muted" }, "Logged component = ground truth from event_info.csv, shown for evaluation; in unlogged mode the diagnosis cannot see this event's own records."));
  }

  function renderControls(result) {
    if (!selected) return;
    const e = selected;
    const llm = h("input", { type: "checkbox", checked: true, id: "dg-llm" });
    const run = h("button", { class: "btn" }, "Diagnose");
    const out = h("div", {}, result || null);
    run.addEventListener("click", async () => {
      run.disabled = true;
      setContent(out, loading("Retrieving evidence and generating the report (first call after a cold start can take ~100 s)..."));
      try {
        const d = await api.post("/decision", { kind: "diagnosis", diagnose: { event_id: e.event_id, mode, use_llm: llm.checked } });
        setContent(out, renderReport(d, e));
      } catch (err) {
        setContent(out, errorBox(err));
      } finally {
        run.disabled = false;
      }
    });
    setContent(report, h("h3", {}, `Event ${e.event_id} - turbine ${e.asset_id}`),
      h("p", { class: "small muted" }, `${fmt.dt(e.start)} to ${fmt.dt(e.end)} - ${e.label}${e.detector_flag !== null ? ` - detector ${e.detector_flag ? "flagged" : "not flagged"}` : ""}`),
      h("div", { class: "controls" },
        field("Mode", seg([["logged", "Logged"], ["unlogged", "Unlogged (own records hidden)"]], mode, (v) => { mode = v; renderControls(); })),
        h("label", { class: "field", style: "flex-direction:row;align-items:center;gap:.4rem" }, llm, "Use the LLM (off = evidence-only report)"), run),
      out);
  }
}

function renderReport(d, e) {
  const r = d.output, c = r.confidence;
  const ev = r.evidence || [];
  const cited = new Set([...(r.likely_cause.chunk_ids || []), ...(r.recommended_action.chunk_ids || [])]);
  const correct = e.component && r.likely_cause_component ? (e.component === r.likely_cause_component) : null;
  return h("div", { class: "stack" },
    h("div", { class: "meta", style: "display:flex;gap:.4rem;flex-wrap:wrap" },
      h("span", { class: `badge ${r.severity === "high" ? "reject" : r.severity === "medium" ? "review" : "neutral"}` }, `severity ${r.severity}`),
      h("span", { class: "badge neutral" }, `confidence ${c.level}`), h("span", { class: "tag" }, `generation ${r.generation}`),
      r.likely_cause_component ? h("span", { class: "tag" }, `component ${r.likely_cause_component}`) : null),
    h("p", {}, h("strong", {}, r.diagnosis)),
    h("div", {}, h("h4", {}, "Likely cause"), h("p", {}, r.likely_cause.text, " ", h("span", { class: "mono small muted" }, (r.likely_cause.chunk_ids || []).join(", ")))),
    h("div", {}, h("h4", {}, "Recommended action"), h("p", {}, r.recommended_action.text)),
    correct === null ? null : h("div", { class: `callout ${correct ? "info" : "bad"}` },
      `Logged component (ground truth): ${e.component}. Diagnosed component: ${r.likely_cause_component}. ${correct ? "Match." : "Mismatch."}`),
    h("details", {}, h("summary", {}, "What the confidence label means"),
      h("p", { class: "small" }, c.explanation),
      h("p", { class: "small muted" }, `Detector reliability: event AUC ${fmt.n(c.reliability.value, 3)}, 95% CI ${fmt.ci(c.reliability.ci95_low, c.reliability.ci95_high)} over ${c.reliability.n_events} events; ` +
        `anomaly score ${c.anomaly_score === null ? "n/a" : fmt.n(c.anomaly_score, 3)}; evidence support ${c.evidence_support}. The label is not a probability that the diagnosis is correct.`),
      h("p", { class: "small muted" }, `Severity basis: ${r.severity_basis}`)),
    h("details", { open: true }, h("summary", {}, `Cited evidence (${ev.length} chunks)`),
      h("div", { class: "sources" }, ev.map((x) => h("details", { class: "source" },
        h("summary", {}, h("span", { class: "mono" }, x.chunk_id), h("span", { class: "tag" }, x.doc_type),
          x.synthetic ? h("span", { class: "tag synthetic" }, "synthetic") : null, cited.has(x.chunk_id) ? h("span", { class: "tag" }, "cited") : null,
          h("span", { class: "muted small" }, `rerank ${fmt.n(x.rerank_score, 2)}`)),
        h("div", { class: "body" }, h("div", { style: "font-weight:600;color:var(--ink)" }, x.title), x.excerpt))))),
    h("h4", {}, "Trust decision"), trustPanel(d),
    r.notes?.length ? h("details", {}, h("summary", {}, "Notes"), h("ul", { class: "small" }, r.notes.map((n) => h("li", {}, n)))) : null);
}
