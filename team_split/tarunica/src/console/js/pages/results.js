// Results: every number comes from GET /results/summary (read from results/ files), with the caveats.
import { api } from "../api.js";
import { h, fmt, loading, errorBox, setContent, table } from "../ui.js";

const pm = (r, k, d = 1) => (r[`${k}_std`] ? `${fmt.n(r[k], d)} +/- ${fmt.n(r[`${k}_std`], d)}` : fmt.n(r[k], d));
const rc = (r) => (r && r.n ? `${r.k}/${r.n}` : "-");

export default function mount(root) {
  const body = h("div", {}, loading("Loading measured results from results/ ..."));
  root.append(h("div", { class: "page-head" }, h("div", {}, h("h2", {}, "Measured results"),
    h("p", {}, "Read live from the result files written by the experiments (fixed seeds, raw outputs in results/). Nothing on this page is typed in by hand; worse-than-baseline results are shown as they are."))), body);
  const load = () => api.get("/results/summary").then((s) => render(body, s)).catch((e) => setContent(body, errorBox(e, load)));
  load();
}

function render(body, s) {
  const t = Object.fromEntries(s.tracks.map((x) => [x.track, x]));
  setContent(body, h("div", { class: "stack" },
    h("div", { class: "callout bad" }, h("strong", {}, "Read these first"), h("ul", { class: "reasons" }, s.headline_caveats.map((c) => h("li", {}, c)))),
    forecasting(t.forecasting), rag(t["rag-diagnosis"]), security(t["security-trust"]), cloud(t.cloud)));
}

function section(tr, ...content) {
  return h("section", { class: "card stack", "aria-labelledby": `res-${tr.track}` },
    h("h3", { id: `res-${tr.track}` }, tr.title), h("p", { class: "small muted" }, `Owner: ${tr.owner}`),
    ...content,
    h("div", { class: "callout" }, h("strong", {}, "Caveats"), h("ul", { class: "reasons" }, tr.caveats.map((c) => h("li", {}, c)))),
    h("details", { class: "small" }, h("summary", {}, `Sources (${tr.sources.length})`), h("ul", {}, tr.sources.map((x) => h("li", { class: "mono" }, x)))));
}

const metricCols = [{ label: "model", key: "model" }, { label: "MAE (kW)", num: true, render: (r) => pm(r, "mae") },
  { label: "RMSE (kW)", num: true, render: (r) => pm(r, "rmse") }, { label: "MAPE (%)", num: true, render: (r) => pm(r, "mape_pct") },
  { label: "R2", num: true, render: (r) => pm(r, "r2", 3) }, { label: "seeds", num: true, key: "seeds" }];

function forecasting(tr) {
  const m = tr.metrics;
  const cov = m.coverage.filter((c) => c.method === "cqr_mondrian");
  return section(tr,
    ...[1, 2].map((p) => h("div", {}, h("h4", {}, `Solar plant ${p}: fixed test split`),
      table(metricCols, m.solar_fixed_split.filter((r) => r.plant === p)))),
    h("h4", {}, "Wind T1: rolling origin by month"), table(metricCols, m.wind_rolling_origin),
    h("h4", {}, "Interval coverage (cqr_mondrian, rolling origin)"),
    table([{ label: "asset", key: "asset" }, { label: "nominal", num: true, render: (c) => fmt.n(c.level, 2) },
      { label: "measured", num: true, render: (c) => fmt.n(c.coverage, 3) }, { label: "95% CI", num: true, render: (c) => fmt.ci(c.ci_low, c.ci_high) },
      { label: "days", num: true, key: "days" }], cov));
}

function rag(tr) {
  const m = tr.metrics;
  const ir = [{ label: "P@5", num: true, render: (r) => fmt.n(r.precision_at_k, 3) }, { label: "R@5", num: true, render: (r) => fmt.n(r.recall_at_k, 3) },
    { label: "MRR", num: true, render: (r) => fmt.n(r.mrr, 3) }, { label: "nDCG@5", num: true, render: (r) => fmt.n(r.ndcg_at_k, 3) }];
  return section(tr,
    h("h4", {}, "Knowledge-base documents"),
    table([{ label: "type", key: "t" }, { label: "real", num: true, key: "real" }, { label: "synthetic", num: true, key: "synthetic" }],
      Object.entries(m.kb_documents_by_type).map(([k, v]) => ({ t: k, ...v }))),
    h("div", { class: "grid grid-2" },
      h("div", {}, h("h4", {}, "Chunking (labelled B2 queries)"), table([{ label: "method", key: "method" }, ...ir, { label: "queries", num: true, key: "queries" }], m.chunking)),
      h("div", {}, h("h4", {}, "Retrieval"), table([{ label: "variant", key: "variant" }, ...ir], m.retrieval))),
    h("h4", {}, "RAG answer quality on knowledge-base questions (judge: local LLM)"),
    table([{ label: "system", key: "system" }, { label: "faithfulness", num: true, render: (r) => fmt.n(r.faithfulness, 3) },
      { label: "hallucination rate", num: true, render: (r) => fmt.n(r.hallucination_rate, 3) },
      { label: "answer relevance", num: true, render: (r) => fmt.n(r.answer_relevance, 3) }, { label: "items", num: true, key: "items" }], m.rag_kb_questions),
    h("h4", {}, "Fault detectors (Wind Farm A, event level, leave-one-asset-out)"),
    table([{ label: "model", key: "model" }, { label: "accuracy", num: true, render: (r) => fmt.n(r.accuracy, 3) },
      { label: "recall", num: true, render: (r) => fmt.n(r.recall, 3) }, { label: "F1", num: true, render: (r) => fmt.n(r.f1, 3) },
      { label: "AUC", num: true, render: (r) => fmt.n(r.auc, 3) }, { label: "seeds", num: true, key: "seeds" }], m.fault_detectors),
    h("p", { class: "small" }, `Deployed detector AUC ${fmt.n(m.detector_auc.auc, 3)}, 95% CI ${fmt.ci(m.detector_auc.ci_low, m.detector_auc.ci_high)}. `,
      `Diagnosis component correct: ${m.diagnosis_component_correct.map((d) => `${d.mode} ${d.component_correct}/${d.events}`).join(", ")}.`));
}

function security(tr) {
  const m = tr.metrics;
  const kn = m.attack_success.knowledge.filter((r) => r.b);
  const tel = m.attack_success.telemetry;
  const de = m.decision_engine, u = m.utility;
  return section(tr,
    h("h4", {}, "Knowledge attacks: attack success by configuration (qwen2.5:7b-instruct)"),
    table([{ label: "attack", key: "attack_type" }, { label: "system", key: "system" }, { label: "(a) undefended", num: true, render: (r) => rc(r.a) },
      { label: "(b) secure", num: true, render: (r) => rc(r.b) }, { label: "(c) accepted", num: true, render: (r) => rc(r.c) }], kn),
    h("h4", {}, "Telemetry attacks: attacked days that succeed"),
    table([{ label: "asset", key: "asset" }, { label: "attack", key: "attack" }, { label: "(a)", num: true, render: (r) => rc(r.a) },
      { label: "(b) C3 gate", num: true, render: (r) => rc(r.b) }, { label: "(c) accepted", num: true, render: (r) => rc(r.c) }], tel),
    h("h4", {}, "Injection defence"),
    h("p", { class: "small" }, `Known C2 template blocked ${rc(m.injection_defence.c2_template.detected)}; held-out phrasings blocked ${rc(m.injection_defence.held_out.detected)}; `,
      `held-out answers accepted that obeyed the attacker: ${rc(de.accepted_answer_correctness.heldout_injection.accepted_that_obeyed)}.`),
    h("h4", {}, "Utility on clean questions"),
    h("p", { class: "small" }, `Faithfulness undefended ${fmt.n(u.a.faithfulness.mean, 3)} vs secure ${fmt.n(u.b.faithfulness.mean, 3)}; hallucination ${fmt.n(u.a.hallucination_rate.mean, 3)} vs ${fmt.n(u.b.hallucination_rate.mean, 3)}; `,
      `wrongly blocked ${rc(u.b.blocked_or_no_trusted_evidence)}; security stages median ${fmt.n(u.b.security_stage_ms_median, 1)} ms; clean answers accepted ${rc(u.c.accept)}.`),
    h("h4", {}, "Decision engine against truly bad items"),
    table([{ label: "kind", key: "k" }, { label: "bad / labelled", num: true, render: (r) => `${r.bad}/${r.items}` },
      { label: "flag precision", num: true, render: (r) => rc(r.nonaccept_precision) }, { label: "flag recall", num: true, render: (r) => rc(r.nonaccept_recall) },
      { label: "reject precision", num: true, render: (r) => rc(r.reject_precision) }], Object.entries(de.pr).map(([k, v]) => ({ k, ...v }))));
}

function cloud(tr) {
  return section(tr, h("p", {}, `Status: ${tr.metrics.status}. Checked ${fmt.dt(tr.metrics.checked)} UTC. Required settings present: `,
    `${Object.values(tr.metrics.settings_present).filter(Boolean).length}/${Object.keys(tr.metrics.settings_present).length}.`));
}
