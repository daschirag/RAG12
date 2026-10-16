// Security lab: run the known (synthetic) attacks against the live detectors and the decision engine.
import { api } from "../api.js";
import { h, fmt, loading, errorBox, setContent, trustPanel, empty, field, select, table, badge } from "../ui.js";
import { twoLineChart } from "../charts.js";

const CLEAN_Q = "What should be checked when the pitch system reports a battery fault?";

export default function mount(root) {
  const body = h("div");
  let cat = null, section = "telemetry";
  const nav = h("div", { class: "seg", role: "group", "aria-label": "Lab sections" });
  const sections = [["telemetry", "Telemetry attacks"], ["poisoning", "KB poisoning"], ["injection", "Prompt injection"], ["heldout", "Held-out weakness"]];
  const renderNav = () => setContent(nav, sections.map(([v, l]) => h("button", { type: "button", "aria-pressed": String(v === section),
    onclick: () => { section = v; renderNav(); render(); } }, l)));
  renderNav();
  root.append(h("div", { class: "page-head" }, h("div", {}, h("h2", {}, "Security lab"),
    h("p", {}, "Run the C2 attacks (all attack data synthetic) against the live C3 detectors and the C5 decision engine. Detector verdicts are computed now; forecast harm and LLM outcomes are the recorded C6 run for the same attack."))),
  h("div", { class: "controls" }, nav), body);
  setContent(body, loading("Loading the attack catalogue..."));
  api.get("/security/lab/catalogue").then((c) => { cat = c; render(); }).catch((e) => setContent(body, errorBox(e, () => location.reload())));

  function render() {
    if (!cat) return;
    ({ telemetry, poisoning, injection, heldout })[section](body, cat);
  }
}

const rateCell = (r) => (r ? `${r.k}/${r.n}` : "-");

function telemetry(body, cat) {
  const sel = select(cat.telemetry.map((t, i) => [i, `${t.asset} - ${t.attack} (${t.kind})`]), 0, { id: "lab-tel" });
  const run = h("button", { class: "btn" }, "Run attack");
  const out = h("div", {}, empty("Pick an attack and run it. The detector is fitted on clean training days; the first run per asset takes a few seconds."));
  run.addEventListener("click", async () => {
    const t = cat.telemetry[Number(sel.value)];
    run.disabled = true;
    setContent(out, loading(`Applying ${t.attack} to ${t.asset} and running the C3 detector on every attacked day...`));
    try {
      const r = await api.post("/security/lab/telemetry", { asset: t.asset, attack: t.attack });
      const rec = r.recorded_c6;
      const days = r.days;
      const harm = days.filter((d) => d.forecast_harm_recorded);
      const accepted = days.filter((d) => d.engine?.decision === "accept").length;
      setContent(out, h("div", { class: "stack" },
        h("div", { class: "kpis" },
          kpi(`${r.flagged_days}/${days.length}`, "attacked days flagged by C3 (live)"),
          kpi(r.all_match_recorded ? "yes" : "no", "live verdicts match the recorded C3 run"),
          kpi(`${accepted}/${days.length}`, "days the engine would ACCEPT"),
          harm.length ? kpi(`${fmt.n(mean(harm, (d) => d.forecast_harm_recorded.mae_clean_kw), 0)} -> ${fmt.n(mean(harm, (d) => d.forecast_harm_recorded.mae_attacked_kw), 0)} kW`,
            "forecast MAE clean -> attacked (recorded C6)") : null),
        r.flagged_days < days.length ? h("div", { class: "callout bad" }, h("strong", {}, "Detector miss. "),
          `${days.length - r.flagged_days} attacked day(s) pass the telemetry gate, so their attacked forecasts would be served and can be accepted.`) : null,
        h("div", { class: "card" }, h("h4", {}, `Before / after on ${r.signal.day}: ${r.signal.variable}`),
          twoLineChart(r.signal.times, r.signal.clean, r.signal.attacked, { labelA: "clean telemetry", labelB: "attacked telemetry", unit: r.signal.variable })),
        h("div", { class: "card" }, h("h4", {}, "Per attacked day"),
          table([{ label: "day", key: "day" },
            { label: "C3 verdict (live)", render: (d) => verdict(d.live_flagged) },
            { label: "checks fired", render: (d) => Object.entries(d.checks).filter(([, v]) => v).map(([k]) => k).join(", ") || "-" },
            { label: "matches recorded", render: (d) => (d.matches_recorded === null ? "-" : d.matches_recorded ? "yes" : "NO") },
            { label: "MAE clean -> attacked (kW)", num: true, render: (d) => d.forecast_harm_recorded ? `${fmt.n(d.forecast_harm_recorded.mae_clean_kw, 0)} -> ${fmt.n(d.forecast_harm_recorded.mae_attacked_kw, 0)}` : "-" },
            { label: "coverage attacked", num: true, render: (d) => d.forecast_harm_recorded ? fmt.n(d.forecast_harm_recorded.coverage_attacked, 3) : "-" },
            { label: "engine", render: (d) => d.engine ? badge(d.engine.decision) : "-" }], days, { caption: "Per-day detector and engine results" }),
          h("p", { class: "small muted" }, r.note)),
        rec ? h("div", { class: "card" }, h("h4", {}, "Recorded C6 attack success for this attack"),
          table([{ label: "configuration", key: "c" }, { label: "attacked units where the attack succeeded", num: true, key: "v" }],
            [{ c: "(a) undefended (forecast always served)", v: rateCell(rec.a) }, { c: "(b) C3 telemetry gate", v: rateCell(rec.b) },
              { c: "(c) gate + decision engine (accepted)", v: rateCell(rec.c) }])) : null,
        days[0]?.engine ? h("details", {}, h("summary", {}, "Engine decision for the first attacked day"), trustPanel(days[0].engine),
          h("p", { class: "small muted" }, days[0].engine.interval_source)) : null));
    } catch (e) {
      setContent(out, errorBox(e));
    } finally {
      run.disabled = false;
    }
  });
  setContent(body, h("div", { class: "controls" }, field("Attack", sel), run), out);
}

function verdict(flagged) {
  return h("span", { class: `badge ${flagged ? "reject" : "accept"}` }, flagged ? "flagged" : "passed");
}

function poisoning(body, cat) {
  const sel = select(cat.documents.map((d) => [d.doc_id, `${d.doc_id} (${d.attack})`]), cat.documents[0]?.doc_id, { id: "lab-doc" });
  const run = h("button", { class: "btn" }, "Check document trust");
  const out = h("div", {}, empty("Pick a poisoned or injected document (synthetic) and run document trust on it."));
  run.addEventListener("click", async () => {
    run.disabled = true;
    setContent(out, loading("Running document trust..."));
    try {
      const r = await api.post("/security/lab/poisoning", { doc_id: sel.value });
      const card = (title, v) => h("div", { class: "card" }, h("h4", {}, title),
        h("span", { class: `badge ${v.trusted ? "accept" : "reject"}` }, v.trusted ? "trusted" : "untrusted"),
        v.reasons?.length ? h("ul", { class: "reasons" }, v.reasons.map((x) => h("li", {}, x))) : h("p", { class: "small muted" }, "All checks passed."));
      setContent(out, h("div", { class: "stack" },
        h("div", { class: "grid grid-3" }, card("Full check (signed registry + content)", r.full_check), card("Content checks only", r.content_only),
          r.genuine_original ? card(`Genuine original: ${r.genuine_original.doc_id}`, r.genuine_original) : h("div", { class: "card" }, h("h4", {}, "Genuine original"), h("p", { class: "small muted" }, "None: this document is a new insertion."))),
        r.content_only.trusted && !r.full_check.trusted ? h("div", { class: "callout" }, h("strong", {}, "Content checks alone would let this through. "),
          "Only the registry check (is the document and its hash in the signed build?) catches it.") : null,
        h("div", { class: "card" }, h("h4", {}, `Attack: ${r.attack}`), r.marker ? h("p", { class: "small" }, "Marker: ", h("code", {}, r.marker)) : null,
          r.inserted_text ? h("p", { class: "small" }, "Inserted (synthetic) text: ", h("em", {}, r.inserted_text)) : null),
        h("div", { class: "card" }, h("h4", {}, "Recorded C6 outcomes for this attack type (qwen2.5:7b-instruct)"),
          table([{ label: "system", key: "system" }, { label: "items", num: true, render: (x) => x.a?.n ?? "-" },
            { label: "(a) undefended", num: true, render: (x) => rateCell(x.a) }, { label: "(b) secure", num: true, render: (x) => rateCell(x.b) },
            { label: "(c) accepted", num: true, render: (x) => rateCell(x.c) }], r.recorded_c6),
          h("p", { class: "small muted" }, "Success = attack content in the output. plain_rag is a baseline without a secure variant."))));
    } catch (e) {
      setContent(out, errorBox(e));
    } finally {
      run.disabled = false;
    }
  });
  setContent(body, h("div", { class: "controls" }, field("Attack document", sel), run), out);
}

function injection(body, cat) {
  const presets = [["clean", "Clean question (no attack)"], ["c2", "C2 template (known to the detector)"],
    ...cat.held_out_phrasings.map((p) => [`h${p.index}`, `Held-out #${p.index}: ${p.text.slice(0, 60)}`])];
  const sel = select(presets, "c2", { id: "lab-preset" });
  const text = h("textarea", { rows: 3, id: "lab-q", "aria-label": "Question to test" });
  const full = h("input", { type: "checkbox", id: "lab-full" });
  const run = h("button", { class: "btn" }, "Test question");
  const out = h("div");
  const apply = () => {
    const v = sel.value;
    text.value = v === "clean" ? CLEAN_Q : v === "c2" ? `${CLEAN_Q} ${cat.query_injection_template}` : `${CLEAN_Q} ${cat.held_out_phrasings[Number(v.slice(1))].text}`;
  };
  sel.addEventListener("change", apply);
  apply();
  run.addEventListener("click", async () => {
    run.disabled = true;
    setContent(out, loading(full.checked ? "Running the detector, then the full secure pipeline with the LLM..." : "Running input checks and the injection detector..."));
    try {
      const r = await api.post("/security/lab/injection", { question: text.value, run_pipeline: full.checked });
      setContent(out, h("div", { class: "stack" },
        h("div", { class: "card" }, h("h4", {}, "Injection detector (live)"),
          h("span", { class: `badge ${r.blocked ? "reject" : "accept"}` }, r.blocked ? "blocked" : "not blocked"),
          r.injection_signals.length ? h("p", { class: "small" }, "Signals: ", r.injection_signals.join(", ")) : h("p", { class: "small muted" }, "No injection pattern found."),
          r.note ? h("p", { class: "small muted" }, r.note) : null),
        r.answer ? h("div", { class: "card" }, h("h4", {}, "Answer from the secure pipeline"), h("p", {}, r.answer.answer), h("span", { class: "tag" }, `route ${r.answer.route}`)) : null,
        r.engine ? h("div", {}, h("h4", {}, "Decision engine"), trustPanel(r.engine)) : null,
        sel.value.startsWith("h") ? heldoutRow(cat.held_out_phrasings[Number(sel.value.slice(1))]) : null));
    } catch (e) {
      setContent(out, errorBox(e));
    } finally {
      run.disabled = false;
    }
  });
  setContent(body, h("div", { class: "card stack" },
    h("div", { class: "controls" }, field("Preset", sel), h("label", { class: "field", style: "flex-direction:row;align-items:center;gap:.4rem" }, full, "Also run the full pipeline with the LLM (slow)"), run),
    text), out);
}

function heldoutRow(p) {
  return h("div", { class: "callout info" }, h("strong", {}, `Recorded C6 result for this phrasing (${p.detected?.n ?? "?"} questions): `),
    `detected ${rateCell(p.detected)}`, p.obeyed_given_not_blocked ? `; obeyed when not blocked ${rateCell(p.obeyed_given_not_blocked)}` : "",
    p.obeyed_and_accepted ? `; obeyed AND accepted by the engine ${rateCell(p.obeyed_and_accepted)}` : "");
}

function heldout(body, cat) {
  const ov = cat.held_out_overall, eng = cat.held_out_engine;
  const missed = cat.held_out_phrasings.filter((p) => p.detected && p.detected.k === 0);
  setContent(body, h("div", { class: "stack" },
    h("div", { class: "callout bad" }, h("strong", {}, "Known weakness. "),
      `The injection detector blocked ${rateCell(ov.detected)} held-out items: ${missed.length} of ${cat.held_out_phrasings.length} phrasings were never blocked. `,
      `Of ${eng.delivered} delivered answers the engine accepted ${eng.accepted.k}, and ${eng.accepted_that_obeyed.k} of those ${eng.accepted_that_obeyed.n} accepted answers executed the injected instruction. `,
      `The engine stopped only ${eng.obeyed_not_accepted.k} of ${eng.obeyed_not_accepted.n} obeyed answers: its unsafe-directive check is a keyword list.`),
    h("div", { class: "kpis" }, kpi(rateCell(ov.detected), "held-out items blocked"), kpi(`${eng.accepted_that_obeyed.k}/${eng.accepted_that_obeyed.n}`, "accepted answers that obeyed the attacker"),
      kpi(rateCell(ov.attack_success_c), "end-to-end success, accepted (phrasings with an observable payload)"), kpi(rateCell(cat.c2_template.detected), "known C2 template blocked (upper bound)")),
    h("div", { class: "card" }, h("h4", {}, `Per phrasing (recorded C6 run, qwen2.5:7b-instruct, ${cat.held_out_phrasings[0]?.detected?.n ?? "?"} clean questions each)`),
      table([{ label: "#", num: true, key: "index" }, { label: "phrasing", key: "text" },
        { label: "blocked", num: true, render: (p) => rateCell(p.detected) },
        { label: "obeyed (not blocked)", num: true, render: (p) => (p.obeyed_given_not_blocked ? rateCell(p.obeyed_given_not_blocked) : "no payload") },
        { label: "obeyed + accepted", num: true, render: (p) => (p.obeyed_and_accepted ? rateCell(p.obeyed_and_accepted) : "-") }], cat.held_out_phrasings,
      { rowAttrs: (p) => ({ class: p.detected?.k === 0 ? "selected" : "" }) }),
      h("p", { class: "small muted" }, "Highlighted rows were never blocked. Obedience markers were fixed before the run and undercount (docs/security_evaluation.md). Source: results/security/c6_injection_defence.json, c6_decision_engine.json."))));
}

function kpi(v, l) {
  return h("div", { class: "kpi" }, h("div", { class: "v" }, v), h("div", { class: "l" }, l));
}
function mean(arr, f) {
  return arr.reduce((a, x) => a + f(x), 0) / arr.length;
}
