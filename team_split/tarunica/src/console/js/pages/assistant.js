// Assistant: chat-style Q&A. Knowledge base -> POST /assistant (config A-F; F = secure pipeline + trust decision).
// Legacy domains -> GET /ask/{domain} (no trust scoring on that path; said in the UI).
import { api } from "../api.js";
import { h, fmt, loading, errorBox, setContent, trustPanel, sourceCards, field, select, badge } from "../ui.js";

const CONFIGS = [["F", "F - full GreenRAG-Sec (secure + trust)"], ["E", "E - hybrid RAG + forecast + diagnosis"],
  ["D", "D - hybrid RAG + forecast"], ["C", "C - hybrid RAG"], ["B", "B - basic RAG"], ["A", "A - LLM only"]];

export default function mount(root) {
  const domainSel = select([["kb", "Knowledge base (secure, trust-scored)"]], "kb", { id: "as-domain" });
  const configSel = select(CONFIGS, "F", { id: "as-config" });
  const chat = h("div", { class: "chat", "aria-live": "polite", "aria-label": "Conversation" });
  const input = h("textarea", { rows: 2, id: "as-input", placeholder: "Ask about turbine faults, SCADA signals, solar inverters or maintenance...", "aria-label": "Question" });
  const send = h("button", { class: "btn", type: "submit" }, "Ask");
  const sugg = h("div", { class: "chips", "aria-label": "Suggested questions" }, h("span", { class: "muted small" }, "Loading suggestions..."));

  setContent(chat, h("div", { class: "state" },
    h("p", {}, h("strong", {}, "Ask a question about the knowledge base.")),
    h("p", { class: "small muted" }, "Answers in config F pass input checks, prompt-injection detection, document trust and an evidence filter before the LLM, then the decision engine scores them. Pick a suggested question to start.")));

  root.append(
    h("div", { class: "page-head" },
      h("div", {}, h("h2", {}, "Assistant"),
        h("p", {}, "Evidence-grounded answers with cited sources and a trust decision. Only answers marked ACCEPT may be used without human review."))),
    h("div", { class: "controls" }, field("Domain", domainSel), field("Configuration (ablation)", configSel)),
    h("div", { class: "card", style: "margin-bottom:1rem" }, h("h4", {}, "Suggested questions"), sugg),
    chat,
    h("form", { class: "composer", onsubmit: (e) => { e.preventDefault(); ask(); } }, input, send));

  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
  domainSel.addEventListener("change", () => { configSel.disabled = domainSel.value !== "kb"; });

  api.get("/domains").then((ds) => {
    for (const d of ds.filter((x) => x !== "knowledge_base")) domainSel.append(h("option", { value: d }, `${d} (legacy /ask, no trust score)`));
  }).catch(() => {});
  api.get("/assistant/suggestions?per_type=1").then((list) => {
    setContent(sugg, list.map((q) => h("button", { class: "chip", type: "button", title: q.query_type, onclick: () => { input.value = q.text; ask(); } }, q.text)));
  }).catch((e) => setContent(sugg, errorBox(e)));

  async function ask() {
    const q = input.value.trim();
    if (!q || send.disabled) return;
    if (chat.querySelector(".state")) chat.replaceChildren();
    input.value = "";
    send.disabled = true;
    chat.append(h("div", { class: "msg user" }, q));
    const slot = h("div", { class: "msg bot" }, loading("Retrieving evidence and generating (the first answer after a cold start can take ~100 s)..."));
    chat.append(slot);
    slot.scrollIntoView({ block: "end", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    try {
      if (domainSel.value === "kb") renderAssistant(slot, await api.post("/assistant", { message: q, config: configSel.value }));
      else renderLegacy(slot, await api.get(`/ask/${encodeURIComponent(domainSel.value)}?q=${encodeURIComponent(q)}&top_k=3`));
    } catch (e) {
      setContent(slot, errorBox(e, () => { input.value = q; slot.remove(); ask(); }));
    } finally {
      send.disabled = false;
      input.focus();
    }
  }
}

function renderAssistant(slot, r) {
  const out = r.output || {};
  const t = r.trust;
  if (t) slot.classList.add(t.decision);
  const sec = out.security;
  const meta = h("div", { class: "meta" },
    h("span", { class: "tag" }, `config ${r.config}`), h("span", { class: "tag" }, `intent ${r.intent}`),
    h("span", { class: "tag" }, `handled by ${r.handled_by}`), out.route ? h("span", { class: "tag" }, `route ${out.route}`) : null,
    t ? badge(t.decision) : h("span", { class: "tag" }, "no trust decision (configs A-E)"));
  const secLine = sec ? h("p", { class: "small muted" },
    sec.blocked ? `Blocked before retrieval: ${(sec.injection_signals || []).join(", ") || (sec.input_reasons || []).join(", ")}.`
      : `Security: ${sec.retrieved_doc_ids?.length ?? 0} documents retrieved, ${sec.rejected?.length ?? 0} chunks rejected by trust verification, ` +
        `${sec.passed_chunk_ids?.length ?? 0} trusted chunks passed to the LLM (security stages ${fmt.n((sec.seconds_security || 0) * 1000, 1)} ms).`) : null;
  const evidence = out.evidence || (out.context || []).map((c) => ({ chunk_id: c.chunk_id, doc_id: c.doc_id }));
  setContent(slot,
    t?.decision === "review" ? h("div", { class: "review-banner" }, h("strong", {}, "In review: "), "check this answer against its sources before using it.") : null,
    t?.decision === "reject" ? h("div", { class: "review-banner reject" }, h("strong", {}, "Rejected: "), "do not use this answer.") : null,
    h("div", { class: "answer" }, r.text), meta, secLine,
    h("details", { open: !!t }, h("summary", {}, "Trust decision"), trustPanel(t)),
    h("details", {}, h("summary", {}, `Sources (${evidence.length})`), sourceCards(evidence, out.cited_chunk_ids || [])));
}

function renderLegacy(slot, r) {
  setContent(slot, h("div", { class: "answer" }, r.answer),
    h("div", { class: "meta" }, h("span", { class: "tag" }, `domain ${r.domain_id}`), h("span", { class: "tag" }, `retrieval ${r.retrieval_mode}`),
      h("span", { class: "tag" }, "legacy /ask: no security pipeline, no trust decision")),
    h("details", {}, h("summary", {}, `Sources (${(r.sources || []).length})`),
      h("ul", { class: "small" }, (r.sources || []).map((src) => h("li", { class: "mono" }, typeof src === "string" ? src : `${src.chunk_id || ""} ${src.equipment_id || ""} ${String(src.timestamp_start || "").slice(0, 10)}`)))));
}
