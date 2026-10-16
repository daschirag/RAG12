// About: architecture, team, and live system status (GET /system/status, /system/info, /ready).
import { api } from "../api.js";
import { h, s, fmt, loading, errorBox, setContent, table } from "../ui.js";

const TEAM = [
  { name: "Minmini K", id: "23BIT0198", track: "Track A - Forecasting", what: "Preprocessing, features, persistence/RF/XGBoost/LSTM/TCN, CQR intervals, rolling-origin evaluation, POST /forecast." },
  { name: "Tarunica S", id: "23BIT0185", track: "Track B - RAG and fault diagnosis", what: "Knowledge base, chunking, hybrid retrieval + rerank, Wind Farm A fault detection, evidence-grounded diagnosis, RAG evaluation, POST /diagnose." },
  { name: "Pritika Kasinathan", id: "23BIT0232", track: "Track C - Security and trust", what: "Threat model, attack simulator, detectors, secure retrieval pipeline, trust-aware decision engine, security evaluation, POST /secure-ask and /decision." },
];

export default function mount(root) {
  const status = h("div", { class: "card" }, loading("Loading system status..."));
  root.append(
    h("div", { class: "page-head" }, h("div", {}, h("h2", {}, "About GreenRAG-Sec"),
      h("p", {}, "A local, security-aware assistant for renewable-energy operations. Every component runs on this machine: FastAPI, ChromaDB and an in-memory hybrid index, MiniLM embeddings and a cross-encoder reranker, and Ollama for the LLM. No cloud service is used."))),
    h("div", { class: "card arch", style: "margin-bottom:1rem" }, h("h3", {}, "Architecture"), diagram(),
      h("p", { class: "small muted" }, "The LLM only recommends. Every recommendation goes through the decision engine; only ACCEPT may be acted on without review, and the API never executes an action.")),
    h("h3", {}, "Team"),
    h("div", { class: "team", style: "margin-bottom:1rem" }, TEAM.map((p) => h("div", { class: "card person" },
      h("h4", {}, p.name), h("div", { class: "mono small" }, p.id), h("div", { class: "role" }, p.track), h("p", { class: "small" }, p.what)))),
    status);
  const load = async () => {
    try {
      const [st, info, ready] = await Promise.all([api.get("/system/status"), api.get("/system/info"), fetchReady()]);
      renderStatus(status, st, info, ready);
    } catch (e) {
      setContent(status, errorBox(e, load));
    }
  };
  load();
  document.addEventListener("ready-changed", () => { if (!root.hidden) load(); });
}

async function fetchReady() {
  const { API_BASE } = await import("../api.js");
  return (await fetch(`${API_BASE}/ready`)).json();
}

function renderStatus(el, st, info, ready) {
  const steps = ready.warmup?.steps || [];
  setContent(el, h("h3", {}, "System status"),
    h("div", { class: "kpis" },
      kpi(`${st.backend_mode} mode`, "GET /system/status"), kpi(ready.status.replace("_", " "), "GET /ready"),
      kpi(info.generator_model, "generator (Ollama)"), kpi(info.embedding_model, "embeddings")),
    h("div", { class: "grid grid-2", style: "margin-top:1rem" },
      h("div", {}, h("h4", {}, "Readiness checks"), h("ul", { class: "checks small" }, ready.checks.map((c) =>
        h("li", {}, h("span", { class: c.ok ? "ok-ink" : c.required ? "bad-ink" : "warn-ink" }, c.ok ? "OK " : c.required ? "FAIL " : "WARN "),
          h("strong", {}, c.name), `: ${c.detail}`)))),
      h("div", {}, h("h4", {}, `Warm-up (${ready.warmup?.state || "-"}${ready.warmup?.elapsed_seconds ? `, ${fmt.n(ready.warmup.elapsed_seconds, 0)} s` : ""})`),
        steps.length ? table([{ label: "step", key: "name" }, { label: "state", key: "state" },
          { label: "seconds", num: true, render: (x) => (x.seconds === null ? "-" : fmt.n(x.seconds, 1)) },
          { label: "error", render: (x) => x.error || "" }], steps) : h("p", { class: "small muted" }, "Not started."))),
    h("h4", {}, "Versions"),
    table([{ label: "component", key: "k" }, { label: "version / value", key: "v" }], [
      { k: "Python", v: info.python }, { k: "Platform", v: info.platform }, { k: "Code (git stamp)", v: info.code_version },
      { k: "Reranker", v: info.rerank_model }, { k: "Vector store", v: info.vector_store },
      { k: "Ollama models installed", v: info.ollama_models.join(", ") || "Ollama not reachable" },
      { k: "LLM timeout", v: `${info.llm_timeout_s} s (then evidence-only fallback)` },
      ...Object.entries(info.packages).map(([k, v]) => ({ k, v }))]));
}

function kpi(v, l) {
  return h("div", { class: "kpi" }, h("div", { class: "v", style: "font-size:1.05rem" }, v), h("div", { class: "l" }, l));
}

function diagram() {
  const box = (x, y, w, t, sub, cls = "") => [s("rect", { class: `box ${cls}`, x, y, width: w, height: 46, rx: 8 }),
    s("text", { x: x + w / 2, y: y + 20, "text-anchor": "middle", "font-weight": 600 }, t), s("text", { class: "sub", x: x + w / 2, y: y + 36, "text-anchor": "middle" }, sub)];
  const edge = (x1, y1, x2, y2) => s("path", { class: "edge", d: `M${x1},${y1} L${x2},${y2}` });
  return s("svg", { viewBox: "0 0 960 330", role: "img", "aria-label": "Architecture: console calls the FastAPI app; questions pass input checks, injection detection, hybrid retrieval, document trust and the evidence filter before the Ollama LLM; forecasts and diagnoses come from their models; every recommendation goes through the trust-aware decision engine which returns accept, review or reject." },
    s("defs", {}, s("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" },
      s("path", { d: "M0,0 L10,5 L0,10 z", fill: "currentColor" }))),
    box(20, 140, 130, "Console", "/console (this page)"),
    box(190, 140, 130, "FastAPI", "/assistant router"),
    box(370, 20, 150, "Input + injection", "C4 checks (block)", "sec"),
    box(550, 20, 150, "Hybrid retrieval", "BM25 + MiniLM + rerank"),
    box(730, 20, 150, "Doc trust + filter", "signed registry", "sec"),
    box(730, 100, 150, "Ollama LLM", "qwen2.5:7b (local)"),
    box(370, 140, 150, "Forecasting", "XGBoost + CQR"),
    box(370, 220, 150, "Fault diagnosis", "B4 detector + KB"),
    box(550, 260, 150, "C3 telemetry gate", "offline (lab only)", "sec"),
    box(730, 200, 150, "Decision engine", "trust score from E, R, C, S", "trust"),
    s("text", { class: "sub", x: 805, y: 268, "text-anchor": "middle" }, "accept / review / reject"),
    edge(150, 163, 188, 163), edge(320, 155, 368, 50), edge(520, 43, 548, 43), edge(700, 43, 728, 43), edge(805, 66, 805, 98),
    edge(805, 146, 805, 198), edge(320, 163, 368, 163), edge(320, 172, 368, 240), edge(520, 163, 728, 215), edge(520, 243, 728, 228),
    edge(625, 260, 728, 240));
}
