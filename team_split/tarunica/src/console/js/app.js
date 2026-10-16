// Console shell: tabs (keyboard accessible), theme, mode pill (/system/status), warm-up (/warmup) and readiness (/ready).
import { api } from "./api.js";
import { h } from "./ui.js";
import assistant from "./pages/assistant.js";
import forecast from "./pages/forecast.js";
import diagnosis from "./pages/diagnosis.js";
import lab from "./pages/lab.js";
import results from "./pages/results.js";
import about from "./pages/about.js";

const PAGES = { assistant, forecast, diagnosis, lab, results, about };
const mounted = new Set();
const tabs = [...document.querySelectorAll('[role="tab"]')];
export const readyState = { status: "checking", body: null };

function show(name, focus = false) {
  if (!PAGES[name]) name = "assistant";
  for (const t of tabs) {
    const on = t.dataset.page === name;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    document.getElementById(`page-${t.dataset.page}`).hidden = !on;
    if (on && focus) t.focus();
  }
  const panel = document.getElementById(`page-${name}`);
  if (!mounted.has(name)) {
    mounted.add(name);
    PAGES[name](panel);
  }
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  document.title = `${tabs.find((t) => t.dataset.page === name).textContent} - GreenRAG-Sec Console`;
}

tabs.forEach((t, i) => {
  t.addEventListener("click", () => show(t.dataset.page));
  t.addEventListener("keydown", (e) => {
    const k = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (k) { e.preventDefault(); show(tabs[(i + k + tabs.length) % tabs.length].dataset.page, true); }
    if (e.key === "Home") { e.preventDefault(); show(tabs[0].dataset.page, true); }
    if (e.key === "End") { e.preventDefault(); show(tabs[tabs.length - 1].dataset.page, true); }
  });
});
window.addEventListener("hashchange", () => show(location.hash.slice(1)));

// theme toggle (stored per browser only; the page works without storage)
const root = document.documentElement;
try { const t = localStorage.getItem("greenrag-theme"); if (t) root.dataset.theme = t; } catch { /* storage unavailable */ }
document.getElementById("theme-toggle").addEventListener("click", () => {
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("greenrag-theme", root.dataset.theme); } catch { /* ignore */ }
});

// mode pill
async function loadMode() {
  const pill = document.getElementById("mode-pill");
  try {
    const st = await api.get("/system/status", { timeoutMs: 8000 });
    pill.textContent = `${st.backend_mode} mode`;
    pill.title = `GET /system/status: ${st.backend_mode} - ${st.vector_store} - ${st.generator_model}`;
  } catch {
    pill.textContent = "API offline";
  }
}

// warm-up + readiness polling
const readyPill = document.getElementById("ready-pill");
const banner = document.getElementById("warm-banner");
const warmText = document.getElementById("warm-text");
async function pollReady() {
  let body;
  try {
    const res = await fetch(`${(await import("./api.js")).API_BASE}/ready`);
    body = await res.json();
  } catch {
    body = { status: "offline", warmup: { steps: [] } };
  }
  readyState.status = body.status;
  readyState.body = body;
  readyPill.dataset.state = body.status;
  document.getElementById("ready-label").textContent = body.status === "offline" ? "API offline" : body.status.replace("_", " ");
  const steps = body.warmup?.steps || [];
  const done = steps.filter((x) => x.state === "done").length;
  const failed = steps.filter((x) => x.state === "failed").map((x) => x.name);
  if (body.status === "warming" || body.status === "cold") {
    banner.hidden = false;
    const running = steps.find((x) => x.state === "running");
    warmText.textContent = `Loading models in the background: ${done}/${steps.length} steps done${running ? ` (now: ${running.name.replace("_", " ")})` : ""}. ` +
      `Cold start takes about 100 s; pages work meanwhile but the first answers are slow.`;
  } else {
    banner.hidden = true;
  }
  readyPill.title = `GET /ready: ${body.status}${failed.length ? `; failed: ${failed.join(", ")}` : ""}`;
  document.dispatchEvent(new CustomEvent("ready-changed", { detail: body }));
  if (["warming", "cold", "offline"].includes(body.status)) setTimeout(pollReady, 3000);
}
readyPill.addEventListener("click", () => show("about"));

async function boot() {
  show(location.hash.slice(1) || "assistant");
  loadMode();
  try { await api.post("/warmup", {}, { timeoutMs: 10000 }); } catch { /* offline: pollReady reports it */ }
  pollReady();
}
boot();
export { h };
