const API_BASE = "http://127.0.0.1:8000";

const DOMAIN_META = {
  solar: {
    label: "Solar",
    accent: "var(--accent-solar)",
    icon: `<svg class="domain-icon" viewBox="0 0 17 17" aria-hidden="true"><circle cx="8.5" cy="8.5" r="3.1" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M8.5 1.5v2.2M8.5 13.3v2.2M1.5 8.5h2.2M13.3 8.5h2.2M3.4 3.4l1.55 1.55M12.05 12.05l1.55 1.55M3.4 13.6l1.55-1.55M12.05 4.95l1.55-1.55" fill="none" stroke="currentColor" stroke-width="1.35" stroke-linecap="round"/></svg>`,
  },
  wind: {
    label: "Wind",
    accent: "var(--accent-wind)",
    icon: `<svg class="domain-icon" viewBox="0 0 17 17" aria-hidden="true"><circle cx="8.5" cy="7.2" r="1.35" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M8.5 7.2L8.5 2.2M8.5 7.2L12.8 11.1M8.5 7.2L4.2 11.1" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/><path d="M8.5 14.5V11.8" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>`,
  },
  fault_diagnosis: {
    label: "Fault",
    accent: "var(--accent-fault)",
    icon: `<svg class="domain-icon" viewBox="0 0 17 17" aria-hidden="true"><path d="M8.5 2.2L15 14.2H2L8.5 2.2Z" fill="none" stroke="currentColor" stroke-width="1.35" stroke-linejoin="round"/><path d="M8.5 6.6v3.4M8.5 12.2h.01" fill="none" stroke="currentColor" stroke-width="1.45" stroke-linecap="round"/><path d="M11.8 9.2l1.6 1.1M12.2 7.4l1.9-.2" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>`,
  },
};

const els = {
  statusPill: document.getElementById("status-pill"),
  statusLabel: document.getElementById("status-label"),
  domainSwitches: document.getElementById("domain-switches"),
  queryInput: document.getElementById("query-input"),
  askBtn: document.getElementById("ask-btn"),
  consoleHint: document.getElementById("console-hint"),
  emptyState: document.getElementById("empty-state"),
  resultView: document.getElementById("result-view"),
  vizSlot: document.getElementById("viz-slot"),
  metricValue: document.getElementById("metric-value"),
  answerBody: document.getElementById("answer-body"),
  sourcesList: document.getElementById("sources-list"),
  errorState: document.getElementById("error-state"),
};

let selectedDomain = null;
let domainsCache = [];
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function setStatus(ok, label) {
  els.statusPill.classList.toggle("is-ok", ok === true);
  els.statusPill.classList.toggle("is-bad", ok === false);
  els.statusLabel.textContent = label;
}

function setDomainAccent(domainId) {
  const meta = DOMAIN_META[domainId];
  document.documentElement.style.setProperty(
    "--domain-accent",
    meta ? meta.accent : "var(--accent-wind)"
  );
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderInlineMarkdown(text) {
  const escaped = escapeHtml(text);
  return escaped.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function parseAnswer(text) {
  const raw = String(text || "").trim();
  if (!raw) {
    return { lede: "", bullets: [] };
  }
  const lines = raw.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const bullets = [];
  const prose = [];
  for (const line of lines) {
    if (/^[-*]\s+/.test(line)) {
      bullets.push(line.replace(/^[-*]\s+/, ""));
    } else {
      prose.push(line);
    }
  }
  let lede = prose.join(" ").trim();
  if (!lede && bullets.length) {
    lede = bullets.shift();
  }
  const sentenceMatch = lede.match(/^(.+?[.!?])(?:\s+|$)/);
  if (sentenceMatch && prose.length <= 1) {
    const rest = lede.slice(sentenceMatch[1].length).trim();
    lede = sentenceMatch[1];
    if (rest && !bullets.length) {
      // keep extra prose as second paragraph-ish via bullets only if dash-style
    }
  }
  return { lede, bullets: bullets.slice(0, 8) };
}

function renderAnswerHtml(answer) {
  const { lede, bullets } = parseAnswer(answer);
  let html = "";
  if (lede) {
    html += `<p class="answer-lede">${renderInlineMarkdown(lede)}</p>`;
  }
  if (bullets.length) {
    html += `<ul class="answer-bullets">${bullets
      .map((b) => `<li>${renderInlineMarkdown(b)}</li>`)
      .join("")}</ul>`;
  }
  return html || `<p class="answer-lede">${escapeHtml(String(answer || ""))}</p>`;
}

function formatSourceChip(src) {
  if (typeof src === "string") {
    return `<span class="source-chip">${escapeHtml(src)}</span>`;
  }
  const chunk = src.chunk_id || "—";
  const equip = src.equipment_id || "—";
  let datePart = "";
  const ts = src.timestamp_start || src.timestamp_end || "";
  if (ts) {
    const d = String(ts).slice(0, 10);
    datePart = ` · ${d}`;
  }
  return `<span class="source-chip">${escapeHtml(chunk)} · ${escapeHtml(equip)}${escapeHtml(datePart)}</span>`;
}

function gaugeGeometry() {
  // Arc from 200° to 340° (140° sweep), center 100,100, radius 70
  const cx = 100;
  const cy = 100;
  const r = 70;
  const startAngle = (-110 * Math.PI) / 180;
  const endAngle = (110 * Math.PI) / 180;
  const x1 = cx + r * Math.cos(startAngle);
  const y1 = cy + r * Math.sin(startAngle);
  const x2 = cx + r * Math.cos(endAngle);
  const y2 = cy + r * Math.sin(endAngle);
  const arcLen = r * (endAngle - startAngle);
  return { cx, cy, r, startAngle, endAngle, x1, y1, x2, y2, arcLen };
}

function renderGaugeSvg({ value, max, loading }) {
  const g = gaugeGeometry();
  const ratio = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  const offset = g.arcLen * (1 - ratio);
  const needleAngle = -110 + 220 * (loading && !prefersReducedMotion ? 0.5 : ratio);
  const loadingClass = loading && !prefersReducedMotion ? " gauge-loading" : "";

  return `
    <div class="gauge-wrap${loadingClass}">
      <svg class="gauge-svg" viewBox="0 0 200 130" role="img" aria-label="Metric gauge">
        <path class="gauge-track" d="M ${g.x1.toFixed(2)} ${g.y1.toFixed(2)} A ${g.r} ${g.r} 0 0 1 ${g.x2.toFixed(2)} ${g.y2.toFixed(2)}" />
        <path class="gauge-progress" d="M ${g.x1.toFixed(2)} ${g.y1.toFixed(2)} A ${g.r} ${g.r} 0 0 1 ${g.x2.toFixed(2)} ${g.y2.toFixed(2)}"
          stroke-dasharray="${g.arcLen.toFixed(2)}" stroke-dashoffset="${loading ? g.arcLen.toFixed(2) : offset.toFixed(2)}" />
        <text class="gauge-tick" x="28" y="118">0</text>
        <text class="gauge-tick" x="158" y="118" text-anchor="start">${escapeHtml(String(max))}</text>
        <line class="gauge-needle" x1="100" y1="100" x2="100" y2="38"
          style="transform: rotate(${needleAngle.toFixed(2)}deg)" />
        <circle class="gauge-hub" cx="100" cy="100" r="6" />
      </svg>
    </div>`;
}

function renderWaveSvg(loading) {
  const loadingClass = loading && !prefersReducedMotion ? " wave-loading" : "";
  return `
    <div class="wave-wrap${loadingClass}">
      <svg class="wave-svg" viewBox="0 0 280 56" role="img" aria-label="Semantic waveform">
        <path class="wave-path" stroke-dasharray="8 6"
          d="M0 28 C20 8, 40 48, 60 28 S100 8, 120 28 S160 48, 180 28 S220 8, 240 28 S260 48, 280 28" />
      </svg>
    </div>`;
}

function showLoadingViz(modeHint) {
  els.emptyState.classList.add("is-hidden");
  els.errorState.classList.add("is-hidden");
  els.resultView.classList.remove("is-hidden");
  els.metricValue.classList.add("is-hidden");
  els.answerBody.innerHTML = `<p class="answer-lede" style="color:var(--text-muted)">Retrieving…</p>`;
  els.sourcesList.innerHTML = "";
  // Prefer gauge sweep while waiting; switch after response based on mode
  els.vizSlot.innerHTML = renderGaugeSvg({ value: 0.5, max: 1, loading: true });
  if (modeHint === "semantic") {
    els.vizSlot.innerHTML = renderWaveSvg(true);
  }
}

function showResult(payload) {
  els.emptyState.classList.add("is-hidden");
  els.errorState.classList.add("is-hidden");
  els.resultView.classList.remove("is-hidden");

  const mode = payload.retrieval_mode;
  const metric = payload.metric;

  if (mode === "numeric" && metric && typeof metric.value === "number") {
    const max = typeof metric.max === "number" ? metric.max : 1;
    els.vizSlot.innerHTML = renderGaugeSvg({
      value: metric.value,
      max,
      loading: false,
    });
    // Force reflow then set final dashoffset/needle for transition
    const progress = els.vizSlot.querySelector(".gauge-progress");
    const needle = els.vizSlot.querySelector(".gauge-needle");
    if (progress && needle && !prefersReducedMotion) {
      const g = gaugeGeometry();
      const ratio = max > 0 ? Math.max(0, Math.min(1, metric.value / max)) : 0;
      progress.style.strokeDashoffset = String(g.arcLen);
      needle.style.transform = "rotate(-110deg)";
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          progress.style.strokeDashoffset = String(g.arcLen * (1 - ratio));
          needle.style.transform = `rotate(${(-110 + 220 * ratio).toFixed(2)}deg)`;
        });
      });
    }
    els.metricValue.textContent = Number(metric.value).toFixed(2);
    els.metricValue.classList.remove("is-hidden");
  } else {
    els.vizSlot.innerHTML = renderWaveSvg(false);
    els.metricValue.classList.add("is-hidden");
  }

  els.answerBody.innerHTML = renderAnswerHtml(payload.answer);
  const sources = Array.isArray(payload.sources) ? payload.sources : [];
  els.sourcesList.innerHTML = sources.length
    ? sources.map(formatSourceChip).join("")
    : `<span class="source-chip">no sources</span>`;
}

function showError(message) {
  els.emptyState.classList.add("is-hidden");
  els.resultView.classList.add("is-hidden");
  els.errorState.classList.remove("is-hidden");
  els.errorState.textContent = message;
}

function renderDomainButtons(domains) {
  els.domainSwitches.innerHTML = "";
  domains.forEach((id) => {
    const meta = DOMAIN_META[id] || { label: id, icon: "" };
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "domain-btn";
    btn.dataset.domain = id;
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", "false");
    btn.innerHTML = `${meta.icon || ""}<span>${escapeHtml(meta.label || id)}</span>`;
    btn.addEventListener("click", () => selectDomain(id));
    els.domainSwitches.appendChild(btn);
  });
}

function selectDomain(domainId) {
  selectedDomain = domainId;
  setDomainAccent(domainId);
  els.domainSwitches.querySelectorAll(".domain-btn").forEach((btn) => {
    const active = btn.dataset.domain === domainId;
    btn.classList.toggle("is-active", active);
    btn.setAttribute("aria-checked", active ? "true" : "false");
  });
  els.consoleHint.textContent = `Querying ${DOMAIN_META[domainId]?.label || domainId} corpus.`;
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_) {
      /* ignore */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

async function initStatus() {
  try {
    const health = await fetchJson(`${API_BASE}/health`);
    setStatus(true, health.status === "ok" ? "ONLINE" : String(health.status || "OK").toUpperCase());
  } catch (err) {
    setStatus(false, "OFFLINE");
    console.warn("Health check failed:", err);
  }
}

async function loadDomains() {
  const raw = await fetchJson(`${API_BASE}/domains`);
  // /domains returns string[] of domain_ids
  const list = Array.isArray(raw) ? raw : raw.domains || [];
  domainsCache = list.map((d) => (typeof d === "string" ? d : d.domain_id || d.id));
  renderDomainButtons(domainsCache);
  if (domainsCache.length) {
    selectDomain(domainsCache[0]);
  }
}

async function ask() {
  const q = els.queryInput.value.trim();
  if (!selectedDomain) {
    showError("Select a domain first.");
    return;
  }
  if (!q) {
    showError("Enter a question.");
    return;
  }

  els.askBtn.disabled = true;
  showLoadingViz("numeric");

  try {
    const url = `${API_BASE}/ask/${encodeURIComponent(selectedDomain)}?q=${encodeURIComponent(q)}&top_k=3`;
    const data = await fetchJson(url);
    if (data.retrieval_mode === "semantic") {
      // briefly show wave if we guessed wrong during loading
    }
    showResult(data);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    els.askBtn.disabled = false;
  }
}

// ---- I2 assistant panel (POST /assistant): trust decision, forecast interval, diagnosis ----
const asst = {
  input: document.getElementById("assistant-input"),
  config: document.getElementById("assistant-config"),
  asset: document.getElementById("assistant-asset"),
  origin: document.getElementById("assistant-origin"),
  confidence: document.getElementById("assistant-confidence"),
  event: document.getElementById("assistant-event"),
  btn: document.getElementById("assistant-btn"),
  result: document.getElementById("assistant-result"),
  trustCard: document.getElementById("trust-card"),
  trustNone: document.getElementById("trust-none"),
  badge: document.getElementById("decision-badge"),
  score: document.getElementById("trust-score"),
  components: document.getElementById("trust-components"),
  reasons: document.getElementById("trust-reasons"),
  chart: document.getElementById("forecast-chart"),
  diag: document.getElementById("diagnosis-card"),
  text: document.getElementById("assistant-text"),
  meta: document.getElementById("assistant-meta"),
  error: document.getElementById("assistant-error"),
};

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function renderTrust(trust) {
  if (!trust) {
    asst.trustCard.classList.add("is-hidden");
    asst.trustNone.classList.remove("is-hidden");
    return;
  }
  asst.trustNone.classList.add("is-hidden");
  asst.trustCard.classList.remove("is-hidden");
  asst.badge.className = `decision-badge is-${trust.decision}`;
  asst.badge.textContent = trust.decision;
  asst.score.textContent = Number(trust.trust_score).toFixed(3);
  asst.components.innerHTML = ["E", "R", "C", "S"]
    .map((k) => {
      const v = Math.max(0, Math.min(1, Number(trust.components[k] || 0)));
      const w = trust.weights && trust.weights[k] !== undefined ? ` w${Number(trust.weights[k]).toFixed(2)}` : "";
      return `<div class="trust-bar" title="${k}${w}"><span>${k}</span>
        <div class="trust-bar-track"><div class="trust-bar-fill" style="width:${(v * 100).toFixed(1)}%"></div></div>
        <span>${v.toFixed(3)}</span></div>`;
    })
    .join("");
  asst.reasons.innerHTML = (trust.reasons || []).map((r) => `<li>${escapeHtml(r)}</li>`).join("");
}

function renderForecastChart(fc, observed) {
  if (!fc) {
    asst.chart.classList.add("is-hidden");
    asst.chart.innerHTML = "";
    return;
  }
  const values = [fc.lower_bound, fc.prediction, fc.upper_bound].concat(
    typeof observed === "number" ? [observed] : []
  );
  const max = Math.max(...values, 1) * 1.1;
  const x = (v) => 20 + (360 * v) / max;
  const obs = typeof observed === "number"
    ? `<circle class="fc-obs" cx="${x(observed).toFixed(1)}" cy="30" r="4"><title>observed ${observed.toFixed(0)} kW</title></circle>`
    : "";
  asst.chart.innerHTML = `
    <svg viewBox="0 0 400 70" role="img" aria-label="Forecast interval">
      <line class="fc-axis" x1="20" y1="52" x2="380" y2="52" />
      <rect class="fc-interval" x="${x(fc.lower_bound).toFixed(1)}" y="16"
        width="${Math.max(1, x(fc.upper_bound) - x(fc.lower_bound)).toFixed(1)}" height="28" />
      <line class="fc-edge" x1="${x(fc.lower_bound).toFixed(1)}" y1="16" x2="${x(fc.lower_bound).toFixed(1)}" y2="44" />
      <line class="fc-edge" x1="${x(fc.upper_bound).toFixed(1)}" y1="16" x2="${x(fc.upper_bound).toFixed(1)}" y2="44" />
      <line class="fc-pred" x1="${x(fc.prediction).toFixed(1)}" y1="12" x2="${x(fc.prediction).toFixed(1)}" y2="48" />
      ${obs}
      <text class="fc-label" x="20" y="66">0 kW</text>
      <text class="fc-label" x="380" y="66" text-anchor="end">${max.toFixed(0)} kW</text>
      <text class="fc-label" x="${x(fc.prediction).toFixed(1)}" y="9" text-anchor="middle">${fc.prediction.toFixed(0)}</text>
    </svg>
    <p class="hint">${Math.round(fc.confidence * 100)}% interval [${fc.lower_bound.toFixed(0)}, ${fc.upper_bound.toFixed(0)}] kW
      (nominal coverage, not a probability that the point forecast is right)</p>`;
  asst.chart.classList.remove("is-hidden");
}

function renderDiagnosis(out, handledBy) {
  if (handledBy !== "fault_diagnosis") {
    asst.diag.classList.add("is-hidden");
    return;
  }
  const sev = out.severity || "undetermined";
  const conf = out.confidence ? out.confidence.level : "-";
  asst.diag.innerHTML = `<strong>${escapeHtml(out.diagnosis || "")}</strong>
    <span class="severity-chip severity-${escapeHtml(sev)}">severity ${escapeHtml(sev)}</span>
    <span class="severity-chip">confidence ${escapeHtml(conf)}</span>`;
  asst.diag.classList.remove("is-hidden");
}

async function runAssistant() {
  const body = { message: asst.input.value.trim(), config: asst.config.value };
  if (asst.origin.value.trim()) {
    body.origin = asst.origin.value.trim();
    body.asset = asst.asset.value;
    body.confidence = Number(asst.confidence.value);
  }
  if (asst.event.value.trim()) {
    body.event_id = Number(asst.event.value);
  }
  asst.btn.disabled = true;
  asst.result.classList.remove("is-hidden");
  asst.error.classList.add("is-hidden");
  asst.text.innerHTML = `<p class="answer-lede" style="color:var(--text-muted)">Running…</p>`;
  try {
    const data = await postJson(`${API_BASE}/assistant`, body);
    renderTrust(data.trust);
    renderForecastChart(data.forecast, data.output ? data.output.observed_kw : null);
    renderDiagnosis(data.output || {}, data.handled_by);
    asst.text.innerHTML = renderAnswerHtml(data.text);
    asst.meta.textContent = `config ${data.config} · intent ${data.intent} · handled by ${data.handled_by}`;
  } catch (err) {
    asst.text.innerHTML = "";
    asst.trustCard.classList.add("is-hidden");
    asst.chart.classList.add("is-hidden");
    asst.diag.classList.add("is-hidden");
    asst.error.textContent = err.message || String(err);
    asst.error.classList.remove("is-hidden");
  } finally {
    asst.btn.disabled = false;
  }
}

asst.btn.addEventListener("click", runAssistant);

els.askBtn.addEventListener("click", ask);
els.queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});

(async function boot() {
  await initStatus();
  try {
    await loadDomains();
  } catch (err) {
    setStatus(false, "OFFLINE");
    showError(`Could not load domains: ${err.message}`);
  }
})();
