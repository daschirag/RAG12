"""Console UI contract: static assets, element ids, design tokens, API routes."""

from __future__ import annotations

import re
from pathlib import Path

from src.backend import main

FRONTEND = Path(__file__).resolve().parents[1] / "src" / "frontend"

# Design tokens present at Phase 0; later phases may add tokens, never remove these.
PHASE0_TOKENS = {
    "--accent-fault", "--accent-ok", "--accent-solar", "--accent-wind", "--bg",
    "--border", "--border-strong", "--domain-accent", "--font-body",
    "--font-display", "--font-mono", "--panel", "--panel-elevated", "--radius",
    "--shadow-inset", "--shadow-panel", "--text", "--text-faint", "--text-muted",
}

REQUIRED_IDS = {
    "status-pill", "status-label", "domain-switches", "query-input", "ask-btn",
    "readout", "empty-state", "result-view", "answer-body", "sources-list",
    "error-state",
}


def _read(name: str) -> str:
    return (FRONTEND / name).read_text(encoding="utf-8")


def test_index_links_assets_and_ids():
    html = _read("index.html")
    assert './styles.css' in html
    assert './app.js' in html
    ids = set(re.findall(r'id="([^"]+)"', html))
    assert REQUIRED_IDS <= ids


def test_design_tokens_preserved():
    css = _read("styles.css")
    tokens = set(re.findall(r"(--[a-z0-9-]+)\s*:", css))
    assert PHASE0_TOKENS <= tokens


def test_app_js_calls_existing_routes():
    js = _read("app.js")
    paths = {route.path for route in main.app.routes}
    for used, route in (
        ("/health", "/health"),
        ("/domains", "/domains"),
        ("/ask/", "/ask/{domain_id}"),
    ):
        assert f"${{API_BASE}}{used}" in js
        assert route in paths


ASSISTANT_IDS = {
    "assistant-input", "assistant-config", "assistant-asset", "assistant-origin", "assistant-confidence",
    "assistant-event", "assistant-btn", "assistant-result", "trust-card", "decision-badge", "trust-score",
    "trust-components", "trust-reasons", "forecast-chart", "diagnosis-card", "assistant-text",
}


def test_assistant_panel_shows_trust_decision_and_forecast_interval():
    """I2: the console posts to /assistant and renders the decision, trust score, components and interval."""
    html, js = _read("index.html"), _read("app.js")
    assert ASSISTANT_IDS <= set(re.findall(r'id="([^"]+)"', html))
    assert "${API_BASE}/assistant" in js and "/assistant" in {r.path for r in main.app.routes}
    for field in ("trust.decision", "trust.trust_score", "trust.components", "fc.lower_bound", "fc.upper_bound",
                  "fc.prediction"):
        assert field in js
    for decision in ("accept", "review", "reject"):
        assert f".decision-badge.is-{decision}" in _read("styles.css")
    for cfg in "ABCDEF":
        assert f'<option value="{cfg}"' in html
