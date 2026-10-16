"""OpenAPI response examples: real responses recorded from the running system, never written by hand.

``python -m src.backend.api.capture_examples`` calls each endpoint against the real services (Ollama on) and
writes ``openapi_examples.json`` next to this file (with the capture time). The API then shows those recorded
responses as examples in /docs. Without the file the docs simply have no response examples.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

EXAMPLES_FILE = Path(__file__).with_name("openapi_examples.json")


@lru_cache(maxsize=1)
def _load() -> dict:
    if not EXAMPLES_FILE.is_file():
        return {}
    return json.loads(EXAMPLES_FILE.read_text(encoding="utf-8"))


def ex(key: str, status: int = 200) -> dict:
    """``responses=`` entry for one endpoint ("POST /forecast") from the recorded capture, or {}."""
    rec = _load().get("responses", {}).get(key)
    if not rec:
        return {}
    note = f"Recorded from the running system at {_load().get('captured_at_utc', '?')} (request: {rec['request']})."
    return {status: {"description": note, "content": {"application/json": {"example": rec["response"]}}}}
