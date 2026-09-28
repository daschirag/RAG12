"""LocalGenerator retries once when Ollama's runner crashes during model load."""

from __future__ import annotations

import ollama
import pytest

from src.backend.config import load_manifest
from src.backend.serving.generator import LocalGenerator

CRASH = ollama.ResponseError(
    "llama-server process has terminated: exit status 0xc0000409: "
    "CUDA error: shared object initialization failed",
    500,
)
CHUNKS = [{"chunk_id": "c1", "text": "Inverter B produced 612.10 kW."}]


class ScriptedChat:
    """Stand-in for ollama.chat that raises/returns items in order."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, model, messages):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return {"message": {"content": outcome}}


def _answer(monkeypatch, chat):
    monkeypatch.setattr(ollama, "chat", chat)
    return LocalGenerator(model="m").generate_answer("q", CHUNKS, load_manifest("solar"))


def test_retries_once_after_runner_crash(monkeypatch):
    chat = ScriptedChat(CRASH, "612.10 kW")
    assert _answer(monkeypatch, chat) == "612.10 kW"
    assert chat.calls == 2


def test_second_crash_propagates(monkeypatch):
    chat = ScriptedChat(CRASH, CRASH)
    with pytest.raises(ollama.ResponseError, match="llama-server"):
        _answer(monkeypatch, chat)
    assert chat.calls == 2


def test_other_errors_are_not_retried(monkeypatch):
    chat = ScriptedChat(ollama.ResponseError("model 'm' not found", 404))
    with pytest.raises(ollama.ResponseError, match="not found"):
        _answer(monkeypatch, chat)
    assert chat.calls == 1


def test_enrichment_retries_then_degrades_to_empty(monkeypatch):
    monkeypatch.setattr(ollama, "chat", ScriptedChat(CRASH, CRASH))
    assert LocalGenerator(model="m").enrich_numeric_fact("Value 612.10.", CHUNKS) == ""
