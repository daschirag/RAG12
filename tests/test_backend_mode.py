"""BACKEND_MODE local/azure abstraction for vector stores and generators."""

from __future__ import annotations

import pytest

from src.backend.retrieval import get_vector_store
from src.backend.retrieval.local_store import LocalVectorStore
from src.backend.serving import get_generator
from src.backend.serving.generator import AzureGenerator, LocalGenerator

AZURE_KEYS = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_API_KEY",
    "AZURE_SEARCH_INDEX",
)


@pytest.mark.parametrize("value", ["local", "LOCAL", " local ", ""])
def test_local_mode_selects_local_backends(monkeypatch, value):
    monkeypatch.setenv("BACKEND_MODE", value)
    assert isinstance(get_vector_store(), LocalVectorStore)
    assert isinstance(get_generator(), LocalGenerator)


def test_unset_mode_defaults_to_local(monkeypatch):
    monkeypatch.delenv("BACKEND_MODE", raising=False)
    assert isinstance(get_vector_store(), LocalVectorStore)
    assert isinstance(get_generator(), LocalGenerator)


def test_unknown_mode_raises(monkeypatch):
    monkeypatch.setenv("BACKEND_MODE", "gcp")
    with pytest.raises(RuntimeError, match="Unknown BACKEND_MODE"):
        get_vector_store()
    with pytest.raises(RuntimeError, match="Unknown BACKEND_MODE"):
        get_generator()


def test_azure_mode_without_keys_fails_clearly(monkeypatch):
    monkeypatch.setenv("BACKEND_MODE", "azure")
    for key in AZURE_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="AZURE_"):
        get_vector_store()


def test_azure_mode_with_keys_selects_azure_backends(monkeypatch):
    from src.backend.retrieval.azure_store import AzureVectorStore

    monkeypatch.setenv("BACKEND_MODE", "azure")
    for key in AZURE_KEYS:
        monkeypatch.setenv(key, "https://example.invalid" if "ENDPOINT" in key else "x")
    # Construction only; no network calls are made until search/ingest.
    assert isinstance(get_vector_store(), AzureVectorStore)
    assert isinstance(get_generator(), AzureGenerator)


def test_cache_dirs_pinned(monkeypatch, tmp_path):
    import os

    from src.backend.runtime_env import configure_cache_dirs

    monkeypatch.setenv("GREENRAG_CACHE_DIR", str(tmp_path))
    # setenv first so monkeypatch restores both vars after the test.
    monkeypatch.setenv("HF_HOME", "somewhere_else")
    monkeypatch.setenv("TORCH_HOME", "somewhere_else")
    root = configure_cache_dirs()
    assert root == tmp_path
    assert os.environ["HF_HOME"] == str(tmp_path / "huggingface")
    assert os.environ["TORCH_HOME"] == str(tmp_path / "torch")
