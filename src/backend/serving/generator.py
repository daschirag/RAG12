"""LLM answer generation for GreenRAG (local Ollama or Azure OpenAI)."""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from ..config.domain_manifest import DomainManifest

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"

_AZURE_OPENAI_REQUIRED = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
)

# Appended after the manifest template for semantic /ask answers.
_GROUNDEDNESS_SUFFIX = """
Additional instructions (mandatory):
- Answer ONLY using the Context above.
- If the Context does not contain enough information to answer, reply exactly:
  "I don't have enough information"
- Report values exactly as given in the Context. Do not fabricate or guess values that are not present in the Context.
- Do NOT use outside knowledge.
""".strip()

_SEMANTIC_FORMAT_SUFFIX = """
Response format (semantic answers):
- First line: one direct plain-language sentence that answers the question.
- Then, only if useful, up to 3 supporting points, each on its own line starting with "- ".
- No nested bullets, no headers, no markdown beyond optional **bold**.
""".strip()

# Inserted only when generate_answer is called with retrieval_mode=numeric
# (legacy path); numeric /ask now uses a separate enrichment prompt.
_NUMERIC_RANKING_HINT = (
    "The context below has already been sorted by the requested metric - "
    "the first entry represents the highest (or lowest, per the question) value found."
)


class Generator(ABC):
    """Pluggable LLM backend for grounded RAG answers."""

    @abstractmethod
    def generate_answer(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        domain_manifest: DomainManifest,
        retrieval_mode: str = "semantic",
    ) -> str:
        """Generate an answer from ``query`` + retrieved ``context_chunks``."""

    @abstractmethod
    def enrich_numeric_fact(
        self,
        deterministic_answer: str,
        context_chunks: list[dict[str, Any]],
    ) -> str:
        """Optional one-sentence interpretation for a fixed numeric fact.

        Returns an empty string on failure so callers can keep the fact alone.
        """


def _format_context(context_chunks: list[dict[str, Any]]) -> str:
    """Join chunk texts into a numbered context block for the prompt."""
    parts: list[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        text = str(chunk.get("text", "")).strip()
        parts.append(f"[{i}] ({chunk_id}) {text}")
    return "\n\n".join(parts) if parts else "(no context)"


def build_prompt(
    query: str,
    context_chunks: list[dict[str, Any]],
    domain_manifest: DomainManifest,
    retrieval_mode: str = "semantic",
) -> str:
    """Fill ``domain_manifest.prompt_template`` and append groundedness rules."""
    context = _format_context(context_chunks)
    template = domain_manifest.prompt_template
    try:
        body = template.format(context=context, question=query)
    except KeyError:
        # Templates may use {context}/{question} only; fall back safely.
        body = (
            f"{template}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )

    if retrieval_mode == "numeric":
        marker = "Context:"
        if marker in body:
            body = body.replace(marker, f"{_NUMERIC_RANKING_HINT}\n\n{marker}", 1)
        else:
            body = f"{_NUMERIC_RANKING_HINT}\n\n{body}"

    parts = [body.rstrip(), _GROUNDEDNESS_SUFFIX]
    if retrieval_mode == "semantic":
        parts.append(_SEMANTIC_FORMAT_SUFFIX)
    return "\n\n".join(parts) + "\n"


def _enrichment_is_consistent(enrichment: str, deterministic_answer: str) -> bool:
    """Reject enrichment that introduces numbers not already in the fact sentence."""
    text = (enrichment or "").strip()
    if not text or len(text) > 400:
        return False
    for match in re.findall(r"-?\d+(?:\.\d+)?", text):
        if match not in deterministic_answer:
            return False
    return True


def build_numeric_enrichment_prompt(
    deterministic_answer: str,
    context_chunks: list[dict[str, Any]],
) -> str:
    """Low-stakes prompt: one interpretive sentence; do not alter the number."""
    context = _format_context(context_chunks)
    return (
        f"Given this fact: {deterministic_answer}\n\n"
        f"Surrounding context:\n{context}\n\n"
        "Add exactly one plain sentence of context or interpretation using the "
        "surrounding chunks. Do not use bullets, lists, or headers. Do not change "
        "or re-state the number differently. Reply with only that one sentence."
    )


def _require_env(name: str) -> str:
    """Return a non-empty env var or raise a clear RuntimeError."""
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required Azure configuration: {name} is unset or empty. "
            f"Set it in your .env (see .env.example). "
            f"BACKEND_MODE=azure needs: {', '.join(_AZURE_OPENAI_REQUIRED)}."
        )
    return value


class LocalGenerator(Generator):
    """Local answers via the Ollama Python client."""

    def __init__(self, model: str | None = None) -> None:
        self.model = (
            model
            or (os.getenv("OLLAMA_MODEL") or "").strip()
            or DEFAULT_OLLAMA_MODEL
        )
        logger.info("LocalGenerator using Ollama model: %s", self.model)

    def generate_answer(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        domain_manifest: DomainManifest,
        retrieval_mode: str = "semantic",
    ) -> str:
        import ollama

        prompt = build_prompt(
            query, context_chunks, domain_manifest, retrieval_mode=retrieval_mode
        )
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        # ollama>=0.3 returns an object or dict depending on version.
        message = getattr(response, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content:
                return str(content).strip()
        if isinstance(response, dict):
            return str(response.get("message", {}).get("content", "")).strip()
        return str(response).strip()

    def enrich_numeric_fact(
        self,
        deterministic_answer: str,
        context_chunks: list[dict[str, Any]],
    ) -> str:
        """Ask Ollama for one interpretive sentence; empty string on failure."""
        import ollama

        prompt = build_numeric_enrichment_prompt(deterministic_answer, context_chunks)
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Numeric enrichment LLM call failed: %s", exc)
            return ""

        message = getattr(response, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            text = str(content).strip() if content else ""
        elif isinstance(response, dict):
            text = str(response.get("message", {}).get("content", "")).strip()
        else:
            text = str(response).strip()

        if not _enrichment_is_consistent(text, deterministic_answer):
            logger.warning("Discarding inconsistent numeric enrichment: %r", text)
            return ""
        return text


class AzureGenerator(Generator):
    """Production answers via Azure OpenAI chat completions."""

    def __init__(self) -> None:
        self.endpoint = _require_env("AZURE_OPENAI_ENDPOINT")
        self.api_key = _require_env("AZURE_OPENAI_API_KEY")
        self.deployment = _require_env("AZURE_OPENAI_DEPLOYMENT")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                api_key=self.api_key,
                api_version="2024-02-01",
                azure_endpoint=self.endpoint,
            )
        return self._client

    def generate_answer(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        domain_manifest: DomainManifest,
        retrieval_mode: str = "semantic",
    ) -> str:
        prompt = build_prompt(
            query, context_chunks, domain_manifest, retrieval_mode=retrieval_mode
        )
        client = self._get_client()
        completion = client.chat.completions.create(
            model=self.deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are GreenRAG, a grounded renewable-energy assistant. "
                        "Follow the user prompt instructions exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        content = completion.choices[0].message.content or ""
        return content.strip()

    def enrich_numeric_fact(
        self,
        deterministic_answer: str,
        context_chunks: list[dict[str, Any]],
    ) -> str:
        """Ask Azure OpenAI for one interpretive sentence; empty string on failure."""
        prompt = build_numeric_enrichment_prompt(deterministic_answer, context_chunks)
        try:
            client = self._get_client()
            completion = client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You add brief grounded interpretation only. "
                            "Never change numbers from the given fact."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            text = (completion.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Numeric enrichment LLM call failed: %s", exc)
            return ""

        if not _enrichment_is_consistent(text, deterministic_answer):
            logger.warning("Discarding inconsistent numeric enrichment: %r", text)
            return ""
        return text


def get_generator() -> Generator:
    """Return LocalGenerator or AzureGenerator based on ``BACKEND_MODE``."""
    mode = (os.getenv("BACKEND_MODE") or "local").strip().lower()

    if mode == "local":
        logger.info("Answer generator backend: local (Ollama)")
        return LocalGenerator()

    if mode == "azure":
        logger.info("Answer generator backend: azure (Azure OpenAI)")
        return AzureGenerator()

    raise RuntimeError(
        f"Unknown BACKEND_MODE={mode!r}. Expected 'local' or 'azure'."
    )
