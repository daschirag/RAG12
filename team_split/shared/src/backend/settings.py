"""Single settings class: every runtime option comes from an environment variable (or the project .env).

| variable | default | meaning |
|---|---|---|
| BACKEND_MODE | local | local (FastAPI + ChromaDB + Ollama) or azure (plan A, docs/azure.md; I4 comparison pending) |
| OLLAMA_MODEL | qwen2.5:7b-instruct | generator and judge model served by Ollama |
| OLLAMA_HOST | http://127.0.0.1:11434 | Ollama base URL (readiness check and warm-up) |
| GREENRAG_CONSOLE_ORIGINS | http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5500,http://localhost:5500 | CORS allow-list (comma separated) |
| GREENRAG_LLM_TIMEOUT_S | 120 | per-call Ollama timeout (one retry, then the evidence-only fallback) |
| GREENRAG_LOG_LEVEL | INFO | log level of the structured request log |
| GREENRAG_CACHE_DIR | D:\\Rag\\cache | HF / torch model cache root (runtime_env.configure_cache_dirs) |
| EMBEDDING_PROVIDER | auto | azure mode only: local (MiniLM) or azure_openai; auto = azure_openai iff AZURE_OPENAI_ENDPOINT is set |
| GENERATOR_PROVIDER | auto | azure mode only: ollama or azure_openai; auto = azure_openai iff AZURE_OPENAI_DEPLOYMENT is set |
| AZURE_KEY_VAULT_URL | (unset) | Key Vault for secrets (search admin key, App Insights connection string); Entra login, no keys |
| AZURE_STORAGE_ACCOUNT_URL | (unset) | Blob account URL (Entra login; shared keys are disabled on the account) |
| AZURE_KB_CONTAINER / AZURE_RESULTS_CONTAINER | kbdocs / results | blob containers |
| AZURE_SEARCH_ENDPOINT / AZURE_SEARCH_INDEX | (unset) | AI Search endpoint and the one shared index (domains are a filter) |
| KB_SOURCE | local | local or blob: where the secure pipeline reads documents.jsonl |
| GREENRAG_APPINSIGHTS | false | true: send the request log to Application Insights |

Secrets (GREENRAG_TRUST_KEY, AZURE_SEARCH_API_KEY, the App Insights connection string, Azure OpenAI keys) are not
settings: ``src/backend/cloud/secrets.py`` reads them from the environment or Key Vault when needed, and they are
never logged, returned or written to a file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINS = ("http://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1:5500", "http://localhost:5500")
REQUIRED_OLLAMA_MODELS = ("qwen2.5:7b-instruct",)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore",
                                      populate_by_name=True)

    backend_mode: str = Field("local", alias="BACKEND_MODE")
    ollama_model: str = Field("qwen2.5:7b-instruct", alias="OLLAMA_MODEL")
    ollama_host: str = Field("http://127.0.0.1:11434", alias="OLLAMA_HOST")
    console_origins: str = Field(",".join(DEFAULT_ORIGINS), alias="GREENRAG_CONSOLE_ORIGINS")
    llm_timeout_s: float = Field(120.0, alias="GREENRAG_LLM_TIMEOUT_S", gt=0)
    log_level: str = Field("INFO", alias="GREENRAG_LOG_LEVEL")
    embedding_provider: str = Field("auto", alias="EMBEDDING_PROVIDER")
    generator_provider: str = Field("auto", alias="GENERATOR_PROVIDER")
    azure_key_vault_url: str = Field("", alias="AZURE_KEY_VAULT_URL")
    azure_storage_account_url: str = Field("", alias="AZURE_STORAGE_ACCOUNT_URL")
    azure_kb_container: str = Field("kbdocs", alias="AZURE_KB_CONTAINER")
    azure_results_container: str = Field("results", alias="AZURE_RESULTS_CONTAINER")
    azure_search_endpoint: str = Field("", alias="AZURE_SEARCH_ENDPOINT")
    azure_search_index: str = Field("", alias="AZURE_SEARCH_INDEX")
    kb_source: str = Field("local", alias="KB_SOURCE")
    appinsights_enabled: bool = Field(False, alias="GREENRAG_APPINSIGHTS")

    @field_validator("embedding_provider", "generator_provider", "kb_source", mode="before")
    @classmethod
    def _choice(cls, v: object, info) -> str:
        allowed = {"embedding_provider": ("auto", "local", "azure_openai"),
                   "generator_provider": ("auto", "ollama", "azure_openai"),
                   "kb_source": ("local", "blob")}[info.field_name]
        v = str(v or allowed[0]).strip().lower()
        if v not in allowed:
            raise ValueError(f"{info.field_name} must be one of {allowed}, got {v!r}")
        return v

    @field_validator("azure_key_vault_url", "azure_storage_account_url", "azure_search_endpoint", "azure_search_index",
                     mode="before")
    @classmethod
    def _url(cls, v: object) -> str:
        return str(v or "").strip().rstrip("/")

    @field_validator("backend_mode", mode="before")
    @classmethod
    def _mode(cls, v: object) -> str:
        v = str(v or "local").strip().lower()
        return v if v in ("local", "azure") else "local"

    @field_validator("ollama_model", mode="before")
    @classmethod
    def _model(cls, v: object) -> str:
        return str(v or "").strip() or "qwen2.5:7b-instruct"

    @field_validator("ollama_host", mode="before")
    @classmethod
    def _host(cls, v: object) -> str:
        return (str(v or "").strip() or "http://127.0.0.1:11434").rstrip("/")

    @property
    def embeddings(self) -> str:
        """Resolved embedding provider for azure mode (plan A: local MiniLM unless Azure OpenAI is configured)."""
        if self.embedding_provider != "auto":
            return self.embedding_provider
        import os

        return "azure_openai" if (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip() else "local"

    @property
    def generator(self) -> str:
        """Resolved generator for azure mode (plan A: local Ollama unless an Azure OpenAI deployment is configured)."""
        if self.generator_provider != "auto":
            return self.generator_provider
        import os

        return "azure_openai" if (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip() else "ollama"

    @property
    def origins(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.console_origins.split(",") if o.strip()]


def get_settings() -> Settings:
    """Read the environment now (cheap; tests change variables between calls)."""
    return Settings()
