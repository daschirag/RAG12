"""Secret lookup: environment (or the git-ignored .env) first, then Azure Key Vault. Values are never logged.

| logical name | environment variable | Key Vault secret |
|---|---|---|
| search_api_key | AZURE_SEARCH_API_KEY | search-admin-key |
| appinsights_connection_string | APPLICATIONINSIGHTS_CONNECTION_STRING | appinsights-connection-string |
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable

from ..settings import get_settings

logger = logging.getLogger(__name__)

SECRETS: dict[str, tuple[str, str]] = {
    "search_api_key": ("AZURE_SEARCH_API_KEY", "search-admin-key"),
    "appinsights_connection_string": ("APPLICATIONINSIGHTS_CONNECTION_STRING", "appinsights-connection-string"),
}

_cache: dict[str, str] = {}
_lock = threading.Lock()


class SecretUnavailableError(RuntimeError):
    pass


def _key_vault_client(vault_url: str):
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    return SecretClient(vault_url=vault_url, credential=DefaultAzureCredential(exclude_interactive_browser_credential=True))


# Tests replace this factory; production builds a SecretClient for the configured vault.
client_factory: Callable[[str], object] = _key_vault_client


def get_secret(name: str) -> str:
    """Value of a known secret. Raises SecretUnavailableError naming the variable and vault secret, never the value."""
    if name not in SECRETS:
        raise KeyError(f"unknown secret {name!r}; known: {sorted(SECRETS)}")
    env_name, vault_name = SECRETS[name]
    value = (os.getenv(env_name) or "").strip()
    if value:
        return value
    with _lock:
        if name in _cache:
            return _cache[name]
        vault = get_settings().azure_key_vault_url
        if not vault:
            raise SecretUnavailableError(
                f"Missing required Azure configuration: set {env_name} in .env or AZURE_KEY_VAULT_URL "
                f"(Key Vault secret '{vault_name}').")
        try:
            value = client_factory(vault).get_secret(vault_name).value or ""
        except Exception as exc:  # noqa: BLE001 - message must not carry the secret; type and vault name only
            raise SecretUnavailableError(f"Key Vault secret '{vault_name}' could not be read from {vault} "
                                         f"({type(exc).__name__}); is 'az login' done and the role assigned?") from None
        if not value:
            raise SecretUnavailableError(f"Key Vault secret '{vault_name}' is empty")
        _cache[name] = value
        logger.info("secret %s loaded from Key Vault", vault_name)      # the name only
        return value


def clear_cache() -> None:
    with _lock:
        _cache.clear()
