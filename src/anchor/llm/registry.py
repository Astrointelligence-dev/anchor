"""Provider registry and model string parsing.

Manages the mapping from provider name to provider class, handles
lazy imports, and provides the create_provider() factory function.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any

from anchor.llm.base import BaseLLMProvider, LLMProvider
from anchor.llm.errors import ProviderNotInstalledError

# RLock, not Lock: create_provider holds this across _try_import_provider(),
# and the imported module registers itself via register_provider() — which
# re-acquires it on the same thread.
_LOCK = threading.RLock()
_PROVIDERS: dict[str, type[BaseLLMProvider]] = {}

# Maps provider name -> module path for lazy loading
_PROVIDER_MODULES: dict[str, str] = {
    "anthropic": "anchor.llm.providers.anthropic",
    "claude_cli": "anchor.llm.providers.claude_cli",
    "openai": "anchor.llm.providers.openai",
    "gemini": "anchor.llm.providers.gemini",
    "grok": "anchor.llm.providers.grok",
    "ollama": "anchor.llm.providers.ollama",
    "openrouter": "anchor.llm.providers.openrouter",
    "litellm": "anchor.llm.providers.litellm",
}

# Maps provider name -> pip package name (for error messages)
_PROVIDER_PACKAGES: dict[str, str] = {
    "anthropic": "anthropic",
    "claude_cli": "claude-agent-sdk",
    "openai": "openai",
    "gemini": "google-genai",
    "grok": "openai",
    "ollama": "ollama",
    "openrouter": "openai",
    "litellm": "litellm",
}

# Maps provider name -> pip extras name
_PROVIDER_EXTRAS: dict[str, str] = {
    "anthropic": "anthropic",
    "claude_cli": "claude-cli",
    "openai": "openai",
    "gemini": "gemini",
    "grok": "openai",
    "ollama": "ollama",
    "openrouter": "openai",
    "litellm": "litellm",
}


def register_provider(name: str, cls: type[BaseLLMProvider]) -> None:
    """Register a provider adapter."""
    with _LOCK:
        _PROVIDERS[name] = cls


def create_provider(
    model: str,
    *,
    api_key: str | None = None,
    fallbacks: list[str] | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Create a provider from a 'provider/model' string.

    Examples:
        create_provider("openai/gpt-4o")
        create_provider("anthropic/claude-sonnet-4-20250514", api_key="sk-...")
        create_provider("ollama/llama3")
        create_provider("anthropic/claude-sonnet-4-20250514", fallbacks=["openai/gpt-4o"])
    """
    provider_name, model_name = _parse_model_string(model)

    with _LOCK:
        if provider_name not in _PROVIDERS:
            _try_import_provider(provider_name)

        if provider_name not in _PROVIDERS:
            package = _PROVIDER_PACKAGES.get(provider_name, provider_name)
            extra = _PROVIDER_EXTRAS.get(provider_name, provider_name)
            raise ProviderNotInstalledError(provider_name, package, extra)

        cls = _PROVIDERS[provider_name]
    primary = cls(model=model_name, api_key=api_key, **kwargs)

    if fallbacks:
        from anchor.llm.fallback import FallbackProvider

        fallback_providers = [create_provider(fb) for fb in fallbacks]
        return FallbackProvider(primary=primary, fallbacks=fallback_providers)

    return primary


def _parse_model_string(model: str) -> tuple[str, str]:
    """Parse 'provider/model' into (provider, model).

    No prefix defaults to 'anthropic' for backward compat.
    Splits on first '/' only.
    """
    if "/" not in model:
        return "anthropic", model
    provider, _, model_name = model.partition("/")
    return provider, model_name


def _try_import_provider(name: str) -> None:
    """Attempt to lazily import a provider module.

    Note: Caller must hold ``_LOCK`` before calling this function.
    """
    module_path = _PROVIDER_MODULES.get(name)
    if module_path:
        try:
            importlib.import_module(module_path)
        except ImportError:
            pass
