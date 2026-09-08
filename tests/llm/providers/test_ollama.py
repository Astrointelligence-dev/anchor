"""Tests for OllamaProvider — thin OpenAI-compatible subclass for local Ollama models.

Tests cover:
- provider_name attribute
- default base_url points to local Ollama endpoint
- default api_key is "ollama" (Ollama doesn't require a real key)
- _resolve_api_key reads OLLAMA_API_KEY from env, falls back to "ollama"
- model_id format: "ollama/<model>"
- Provider is registered in the registry
"""

from __future__ import annotations

import os
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_provider():
    from anchor.llm.providers.ollama import OllamaProvider
    return OllamaProvider


def _make_provider(**kwargs):
    cls = _import_provider()
    defaults = {"model": "llama3", "max_retries": 0}
    defaults.update(kwargs)
    return cls(**defaults)


# ---------------------------------------------------------------------------
# Test: provider_name
# ---------------------------------------------------------------------------

class TestProviderName:
    def test_provider_name_is_ollama(self):
        provider = _make_provider()
        assert provider.provider_name == "ollama"


# ---------------------------------------------------------------------------
# Test: base_url
# ---------------------------------------------------------------------------

class TestBaseUrl:
    def test_default_base_url_is_local_ollama(self):
        provider = _make_provider()
        assert provider._base_url == "http://localhost:11434/v1"

    def test_custom_base_url_overrides_default(self):
        provider = _make_provider(base_url="http://192.168.1.10:11434/v1")
        assert provider._base_url == "http://192.168.1.10:11434/v1"


# ---------------------------------------------------------------------------
# Test: _resolve_api_key
# ---------------------------------------------------------------------------

class TestResolveApiKey:
    def test_defaults_to_ollama_string(self):
        cls = _import_provider()
        env = {k: v for k, v in os.environ.items() if k != "OLLAMA_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            provider = cls(model="llama3", max_retries=0)
        assert provider._api_key == "ollama"

    def test_reads_ollama_api_key_from_env(self):
        cls = _import_provider()
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "custom-key"}, clear=False):
            provider = cls(model="llama3", max_retries=0)
        assert provider._api_key == "custom-key"

    def test_explicit_api_key_overrides_env(self):
        cls = _import_provider()
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "env-key"}, clear=False):
            provider = cls(model="llama3", api_key="explicit-key", max_retries=0)
        assert provider._api_key == "explicit-key"


# ---------------------------------------------------------------------------
# Test: model_id
# ---------------------------------------------------------------------------

class TestModelId:
    def test_model_id_format(self):
        provider = _make_provider(model="llama3")
        assert provider.model_id == "ollama/llama3"

    def test_model_id_with_other_model(self):
        provider = _make_provider(model="mistral:7b")
        assert provider.model_id == "ollama/mistral:7b"


# ---------------------------------------------------------------------------
# Test: registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_ollama_is_registered(self):
        from anchor.llm.registry import _PROVIDERS
        # Import the provider module to trigger registration
        _import_provider()
        assert "ollama" in _PROVIDERS
        assert _PROVIDERS["ollama"] is _import_provider()
