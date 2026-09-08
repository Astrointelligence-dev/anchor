"""Tests for GrokProvider — thin OpenAI-compatible subclass targeting xAI's API.

Tests cover:
- provider_name attribute
- default base_url points to xAI endpoint
- _resolve_api_key reads XAI_API_KEY and GROK_API_KEY from env
- model_id format: "grok/<model>"
- Provider is registered in the registry
"""

from __future__ import annotations

import os
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_provider():
    from anchor.llm.providers.grok import GrokProvider
    return GrokProvider


def _make_provider(**kwargs):
    cls = _import_provider()
    defaults = {"model": "grok-3", "api_key": "test-key", "max_retries": 0}
    defaults.update(kwargs)
    return cls(**defaults)


# ---------------------------------------------------------------------------
# Test: provider_name
# ---------------------------------------------------------------------------

class TestProviderName:
    def test_provider_name_is_grok(self):
        provider = _make_provider()
        assert provider.provider_name == "grok"


# ---------------------------------------------------------------------------
# Test: base_url
# ---------------------------------------------------------------------------

class TestBaseUrl:
    def test_default_base_url_is_xai(self):
        provider = _make_provider()
        assert provider._base_url == "https://api.x.ai/v1"

    def test_custom_base_url_overrides_default(self):
        provider = _make_provider(base_url="https://custom.example.com/v1")
        assert provider._base_url == "https://custom.example.com/v1"


# ---------------------------------------------------------------------------
# Test: _resolve_api_key
# ---------------------------------------------------------------------------

class TestResolveApiKey:
    def test_reads_xai_api_key(self):
        cls = _import_provider()
        with patch.dict(os.environ, {"XAI_API_KEY": "xai-secret"}, clear=False):
            provider = cls(model="grok-3", max_retries=0)
        assert provider._api_key == "xai-secret"

    def test_reads_grok_api_key_fallback(self):
        cls = _import_provider()
        env = {k: v for k, v in os.environ.items() if k not in ("XAI_API_KEY", "GROK_API_KEY")}
        env["GROK_API_KEY"] = "grok-secret"
        with patch.dict(os.environ, env, clear=True):
            provider = cls(model="grok-3", max_retries=0)
        assert provider._api_key == "grok-secret"

    def test_xai_api_key_takes_precedence_over_grok_api_key(self):
        cls = _import_provider()
        env = {k: v for k, v in os.environ.items() if k not in ("XAI_API_KEY", "GROK_API_KEY")}
        env["XAI_API_KEY"] = "xai-primary"
        env["GROK_API_KEY"] = "grok-fallback"
        with patch.dict(os.environ, env, clear=True):
            provider = cls(model="grok-3", max_retries=0)
        assert provider._api_key == "xai-primary"

    def test_returns_none_when_no_key_in_env(self):
        cls = _import_provider()
        env = {k: v for k, v in os.environ.items() if k not in ("XAI_API_KEY", "GROK_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            provider = cls(model="grok-3", max_retries=0)
        assert provider._api_key is None

    def test_explicit_api_key_overrides_env(self):
        cls = _import_provider()
        with patch.dict(os.environ, {"XAI_API_KEY": "env-key"}, clear=False):
            provider = cls(model="grok-3", api_key="explicit-key", max_retries=0)
        assert provider._api_key == "explicit-key"


# ---------------------------------------------------------------------------
# Test: model_id
# ---------------------------------------------------------------------------

class TestModelId:
    def test_model_id_format(self):
        provider = _make_provider(model="grok-3")
        assert provider.model_id == "grok/grok-3"

    def test_model_id_with_other_model(self):
        provider = _make_provider(model="grok-2-vision")
        assert provider.model_id == "grok/grok-2-vision"


# ---------------------------------------------------------------------------
# Test: registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_grok_is_registered(self):
        from anchor.llm.registry import _PROVIDERS
        # Import the provider module to trigger registration
        _import_provider()
        assert "grok" in _PROVIDERS
        assert _PROVIDERS["grok"] is _import_provider()
