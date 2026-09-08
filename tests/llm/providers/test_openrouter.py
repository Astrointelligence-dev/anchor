"""Tests for OpenRouterProvider — thin OpenAI-compatible subclass targeting OpenRouter.

Tests cover:
- provider_name attribute
- default base_url points to OpenRouter endpoint
- _resolve_api_key reads OPENROUTER_API_KEY from env
- model_id format: "openrouter/<model>"
- Provider is registered in the registry
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from anchor.llm.models import Message, Role

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_provider():
    from anchor.llm.providers.openrouter import OpenRouterProvider
    return OpenRouterProvider


def _make_provider(**kwargs):
    cls = _import_provider()
    defaults = {"model": "openai/gpt-4o", "api_key": "test-key", "max_retries": 0}
    defaults.update(kwargs)
    return cls(**defaults)


# ---------------------------------------------------------------------------
# Test: provider_name
# ---------------------------------------------------------------------------

class TestProviderName:
    def test_provider_name_is_openrouter(self):
        provider = _make_provider()
        assert provider.provider_name == "openrouter"


# ---------------------------------------------------------------------------
# Test: base_url
# ---------------------------------------------------------------------------

class TestBaseUrl:
    def test_default_base_url_is_openrouter(self):
        provider = _make_provider()
        assert provider._base_url == "https://openrouter.ai/api/v1"

    def test_custom_base_url_overrides_default(self):
        provider = _make_provider(base_url="https://custom.example.com/v1")
        assert provider._base_url == "https://custom.example.com/v1"


# ---------------------------------------------------------------------------
# Test: _resolve_api_key
# ---------------------------------------------------------------------------

class TestResolveApiKey:
    def test_reads_openrouter_api_key(self):
        cls = _import_provider()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-secret"}, clear=False):
            provider = cls(model="openai/gpt-4o", max_retries=0)
        assert provider._api_key == "or-secret"

    def test_returns_none_when_no_key_in_env(self):
        cls = _import_provider()
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            provider = cls(model="openai/gpt-4o", max_retries=0)
        assert provider._api_key is None

    def test_explicit_api_key_overrides_env(self):
        cls = _import_provider()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}, clear=False):
            provider = cls(model="openai/gpt-4o", api_key="explicit-key", max_retries=0)
        assert provider._api_key == "explicit-key"


# ---------------------------------------------------------------------------
# Test: model_id
# ---------------------------------------------------------------------------

class TestModelId:
    def test_model_id_format(self):
        provider = _make_provider(model="openai/gpt-4o")
        assert provider.model_id == "openrouter/openai/gpt-4o"

    def test_model_id_with_anthropic_model(self):
        provider = _make_provider(model="anthropic/claude-3-5-sonnet")
        assert provider.model_id == "openrouter/anthropic/claude-3-5-sonnet"


# ---------------------------------------------------------------------------
# Test: registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_openrouter_is_registered(self):
        from anchor.llm.registry import _PROVIDERS
        # Import the provider module to trigger registration
        _import_provider()
        assert "openrouter" in _PROVIDERS
        assert _PROVIDERS["openrouter"] is _import_provider()


# ---------------------------------------------------------------------------
# Test: billed cost (usage.include) and extra_body merge
# ---------------------------------------------------------------------------


def _usage_mock(cost=None):
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    usage.cost = cost
    usage.prompt_tokens_details.cached_tokens = None
    return usage


def _sdk_response(cost=None):
    msg = MagicMock()
    msg.content = "ok"
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = _usage_mock(cost)
    resp.model = "openai/gpt-4o"
    return resp


class TestUsageAndCost:
    @patch("anchor.llm.providers.openai.openai")
    def test_usage_include_is_requested_by_default(self, mock_openai):
        provider = _make_provider()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _sdk_response()

        provider._do_invoke([Message(role=Role.USER, content="Hi")], tools=None)
        sent = mock_client.chat.completions.create.call_args.kwargs["extra_body"]
        assert sent == {"usage": {"include": True}}

    @patch("anchor.llm.providers.openai.openai")
    def test_caller_extra_body_is_merged_and_wins(self, mock_openai):
        provider = _make_provider(
            extra_body={"provider": {"order": ["Google AI Studio"]}, "usage": {"include": False}},
        )
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value = _sdk_response()

        provider._do_invoke([Message(role=Role.USER, content="Hi")], tools=None)
        sent = mock_client.chat.completions.create.call_args.kwargs["extra_body"]
        assert sent == {"provider": {"order": ["Google AI Studio"]}, "usage": {"include": False}}

    def test_billed_cost_lands_in_total_cost(self):
        provider = _make_provider()
        out = provider._parse_response(_sdk_response(cost=0.0042))
        assert out.usage.total_cost == 0.0042
        assert out.provider == "openrouter"
