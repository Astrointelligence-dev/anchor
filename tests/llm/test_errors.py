"""Tests for anchor.llm.errors."""

from __future__ import annotations

import pytest

from anchor.llm.errors import (
    AuthenticationError,
    ContentFilterError,
    LLMTimeoutError,
    ModelNotFoundError,
    ProviderError,
    ProviderNotInstalledError,
    RateLimitError,
    ServerError,
)


class TestProviderError:
    def test_base_error(self):
        e = ProviderError("something failed", provider="openai")
        assert str(e) == "something failed"
        assert e.provider == "openai"
        assert e.is_transient is False

    def test_transient_error(self):
        e = ProviderError("retry me", provider="openai", is_transient=True)
        assert e.is_transient is True

    def test_is_exception(self):
        with pytest.raises(ProviderError):
            raise ProviderError("boom", provider="test")


class TestAuthenticationError:
    def test_not_transient(self):
        e = AuthenticationError("bad key", provider="anthropic")
        assert e.is_transient is False
        assert e.provider == "anthropic"

    def test_inherits_provider_error(self):
        e = AuthenticationError("bad key", provider="openai")
        assert isinstance(e, ProviderError)


class TestRateLimitError:
    def test_is_transient(self):
        e = RateLimitError("429", provider="openai")
        assert e.is_transient is True
        assert e.retry_after is None

    def test_with_retry_after(self):
        e = RateLimitError("429", provider="openai", retry_after=5.0)
        assert e.retry_after == 5.0


class TestServerError:
    def test_is_transient(self):
        e = ServerError("500", provider="gemini")
        assert e.is_transient is True


class TestTimeoutError:
    def test_is_transient(self):
        e = LLMTimeoutError("timed out", provider="ollama")
        assert e.is_transient is True


class TestModelNotFoundError:
    def test_not_transient(self):
        e = ModelNotFoundError("no such model", provider="openai")
        assert e.is_transient is False


class TestContentFilterError:
    def test_not_transient(self):
        e = ContentFilterError("blocked", provider="openai")
        assert e.is_transient is False


class TestProviderNotInstalledError:
    def test_message(self):
        e = ProviderNotInstalledError("OpenAI", "openai", "openai")
        assert "openai" in str(e)
        assert "pip install anchor[openai]" in str(e)

    def test_not_provider_error(self):
        """ProviderNotInstalledError is a setup error, not a runtime provider error."""
        e = ProviderNotInstalledError("X", "x", "x")
        assert not isinstance(e, ProviderError)
