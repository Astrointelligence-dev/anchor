"""Tests for anchor.llm.base — Protocol and BaseLLMProvider."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from unittest.mock import patch

import pytest

from anchor.llm.base import BaseLLMProvider
from anchor.llm.errors import AuthenticationError, RateLimitError, ServerError
from anchor.llm.models import (
    LLMResponse,
    Message,
    Role,
    StopReason,
    StreamChunk,
    Usage,
)


def _make_response(**kwargs) -> LLMResponse:
    defaults = {
        "content": "hello",
        "usage": Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        "model": "test-model",
        "provider": "test",
        "stop_reason": StopReason.STOP,
    }
    defaults.update(kwargs)
    return LLMResponse(**defaults)


class ConcreteProvider(BaseLLMProvider):
    """Minimal concrete implementation for testing."""

    provider_name = "test"

    def _resolve_api_key(self) -> str | None:
        return "test-key"

    def _do_invoke(self, messages, tools, **kwargs) -> LLMResponse:
        return _make_response()

    def _do_stream(self, messages, tools, **kwargs) -> Iterator[StreamChunk]:
        yield StreamChunk(content="hello")
        yield StreamChunk(stop_reason=StopReason.STOP)

    async def _do_ainvoke(self, messages, tools, **kwargs) -> LLMResponse:
        return _make_response()

    async def _do_astream(self, messages, tools, **kwargs) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(content="hello")
        yield StreamChunk(stop_reason=StopReason.STOP)


class TestBaseLLMProviderProperties:
    def test_model_id(self):
        p = ConcreteProvider(model="my-model")
        assert p.model_id == "test/my-model"

    def test_provider_name(self):
        p = ConcreteProvider(model="my-model")
        assert p.provider_name == "test"


class TestBaseLLMProviderInvoke:
    def test_invoke_success(self):
        p = ConcreteProvider(model="m")
        msgs = [Message(role=Role.USER, content="hi")]
        result = p.invoke(msgs)
        assert result.content == "hello"

    def test_stream_success(self):
        p = ConcreteProvider(model="m")
        msgs = [Message(role=Role.USER, content="hi")]
        chunks = list(p.stream(msgs))
        assert len(chunks) == 2
        assert chunks[0].content == "hello"


class TestRetryLogic:
    @patch("time.sleep")
    def test_retries_on_transient_error(self, mock_sleep):
        call_count = 0

        class RetryProvider(ConcreteProvider):
            def _do_invoke(self, messages, tools, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ServerError("500", provider="test")
                return _make_response()

        p = RetryProvider(model="m", max_retries=2)
        result = p.invoke([Message(role=Role.USER, content="hi")])
        assert result.content == "hello"
        assert call_count == 3
        assert mock_sleep.call_count == 2

    def test_no_retry_on_non_transient_error(self):
        class AuthFailProvider(ConcreteProvider):
            def _do_invoke(self, messages, tools, **kwargs):
                raise AuthenticationError("bad key", provider="test")

        p = AuthFailProvider(model="m", max_retries=2)
        with pytest.raises(AuthenticationError):
            p.invoke([Message(role=Role.USER, content="hi")])

    @patch("time.sleep")
    def test_raises_after_max_retries(self, mock_sleep):
        class AlwaysFailProvider(ConcreteProvider):
            def _do_invoke(self, messages, tools, **kwargs):
                raise ServerError("500", provider="test")

        p = AlwaysFailProvider(model="m", max_retries=1)
        with pytest.raises(ServerError):
            p.invoke([Message(role=Role.USER, content="hi")])

    @patch("time.sleep")
    def test_respects_rate_limit_retry_after(self, mock_sleep):
        call_count = 0

        class RateLimitProvider(ConcreteProvider):
            def _do_invoke(self, messages, tools, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RateLimitError("429", provider="test", retry_after=0.01)
                return _make_response()

        p = RateLimitProvider(model="m", max_retries=1)
        result = p.invoke([Message(role=Role.USER, content="hi")])
        assert result.content == "hello"
        assert call_count == 2


class TestAsyncRetryLogic:
    async def test_ainvoke_success(self):
        p = ConcreteProvider(model="m")
        result = await p.ainvoke([Message(role=Role.USER, content="hi")])
        assert result.content == "hello"

    async def test_astream_success(self):
        p = ConcreteProvider(model="m")
        chunks = []
        async for chunk in p.astream([Message(role=Role.USER, content="hi")]):
            chunks.append(chunk)
        assert len(chunks) == 2

    async def test_async_retries_on_transient(self):
        call_count = 0

        class AsyncRetryProvider(ConcreteProvider):
            async def _do_ainvoke(self, messages, tools, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ServerError("500", provider="test")
                return _make_response()

        with patch("anchor.llm.base.asyncio.sleep", return_value=None):
            p = AsyncRetryProvider(model="m", max_retries=1)
            result = await p.ainvoke([Message(role=Role.USER, content="hi")])
        assert result.content == "hello"


class TestStreamRetryLogic:
    @patch("time.sleep")
    def test_stream_succeeds(self, mock_sleep):
        p = ConcreteProvider(model="m")
        msgs = [Message(role=Role.USER, content="hi")]
        chunks = list(p.stream(msgs))
        assert len(chunks) == 2
        assert chunks[0].content == "hello"
        assert mock_sleep.call_count == 0

    @patch("time.sleep")
    def test_stream_retries_on_transient_error(self, mock_sleep):
        call_count = 0

        class StreamRetryProvider(ConcreteProvider):
            def _do_stream(self, messages, tools, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ServerError("500", provider="test")
                yield StreamChunk(content="recovered")
                yield StreamChunk(stop_reason=StopReason.STOP)

        p = StreamRetryProvider(model="m", max_retries=1)
        chunks = list(p.stream([Message(role=Role.USER, content="hi")]))
        assert call_count == 2
        assert chunks[0].content == "recovered"
        assert mock_sleep.call_count == 1

    async def test_astream_succeeds(self):
        p = ConcreteProvider(model="m")
        chunks = []
        async for chunk in p.astream([Message(role=Role.USER, content="hi")]):
            chunks.append(chunk)
        assert len(chunks) == 2
        assert chunks[0].content == "hello"

    async def test_astream_retries_on_transient_error(self):
        call_count = 0

        class AsyncStreamRetryProvider(ConcreteProvider):
            async def _do_astream(self, messages, tools, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ServerError("500", provider="test")
                yield StreamChunk(content="recovered")
                yield StreamChunk(stop_reason=StopReason.STOP)

        with patch("anchor.llm.base.asyncio.sleep", return_value=None):
            p = AsyncStreamRetryProvider(model="m", max_retries=1)
            chunks = []
            async for chunk in p.astream([Message(role=Role.USER, content="hi")]):
                chunks.append(chunk)
        assert call_count == 2
        assert chunks[0].content == "recovered"
