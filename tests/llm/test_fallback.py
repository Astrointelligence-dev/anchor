"""Tests for anchor.llm.fallback."""

from __future__ import annotations

import pytest

from anchor.llm.base import BaseLLMProvider
from anchor.llm.errors import AuthenticationError, ServerError
from anchor.llm.fallback import FallbackProvider
from anchor.llm.models import (
    LLMResponse,
    Message,
    Role,
    StopReason,
    StreamChunk,
    Usage,
)


def _resp(provider: str = "p", content: str = "ok") -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="m",
        provider=provider,
        stop_reason=StopReason.STOP,
    )


class StubProvider(BaseLLMProvider):
    provider_name = "stub"

    def __init__(self, model="m", invoke_fn=None, stream_fn=None, **kwargs):
        super().__init__(model=model, max_retries=0, **kwargs)
        self._invoke_fn = invoke_fn
        self._stream_fn = stream_fn

    def _resolve_api_key(self):
        return "k"

    def _do_invoke(self, messages, tools, **kwargs):
        if self._invoke_fn:
            return self._invoke_fn()
        return _resp(self.provider_name)

    def _do_stream(self, messages, tools, **kwargs):
        if self._stream_fn:
            yield from self._stream_fn()
        else:
            yield StreamChunk(content="ok")

    async def _do_ainvoke(self, messages, tools, **kwargs):
        return self._do_invoke(messages, tools, **kwargs)

    async def _do_astream(self, messages, tools, **kwargs):
        for chunk in self._do_stream(messages, tools, **kwargs):
            yield chunk


class TestFallbackProviderProperties:
    def test_model_id_from_primary(self):
        primary = StubProvider(model="m1")
        fb = FallbackProvider(primary=primary, fallbacks=[StubProvider(model="m2")])
        assert fb.model_id == "stub/m1"

    def test_provider_name_from_primary(self):
        primary = StubProvider(model="m1")
        fb = FallbackProvider(primary=primary, fallbacks=[StubProvider(model="m2")])
        assert fb.provider_name == "stub"


class TestFallbackInvoke:
    def test_primary_succeeds(self):
        fb = FallbackProvider(
            primary=StubProvider(invoke_fn=lambda: _resp("primary")),
            fallbacks=[StubProvider(invoke_fn=lambda: _resp("fallback"))],
        )
        msgs = [Message(role=Role.USER, content="hi")]
        result = fb.invoke(msgs)
        assert result.provider == "primary"

    def test_falls_back_on_transient_error(self):
        def fail():
            raise ServerError("500", provider="primary")

        fb = FallbackProvider(
            primary=StubProvider(invoke_fn=fail),
            fallbacks=[StubProvider(invoke_fn=lambda: _resp("fallback"))],
        )
        result = fb.invoke([Message(role=Role.USER, content="hi")])
        assert result.provider == "fallback"

    def test_non_transient_error_not_caught(self):
        def fail():
            raise AuthenticationError("bad key", provider="primary")

        fb = FallbackProvider(
            primary=StubProvider(invoke_fn=fail),
            fallbacks=[StubProvider(invoke_fn=lambda: _resp("fallback"))],
        )
        with pytest.raises(AuthenticationError):
            fb.invoke([Message(role=Role.USER, content="hi")])

    def test_all_fail_raises_last_error(self):
        def fail():
            raise ServerError("500", provider="all")

        fb = FallbackProvider(
            primary=StubProvider(invoke_fn=fail),
            fallbacks=[StubProvider(invoke_fn=fail)],
        )
        with pytest.raises(ServerError):
            fb.invoke([Message(role=Role.USER, content="hi")])


class TestFallbackStream:
    def test_primary_stream_succeeds(self):
        fb = FallbackProvider(
            primary=StubProvider(),
            fallbacks=[StubProvider()],
        )
        chunks = list(fb.stream([Message(role=Role.USER, content="hi")]))
        assert len(chunks) >= 1

    def test_falls_back_before_first_chunk(self):
        """If primary fails before yielding, fallback kicks in."""
        def fail_stream():
            raise ServerError("500", provider="primary")
            yield  # make it a generator

        fb = FallbackProvider(
            primary=StubProvider(stream_fn=fail_stream),
            fallbacks=[StubProvider()],
        )
        chunks = list(fb.stream([Message(role=Role.USER, content="hi")]))
        assert any(c.content == "ok" for c in chunks)

    def test_mid_stream_failure_propagates(self):
        """After first chunk yielded, errors propagate — no silent switch."""
        def fail_mid_stream():
            yield StreamChunk(content="partial")
            raise ServerError("500", provider="primary")

        fb = FallbackProvider(
            primary=StubProvider(stream_fn=fail_mid_stream),
            fallbacks=[StubProvider()],
        )
        stream = fb.stream([Message(role=Role.USER, content="hi")])
        first = next(stream)
        assert first.content == "partial"
        with pytest.raises(ServerError):
            next(stream)


class TestFallbackAsyncInvoke:
    async def test_ainvoke_uses_primary_provider(self):
        fb = FallbackProvider(
            primary=StubProvider(invoke_fn=lambda: _resp("primary")),
            fallbacks=[StubProvider(invoke_fn=lambda: _resp("fallback"))],
        )
        result = await fb.ainvoke([Message(role=Role.USER, content="hi")])
        assert result.provider == "primary"

    async def test_ainvoke_falls_back_on_error(self):
        def fail():
            raise ServerError("500", provider="primary")

        fb = FallbackProvider(
            primary=StubProvider(invoke_fn=fail),
            fallbacks=[StubProvider(invoke_fn=lambda: _resp("fallback"))],
        )
        result = await fb.ainvoke([Message(role=Role.USER, content="hi")])
        assert result.provider == "fallback"


class TestFallbackAsyncStream:
    async def test_astream_uses_primary_provider(self):
        fb = FallbackProvider(
            primary=StubProvider(),
            fallbacks=[StubProvider()],
        )
        chunks = []
        async for chunk in fb.astream([Message(role=Role.USER, content="hi")]):
            chunks.append(chunk)
        assert len(chunks) >= 1

    async def test_astream_falls_back_on_error(self):
        """If primary fails before yielding, fallback kicks in."""
        def fail_stream():
            raise ServerError("500", provider="primary")
            yield  # make it a generator

        fb = FallbackProvider(
            primary=StubProvider(stream_fn=fail_stream),
            fallbacks=[StubProvider()],
        )
        chunks = []
        async for chunk in fb.astream([Message(role=Role.USER, content="hi")]):
            chunks.append(chunk)
        assert any(c.content == "ok" for c in chunks)
