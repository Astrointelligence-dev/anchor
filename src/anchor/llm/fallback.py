"""FallbackProvider — wraps multiple providers with fallback-on-failure.

Fallback rules:
- invoke/ainvoke: On transient ProviderError, tries next provider.
- stream/astream: Fallback ONLY before first chunk. Once streaming has
  started yielding, failure raises immediately (no silent mid-stream switch).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from anchor.llm.base import LLMProvider
from anchor.llm.errors import ProviderError
from anchor.llm.models import LLMResponse, Message, StreamChunk, ToolSchema


class FallbackProvider:
    """Wraps multiple providers with fallback-on-failure logic."""

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: list[LLMProvider],
    ):
        self._providers = [primary, *fallbacks]
        self._primary = primary

    @property
    def model_id(self) -> str:
        return self._primary.model_id

    @property
    def provider_name(self) -> str:
        return self._primary.provider_name

    def invoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        last_error: ProviderError | None = None
        for provider in self._providers:
            try:
                return provider.invoke(messages, tools=tools, **kwargs)
            except ProviderError as e:
                if not e.is_transient:
                    raise
                last_error = e
        raise last_error  # type: ignore[misc]

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        last_error: ProviderError | None = None
        for provider in self._providers:
            try:
                stream_iter = provider.stream(messages, tools=tools, **kwargs)
                first_chunk = next(stream_iter)
            except StopIteration:
                return
            except ProviderError as e:
                if not e.is_transient:
                    raise
                last_error = e
                continue
            # Committed to this provider — errors from here propagate directly
            yield first_chunk
            yield from stream_iter
            return
        if last_error:
            raise last_error

    async def ainvoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        last_error: ProviderError | None = None
        for provider in self._providers:
            try:
                return await provider.ainvoke(messages, tools=tools, **kwargs)
            except ProviderError as e:
                if not e.is_transient:
                    raise
                last_error = e
        raise last_error  # type: ignore[misc]

    async def astream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        last_error: ProviderError | None = None
        for provider in self._providers:
            try:
                stream_iter = provider.astream(messages, tools=tools, **kwargs)
                first_chunk = await stream_iter.__anext__()
            except StopAsyncIteration:
                return
            except ProviderError as e:
                if not e.is_transient:
                    raise
                last_error = e
                continue
            # Committed to this provider — errors from here propagate directly
            yield first_chunk
            async for chunk in stream_iter:
                yield chunk
            return
        if last_error:
            raise last_error
