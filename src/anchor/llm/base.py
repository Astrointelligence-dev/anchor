"""LLMProvider Protocol and BaseLLMProvider ABC.

The Protocol defines what all providers must satisfy (structural subtyping).
The ABC provides shared retry, timeout, and property logic.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator, Protocol, runtime_checkable

from anchor.llm.errors import ProviderError, RateLimitError
from anchor.llm.models import (
    LLMResponse,
    Message,
    StreamChunk,
    ToolSchema,
)


@runtime_checkable
class LLMProvider(Protocol):
    """Unified interface all LLM providers must satisfy."""

    @property
    def model_id(self) -> str: ...

    @property
    def provider_name(self) -> str: ...

    def invoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]: ...

    async def ainvoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    async def astream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]: ...


class BaseLLMProvider(ABC):
    """Abstract base class with shared retry, timeout, and property logic."""

    provider_name: str  # set by subclass as class attribute

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        self._model = model
        self._api_key = api_key or self._resolve_api_key()
        self._base_url = base_url
        self._max_retries = max_retries
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return f"{self.provider_name}/{self._model}"

    # --- Abstract methods for subclasses ---

    @abstractmethod
    def _resolve_api_key(self) -> str | None: ...

    @abstractmethod
    def _do_invoke(
        self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs: Any
    ) -> LLMResponse: ...

    @abstractmethod
    def _do_stream(
        self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs: Any
    ) -> Iterator[StreamChunk]: ...

    @abstractmethod
    async def _do_ainvoke(
        self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs: Any
    ) -> LLMResponse: ...

    @abstractmethod
    async def _do_astream(
        self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs: Any
    ) -> AsyncIterator[StreamChunk]: ...

    # --- Public methods with retry ---

    def invoke(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return self._with_retry(self._do_invoke, messages, tools, **kwargs)

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        last_error: ProviderError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                stream_iter = self._do_stream(messages, tools, **kwargs)
                first_chunk = next(stream_iter)
            except StopIteration:
                return
            except ProviderError as e:
                if not e.is_transient or attempt == self._max_retries:
                    raise
                last_error = e
                delay = min(2**attempt, 8)
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = e.retry_after
                time.sleep(delay)
                continue
            # Committed: chunks reached the consumer, so a mid-stream error
            # must propagate — a retry here would re-yield delivered text.
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
        return await self._with_async_retry(
            self._do_ainvoke, messages, tools, **kwargs
        )

    async def astream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        last_error: ProviderError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                stream_iter = self._do_astream(messages, tools, **kwargs)
                first_chunk = await stream_iter.__anext__()
            except StopAsyncIteration:
                return
            except ProviderError as e:
                if not e.is_transient or attempt == self._max_retries:
                    raise
                last_error = e
                delay = min(2**attempt, 8)
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = e.retry_after
                await asyncio.sleep(delay)
                continue
            # Committed: chunks reached the consumer, so a mid-stream error
            # must propagate — a retry here would re-yield delivered text.
            yield first_chunk
            async for chunk in stream_iter:
                yield chunk
            return
        if last_error:
            raise last_error

    # --- Retry logic ---

    def _with_retry(self, fn, *args, **kwargs):
        last_error: ProviderError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except ProviderError as e:
                if not e.is_transient or attempt == self._max_retries:
                    raise
                last_error = e
                delay = min(2**attempt, 8)
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = e.retry_after
                time.sleep(delay)
        raise last_error  # type: ignore[misc]

    async def _with_async_retry(self, fn, *args, **kwargs):
        """Retry for async coroutines (ainvoke). NOT for async generators (astream)."""
        last_error: ProviderError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await fn(*args, **kwargs)
            except ProviderError as e:
                if not e.is_transient or attempt == self._max_retries:
                    raise
                last_error = e
                delay = min(2**attempt, 8)
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = e.retry_after
                await asyncio.sleep(delay)
        raise last_error  # type: ignore[misc]
