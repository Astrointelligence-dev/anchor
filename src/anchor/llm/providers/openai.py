"""OpenAIProvider adapter for the multi-provider LLM layer.

Converts between Anchor's unified models and the OpenAI SDK.
Self-registers via register_provider() at module import time.

The `openai` SDK is imported lazily inside methods so this module
can be imported even when the SDK is not installed (import fails only
when you actually try to use the provider).

This class is designed to be subclassed for OpenAI-compatible APIs:
- GrokProvider (xAI)
- OpenRouterProvider
- OllamaProvider
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Iterator

from anchor.llm.base import BaseLLMProvider
from anchor.llm.errors import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    ServerError,
    TimeoutError,
)
from anchor.llm.models import (
    LLMResponse,
    Message,
    StreamChunk,
    ToolSchema,
)
from anchor.llm.providers._openai_compat import (
    build_call_kwargs,
    convert_messages,
    convert_tool,
    map_stop_reason,
    parse_response,
    parse_stream_chunks,
)
from anchor.llm.registry import register_provider

# Module-level reference — populated at import time if openai is installed.
# Allows error mapping in _map_error() without re-importing each time.
try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore[assignment]


def _ensure_sdk() -> Any:
    """Import and return the openai module, raising clearly if missing."""
    if openai is None:  # pragma: no cover
        from anchor.llm.errors import ProviderNotInstalledError
        raise ProviderNotInstalledError("openai", "openai", "openai")
    return openai


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class OpenAIProvider(BaseLLMProvider):
    """Adapter for the OpenAI Chat Completions API.

    Also serves as a base class for GrokProvider, OpenRouterProvider, and
    OllamaProvider — all OpenAI-compatible APIs — which override
    `provider_name` and `_resolve_api_key()` and pass a custom `base_url`.
    """

    provider_name = "openai"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client: Any = None
        self._async_client: Any = None

    # ------------------------------------------------------------------
    # Client caching
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return a cached sync OpenAI client, creating it on first use."""
        if self._client is None:
            sdk = _ensure_sdk()
            self._client = sdk.OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client

    def _get_async_client(self) -> Any:
        """Return a cached async OpenAI client, creating it on first use."""
        if self._async_client is None:
            sdk = _ensure_sdk()
            self._async_client = sdk.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._async_client

    # ------------------------------------------------------------------
    # BaseLLMProvider abstract method implementations
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str | None:
        return os.environ.get("OPENAI_API_KEY")

    def _do_invoke(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_client()
        call_kwargs = build_call_kwargs(
            self._model, convert_messages(messages), tools, **kwargs,
        )

        try:
            response = client.chat.completions.create(**call_kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc

        return parse_response(response, self.provider_name)

    def _do_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        call_kwargs = build_call_kwargs(
            self._model, convert_messages(messages), tools, stream=True, **kwargs,
        )

        try:
            stream = client.chat.completions.create(**call_kwargs)
            for raw_chunk in stream:
                for chunk in parse_stream_chunks(raw_chunk):
                    yield chunk
        except Exception as exc:
            raise self._map_error(exc) from exc

    async def _do_ainvoke(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_async_client()
        call_kwargs = build_call_kwargs(
            self._model, convert_messages(messages), tools, **kwargs,
        )

        try:
            response = await client.chat.completions.create(**call_kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc

        return parse_response(response, self.provider_name)

    async def _do_astream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_async_client()
        call_kwargs = build_call_kwargs(
            self._model, convert_messages(messages), tools, stream=True, **kwargs,
        )

        try:
            stream = await client.chat.completions.create(**call_kwargs)
            async for raw_chunk in stream:
                for chunk in parse_stream_chunks(raw_chunk):
                    yield chunk
        except Exception as exc:
            raise self._map_error(exc) from exc

    # ------------------------------------------------------------------
    # Delegating helpers (keep the instance-method interface for
    # subclasses and tests while reusing shared implementations)
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Anchor messages to OpenAI Chat Completions format."""
        return convert_messages(messages)

    def _convert_tool(self, tool: ToolSchema) -> dict[str, Any]:
        """Convert a ToolSchema to OpenAI tool definition format."""
        return convert_tool(tool)

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse an OpenAI SDK response into an LLMResponse."""
        return parse_response(response, self.provider_name)

    def _parse_stream_chunks(self, chunk: Any) -> list[StreamChunk]:
        """Parse a single OpenAI stream chunk into StreamChunks."""
        return parse_stream_chunks(chunk)

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    def _map_error(self, exc: Exception) -> ProviderError:
        """Map an OpenAI SDK exception to our error hierarchy.

        Uses class name matching so this works correctly even when the
        `openai` module reference is replaced by a mock in tests.
        """
        # Walk the full MRO and collect all class names — handles
        # both real SDK exceptions and dynamically-created test mocks.
        mro_names = {cls.__name__ for cls in type(exc).__mro__}

        if "AuthenticationError" in mro_names:
            return AuthenticationError(str(exc), provider=self.provider_name)

        if "RateLimitError" in mro_names:
            return RateLimitError(str(exc), provider=self.provider_name)

        if "NotFoundError" in mro_names:
            return ModelNotFoundError(str(exc), provider=self.provider_name)

        if "APIConnectionError" in mro_names or "APIConnectTimeoutError" in mro_names:
            return TimeoutError(str(exc), provider=self.provider_name)

        if "APITimeoutError" in mro_names:
            return TimeoutError(str(exc), provider=self.provider_name)

        if "APIStatusError" in mro_names:
            status_code = getattr(exc, "status_code", 0)
            if status_code >= 500:
                return ServerError(str(exc), provider=self.provider_name)
            # Other 4xx — non-transient ProviderError
            return ProviderError(str(exc), provider=self.provider_name, is_transient=False)

        # Fallback
        return ProviderError(str(exc), provider=self.provider_name)


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register_provider("openai", OpenAIProvider)
