"""LiteLLMProvider adapter for the multi-provider LLM layer.

LiteLLM is an optional catch-all that routes to 100+ LLM providers
using a unified OpenAI-compatible interface. It handles API keys
internally via environment variables, so `_resolve_api_key()` returns None.

The `litellm` SDK is imported lazily inside methods so this module
can be imported even when the SDK is not installed (import fails only
when you actually try to use the provider).

Self-registers via register_provider() at module import time.
"""

from __future__ import annotations

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
    parse_response,
    parse_stream_chunks,
)
from anchor.llm.registry import register_provider


# ---------------------------------------------------------------------------
# LiteLLMProvider
# ---------------------------------------------------------------------------


class LiteLLMProvider(BaseLLMProvider):
    """Adapter for the LiteLLM library.

    LiteLLM acts as a catch-all, routing to 100+ providers using an
    OpenAI-compatible response format. It manages API key resolution
    internally through environment variables (e.g., OPENAI_API_KEY,
    ANTHROPIC_API_KEY, etc.), so _resolve_api_key() returns None.

    Usage:
        provider = LiteLLMProvider(model="gpt-4o")
        # or for other providers:
        provider = LiteLLMProvider(model="anthropic/claude-3-5-sonnet-20241022")
        provider = LiteLLMProvider(model="gemini/gemini-2.0-flash")
    """

    provider_name = "litellm"

    # ------------------------------------------------------------------
    # BaseLLMProvider abstract method implementations
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str | None:
        """LiteLLM manages API keys via environment variables internally."""
        return None

    def _do_invoke(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> LLMResponse:
        import litellm  # noqa: PLC0415 — lazy import

        call_kwargs = build_call_kwargs(
            self._model, convert_messages(messages), tools, **kwargs,
        )
        if self._timeout is not None:
            call_kwargs.setdefault("timeout", self._timeout)

        try:
            response = litellm.completion(**call_kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc

        return parse_response(response, self.provider_name)

    def _do_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        import litellm  # noqa: PLC0415 — lazy import

        call_kwargs = build_call_kwargs(
            self._model, convert_messages(messages), tools, stream=True, **kwargs,
        )
        if self._timeout is not None:
            call_kwargs.setdefault("timeout", self._timeout)

        try:
            stream = litellm.completion(**call_kwargs)
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
        import litellm  # noqa: PLC0415 — lazy import

        call_kwargs = build_call_kwargs(
            self._model, convert_messages(messages), tools, **kwargs,
        )
        if self._timeout is not None:
            call_kwargs.setdefault("timeout", self._timeout)

        try:
            response = await litellm.acompletion(**call_kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc

        return parse_response(response, self.provider_name)

    async def _do_astream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        import litellm  # noqa: PLC0415 — lazy import

        call_kwargs = build_call_kwargs(
            self._model, convert_messages(messages), tools, stream=True, **kwargs,
        )
        if self._timeout is not None:
            call_kwargs.setdefault("timeout", self._timeout)

        try:
            stream = await litellm.acompletion(**call_kwargs)
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
        """Convert Anchor messages to OpenAI-compatible format used by LiteLLM."""
        return convert_messages(messages)

    def _convert_tool(self, tool: ToolSchema) -> dict[str, Any]:
        """Convert a ToolSchema to OpenAI-compatible tool definition format."""
        return convert_tool(tool)

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse a LiteLLM response (OpenAI-compatible) into an LLMResponse."""
        return parse_response(response, self.provider_name)

    def _parse_stream_chunks(self, chunk: Any) -> list[StreamChunk]:
        """Parse a single LiteLLM stream chunk into StreamChunks."""
        return parse_stream_chunks(chunk)

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    def _map_error(self, exc: Exception) -> ProviderError:
        """Map a LiteLLM exception to our error hierarchy.

        LiteLLM raises its own exception types that mirror the OpenAI names.
        Uses class name matching so this works with both real SDK exceptions
        and dynamically-created test mocks.
        """
        # Walk the full MRO and collect all class names — handles
        # both real SDK exceptions and dynamically-created test mocks.
        mro_names = {cls.__name__ for cls in type(exc).__mro__}

        if "AuthenticationError" in mro_names:
            return AuthenticationError(str(exc), provider=self.provider_name)

        if "RateLimitError" in mro_names:
            return RateLimitError(str(exc), provider=self.provider_name)

        if "NotFoundError" in mro_names or "ModelNotFoundError" in mro_names:
            return ModelNotFoundError(str(exc), provider=self.provider_name)

        if "APIConnectionError" in mro_names or "APIConnectTimeoutError" in mro_names:
            return TimeoutError(str(exc), provider=self.provider_name)

        if "APITimeoutError" in mro_names or "Timeout" in mro_names:
            return TimeoutError(str(exc), provider=self.provider_name)

        if "APIStatusError" in mro_names:
            status_code = getattr(exc, "status_code", 0)
            if status_code >= 500:
                return ServerError(str(exc), provider=self.provider_name)
            return ProviderError(str(exc), provider=self.provider_name, is_transient=False)

        if "ServiceUnavailableError" in mro_names:
            return ServerError(str(exc), provider=self.provider_name)

        # Fallback
        return ProviderError(str(exc), provider=self.provider_name)


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register_provider("litellm", LiteLLMProvider)
