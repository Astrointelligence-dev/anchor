"""AnthropicProvider adapter for the multi-provider LLM layer.

Converts between Anchor's unified models and the Anthropic SDK.
Self-registers via register_provider() at module import time.

The `anthropic` SDK is imported lazily inside methods so this module
can be imported even when the SDK is not installed (import fails only
when you actually try to use the provider).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

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
    Role,
    StopReason,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    ToolSchema,
    Usage,
)
from anchor.llm.registry import register_provider

# Module-level reference — populated by _ensure_sdk() to allow error mapping
# outside of individual method calls (e.g. in _map_error).
try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]


def _ensure_sdk() -> Any:
    """Import and return the anthropic module, raising clearly if missing."""
    if anthropic is None:  # pragma: no cover
        from anchor.llm.errors import ProviderNotInstalledError
        raise ProviderNotInstalledError("anthropic", "anthropic", "anthropic")
    return anthropic


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------

_STOP_REASON_MAP: dict[str, StopReason] = {
    "end_turn": StopReason.STOP,
    "max_tokens": StopReason.MAX_TOKENS,
    "tool_use": StopReason.TOOL_USE,
}


def _map_stop_reason(stop_reason: str | None) -> StopReason:
    if stop_reason is None:
        return StopReason.STOP
    return _STOP_REASON_MAP.get(stop_reason, StopReason.STOP)


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


class AnthropicProvider(BaseLLMProvider):
    """Adapter for the Anthropic Messages API."""

    provider_name = "anthropic"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client: Any = None
        self._async_client: Any = None

    # ------------------------------------------------------------------
    # Client caching
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return a cached sync Anthropic client, creating it on first use."""
        if self._client is None:
            sdk = _ensure_sdk()
            self._client = sdk.Anthropic(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _get_async_client(self) -> Any:
        """Return a cached async Anthropic client, creating it on first use."""
        if self._async_client is None:
            sdk = _ensure_sdk()
            self._async_client = sdk.AsyncAnthropic(api_key=self._api_key, base_url=self._base_url)
        return self._async_client

    # ------------------------------------------------------------------
    # BaseLLMProvider abstract method implementations
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY")

    def _do_invoke(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_client()
        system, converted = self._extract_system_and_convert(messages)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system is not None:
            call_kwargs["system"] = system
        if tools:
            call_kwargs["tools"] = [self._convert_tool(t) for t in tools]
        if kwargs.get("temperature") is not None:
            call_kwargs["temperature"] = kwargs["temperature"]
        if kwargs.get("stop"):
            call_kwargs["stop_sequences"] = kwargs["stop"]

        try:
            response = client.messages.create(**call_kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc

        return self._parse_response(response)

    def _do_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        system, converted = self._extract_system_and_convert(messages)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system is not None:
            call_kwargs["system"] = system
        if tools:
            call_kwargs["tools"] = [self._convert_tool(t) for t in tools]
        if kwargs.get("temperature") is not None:
            call_kwargs["temperature"] = kwargs["temperature"]
        if kwargs.get("stop"):
            call_kwargs["stop_sequences"] = kwargs["stop"]

        try:
            input_tokens: int | None = None
            with client.messages.stream(**call_kwargs) as stream:
                for event in stream:
                    # Track input_tokens from message_start
                    if event.type == "message_start":
                        msg_usage = getattr(event.message, "usage", None)
                        if msg_usage is not None:
                            input_tokens = getattr(msg_usage, "input_tokens", None)
                    chunk = self._parse_stream_event(event, input_tokens=input_tokens)
                    if chunk is not None:
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
        system, converted = self._extract_system_and_convert(messages)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system is not None:
            call_kwargs["system"] = system
        if tools:
            call_kwargs["tools"] = [self._convert_tool(t) for t in tools]
        if kwargs.get("temperature") is not None:
            call_kwargs["temperature"] = kwargs["temperature"]
        if kwargs.get("stop"):
            call_kwargs["stop_sequences"] = kwargs["stop"]

        try:
            response = await client.messages.create(**call_kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc

        return self._parse_response(response)

    async def _do_astream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_async_client()
        system, converted = self._extract_system_and_convert(messages)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system is not None:
            call_kwargs["system"] = system
        if tools:
            call_kwargs["tools"] = [self._convert_tool(t) for t in tools]
        if kwargs.get("temperature") is not None:
            call_kwargs["temperature"] = kwargs["temperature"]
        if kwargs.get("stop"):
            call_kwargs["stop_sequences"] = kwargs["stop"]

        try:
            input_tokens: int | None = None
            async with client.messages.stream(**call_kwargs) as stream:
                async for event in stream:
                    # Track input_tokens from message_start
                    if event.type == "message_start":
                        msg_usage = getattr(event.message, "usage", None)
                        if msg_usage is not None:
                            input_tokens = getattr(msg_usage, "input_tokens", None)
                    chunk = self._parse_stream_event(event, input_tokens=input_tokens)
                    if chunk is not None:
                        yield chunk
        except Exception as exc:
            raise self._map_error(exc) from exc

    # ------------------------------------------------------------------
    # Message conversion helpers
    # ------------------------------------------------------------------

    def _extract_system_and_convert(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Split system message out and convert remaining to Anthropic format."""
        system: str | None = None
        converted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Take last system message if multiple (edge case)
                if isinstance(msg.content, str):
                    system = msg.content
                continue

            if msg.role == Role.TOOL:
                # Tool result → user message with tool_result content block
                if msg.tool_result is not None:
                    block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_result.tool_call_id,
                        "content": msg.tool_result.content,
                    }
                    if msg.tool_result.is_error:
                        block["is_error"] = True
                    converted.append({"role": "user", "content": [block]})
                continue

            if msg.role == Role.ASSISTANT and msg.tool_calls:
                # Assistant message with tool calls → content blocks
                blocks: list[dict[str, Any]] = []
                if msg.content:
                    if isinstance(msg.content, str):
                        blocks.append({"type": "text", "text": msg.content})
                    else:
                        for block in msg.content:
                            if block.type == "text" and block.text is not None:
                                blocks.append({"type": "text", "text": block.text})
                for tc in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                converted.append({"role": "assistant", "content": blocks})
                continue

            # Regular user / assistant messages
            role_str = "user" if msg.role == Role.USER else "assistant"
            if isinstance(msg.content, str):
                converted.append({"role": role_str, "content": msg.content})
            elif isinstance(msg.content, list):
                blocks = []
                for block in msg.content:
                    if block.type == "text" and block.text is not None:
                        blocks.append({"type": "text", "text": block.text})
                    elif block.type == "image_url" and block.image_url is not None:
                        blocks.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": block.image_url},
                            }
                        )
                    elif block.type == "image_base64" and block.image_base64 is not None:
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": block.media_type or "image/png",
                                    "data": block.image_base64,
                                },
                            }
                        )
                converted.append({"role": role_str, "content": blocks})
            else:
                # None content — empty message, skip
                pass

        return system, converted

    # ------------------------------------------------------------------
    # Tool schema conversion
    # ------------------------------------------------------------------

    def _convert_tool(self, tool: ToolSchema) -> dict[str, Any]:
        """Convert a ToolSchema to Anthropic tool definition format."""
        converted: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        if tool.input_examples:
            converted["input_examples"] = list(tool.input_examples)
        return converted

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse an Anthropic SDK response into an LLMResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        content = "".join(text_parts) if text_parts else None
        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=response.model,
            provider=self.provider_name,
            stop_reason=_map_stop_reason(response.stop_reason),
        )

    # ------------------------------------------------------------------
    # Stream event parsing
    # ------------------------------------------------------------------

    def _parse_stream_event(
        self, event: Any, *, input_tokens: int | None = None
    ) -> StreamChunk | None:
        """Parse a single Anthropic stream event into a StreamChunk, or None."""
        event_type = event.type

        if event_type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                return StreamChunk(content=delta.text)
            if delta.type == "input_json_delta":
                return StreamChunk(
                    tool_call_delta=ToolCallDelta(
                        index=event.index,
                        arguments_fragment=delta.partial_json,
                    )
                )

        if event_type == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                return StreamChunk(
                    tool_call_delta=ToolCallDelta(
                        index=event.index,
                        id=block.id,
                        name=block.name,
                    )
                )
            # text content_block_start carries no useful data
            return None

        if event_type == "message_delta":
            stop_reason = _map_stop_reason(event.delta.stop_reason)
            # Extract output_tokens from the message_delta usage
            usage: Usage | None = None
            event_usage = getattr(event, "usage", None)
            if event_usage is not None:
                output_tokens = getattr(event_usage, "output_tokens", 0)
                prompt = input_tokens or 0
                usage = Usage(
                    prompt_tokens=prompt,
                    completion_tokens=output_tokens,
                    total_tokens=prompt + output_tokens,
                )
            return StreamChunk(stop_reason=stop_reason, usage=usage)

        return None

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    def _map_error(self, exc: Exception) -> ProviderError:
        """Map an Anthropic SDK exception to our error hierarchy.

        Uses class name matching so this works correctly even when the
        `anthropic` module reference is replaced by a mock in tests.
        """
        # Walk the full MRO and collect all class names — this handles
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

register_provider("anthropic", AnthropicProvider)
