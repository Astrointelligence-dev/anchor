"""GeminiProvider adapter for the multi-provider LLM layer.

Converts between Anchor's unified models and the Google GenAI SDK.
Self-registers via register_provider() at module import time.

The `google.genai` SDK is imported lazily inside methods so this module
can be imported even when the SDK is not installed (import fails only
when you actually try to use the provider).

Key Gemini differences vs OpenAI/Anthropic:
- System messages passed via system_instruction in GenerateContentConfig
- Assistant role is "model", not "assistant"
- Tool calls use function_call parts, tool results use function_response parts
- No stable tool call ID — we auto-generate one using the function name
- Finish reason strings are uppercase: "STOP", "MAX_TOKENS"
- Async uses client.aio.models.generate_content()
"""

from __future__ import annotations

import os
import uuid
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
    Role,
    StopReason,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    ToolSchema,
    Usage,
)
from anchor.llm.registry import register_provider

# Module-level reference — populated at import time if google.genai is installed.
# Used by _map_error() to inspect exception class names without re-importing.
try:
    import google.genai as genai
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]


def _ensure_sdk() -> Any:
    """Import and return the google.genai module, raising clearly if missing."""
    if genai is None:  # pragma: no cover
        from anchor.llm.errors import ProviderNotInstalledError
        raise ProviderNotInstalledError("gemini", "google-genai", "gemini")
    return genai


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------

_STOP_REASON_MAP: dict[str, StopReason] = {
    "STOP": StopReason.STOP,
    "MAX_TOKENS": StopReason.MAX_TOKENS,
    "FINISH_REASON_STOP": StopReason.STOP,
    "FINISH_REASON_MAX_TOKENS": StopReason.MAX_TOKENS,
}


def _map_stop_reason(finish_reason: str | None) -> StopReason:
    if finish_reason is None:
        return StopReason.STOP
    return _STOP_REASON_MAP.get(str(finish_reason), StopReason.STOP)


_TOOL_CHOICE_MODES = {"auto": "AUTO", "any": "ANY", "none": "NONE"}


def _convert_tool_choice(tool_choice: str | dict[str, Any]) -> dict[str, Any]:
    """Map the generic tool_choice (Anthropic shape) to Gemini's tool_config."""
    if isinstance(tool_choice, str):
        mode = _TOOL_CHOICE_MODES.get(tool_choice)
        if mode is None:
            msg = (
                f"Invalid tool_choice '{tool_choice}': expected one of "
                f"{sorted(_TOOL_CHOICE_MODES)} or {{'type': 'tool', 'name': ...}}"
            )
            raise ValueError(msg)
        return {"function_calling_config": {"mode": mode}}
    return {
        "function_calling_config": {
            "mode": "ANY",
            "allowed_function_names": [tool_choice.get("name", "")],
        },
    }


# ---------------------------------------------------------------------------
# GeminiProvider
# ---------------------------------------------------------------------------


class GeminiProvider(BaseLLMProvider):
    """Adapter for the Google Gemini API via the google-genai SDK."""

    provider_name = "gemini"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client: Any = None

    # ------------------------------------------------------------------
    # Client caching
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return a cached Gemini client, creating it on first use.

        Gemini uses a single Client for both sync and async (via client.aio).
        """
        if self._client is None:
            sdk = _ensure_sdk()
            self._client = sdk.Client(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------
    # BaseLLMProvider abstract method implementations
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str | None:
        return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    def _do_invoke(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_client()
        system, converted = self._extract_system_and_convert(messages)

        config = self._build_config(_ensure_sdk(), system, tools, **kwargs)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "contents": converted,
            "config": config,
        }

        try:
            response = client.models.generate_content(**call_kwargs)
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

        config = self._build_config(_ensure_sdk(), system, tools, **kwargs)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "contents": converted,
            "config": config,
        }

        try:
            stream = client.models.generate_content_stream(**call_kwargs)
            for raw_chunk in stream:
                chunk = self._parse_stream_chunk(raw_chunk)
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
        client = self._get_client()
        system, converted = self._extract_system_and_convert(messages)

        config = self._build_config(_ensure_sdk(), system, tools, **kwargs)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "contents": converted,
            "config": config,
        }

        try:
            response = await client.aio.models.generate_content(**call_kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc

        return self._parse_response(response)

    async def _do_astream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_client()
        system, converted = self._extract_system_and_convert(messages)

        config = self._build_config(_ensure_sdk(), system, tools, **kwargs)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "contents": converted,
            "config": config,
        }

        try:
            async for raw_chunk in await client.aio.models.generate_content_stream(
                **call_kwargs
            ):
                chunk = self._parse_stream_chunk(raw_chunk)
                if chunk is not None:
                    yield chunk
        except Exception as exc:
            raise self._map_error(exc) from exc

    # ------------------------------------------------------------------
    # Config builder
    # ------------------------------------------------------------------

    def _build_config(
        self,
        sdk: Any,
        system: str | None,
        tools: list[ToolSchema] | None,
        **kwargs: Any,
    ) -> Any:
        """Build a GenerateContentConfig (or dict fallback) for the SDK call."""
        config_kwargs: dict[str, Any] = {}

        if system is not None:
            config_kwargs["system_instruction"] = system

        if tools:
            config_kwargs["tools"] = [self._convert_tool(t) for t in tools]

        if kwargs.get("max_tokens") is not None:
            config_kwargs["max_output_tokens"] = kwargs["max_tokens"]

        if kwargs.get("temperature") is not None:
            config_kwargs["temperature"] = kwargs["temperature"]

        if kwargs.get("stop"):
            config_kwargs["stop_sequences"] = kwargs["stop"]

        if kwargs.get("tool_choice") is not None and tools:
            config_kwargs["tool_config"] = _convert_tool_choice(kwargs["tool_choice"])

        # Try to use the SDK's GenerateContentConfig if available; fall back to
        # a simple namespace object so tests can inspect attributes.
        try:
            return sdk.types.GenerateContentConfig(**config_kwargs)
        except Exception:
            # In tests or when type class is unavailable, use a simple object.
            cfg = type("GenerateContentConfig", (), {})()
            for key, value in config_kwargs.items():
                setattr(cfg, key, value)
            return cfg

    # ------------------------------------------------------------------
    # Message conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_tool_name(messages: list[Message], tool_call_id: str) -> str:
        """Find the function name for a tool_call_id by scanning message history."""
        for msg in messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.id == tool_call_id:
                        return tc.name
        return tool_call_id  # fallback to ID if not found

    def _extract_system_and_convert(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Split system message out and convert remaining to Gemini Content format.

        Gemini uses:
        - role "user" for user messages
        - role "model" for assistant messages (NOT "assistant")
        - Parts are dicts: {"text": "..."} or {"function_call": {...}} etc.
        - Tool results are user-role messages with function_response parts.
        """
        system: str | None = None
        converted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Take last system message (edge case: multiple system messages)
                if isinstance(msg.content, str):
                    system = msg.content
                continue

            if msg.role == Role.TOOL:
                # Tool result → user message with function_response part
                if msg.tool_result is not None:
                    converted.append(
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "function_response": {
                                        "name": self._find_tool_name(messages, msg.tool_result.tool_call_id),
                                        "response": {"content": msg.tool_result.content},
                                    }
                                }
                            ],
                        }
                    )
                continue

            if msg.role == Role.ASSISTANT and msg.tool_calls:
                # Assistant message with tool calls → model role with function_call parts
                parts: list[dict[str, Any]] = []
                if msg.content:
                    if isinstance(msg.content, str):
                        parts.append({"text": msg.content})
                    else:
                        for block in msg.content:
                            if block.type == "text" and block.text is not None:
                                parts.append({"text": block.text})
                for tc in msg.tool_calls:
                    parts.append(
                        {
                            "function_call": {
                                "name": tc.name,
                                "args": tc.arguments,
                            }
                        }
                    )
                converted.append({"role": "model", "parts": parts})
                continue

            # Regular user / assistant messages
            role_str = "user" if msg.role == Role.USER else "model"
            if isinstance(msg.content, str):
                converted.append({"role": role_str, "parts": [{"text": msg.content}]})
            elif isinstance(msg.content, list):
                parts = []
                for block in msg.content:
                    if block.type == "text" and block.text is not None:
                        parts.append({"text": block.text})
                    # Gemini supports inline data for images but we focus on text here
                converted.append({"role": role_str, "parts": parts})
            # None content messages are skipped

        return system, converted

    # ------------------------------------------------------------------
    # Tool schema conversion
    # ------------------------------------------------------------------

    def _convert_tool(self, tool: ToolSchema) -> dict[str, Any]:
        """Convert a ToolSchema to Gemini FunctionDeclaration dict format."""
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse a Gemini SDK response into an LLMResponse."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        candidates = getattr(response, "candidates", [])
        finish_reason: str | None = None

        if candidates:
            candidate = candidates[0]
            finish_reason_raw = getattr(candidate, "finish_reason", None)
            # finish_reason may be an enum or string
            if finish_reason_raw is not None:
                finish_reason = str(finish_reason_raw).split(".")[-1]  # strip enum prefix
            content = getattr(candidate, "content", None)
            if content is not None:
                for part in getattr(content, "parts", []):
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        # Gemini doesn't provide a tool call ID; generate one.
                        tc_id = f"{fc.name}-{uuid.uuid4().hex[:8]}"
                        args = dict(fc.args) if hasattr(fc.args, "items") else {}
                        tool_calls.append(
                            ToolCall(id=tc_id, name=fc.name, arguments=args)
                        )
                    else:
                        text = getattr(part, "text", None)
                        if text:
                            text_parts.append(text)

        content_str = "".join(text_parts) if text_parts else None

        # Determine stop reason — tool calls always override
        stop_reason: StopReason
        if tool_calls:
            stop_reason = StopReason.TOOL_USE
        else:
            stop_reason = _map_stop_reason(finish_reason)

        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is not None:
            prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage_meta, "total_token_count", 0) or (
                prompt_tokens + completion_tokens
            )
        else:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return LLMResponse(
            content=content_str,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=self._model,
            provider=self.provider_name,
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    # Stream chunk parsing
    # ------------------------------------------------------------------

    def _parse_stream_chunk(self, raw_chunk: Any) -> StreamChunk | None:
        """Parse a single Gemini stream chunk into a StreamChunk, or None."""
        candidates = getattr(raw_chunk, "candidates", [])
        if not candidates:
            return None

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if content is None:
            return None

        for part in getattr(content, "parts", []):
            fc = getattr(part, "function_call", None)
            if fc is not None:
                args = dict(fc.args) if hasattr(fc.args, "items") else {}
                return StreamChunk(
                    tool_call_delta=ToolCallDelta(
                        index=0,
                        name=fc.name,
                        arguments_fragment=str(args),
                    )
                )
            text = getattr(part, "text", None)
            if text:
                return StreamChunk(content=text)

        # Chunk with no useful parts (e.g., just finish_reason signal)
        finish_reason_raw = getattr(candidate, "finish_reason", None)
        if finish_reason_raw is not None:
            finish_str = str(finish_reason_raw).split(".")[-1]
            stop_reason = _map_stop_reason(finish_str)
            return StreamChunk(stop_reason=stop_reason)

        return None

    # ------------------------------------------------------------------
    # Error mapping
    # ------------------------------------------------------------------

    def _map_error(self, exc: Exception) -> ProviderError:
        """Map a google.genai SDK exception to our error hierarchy.

        Gemini SDK errors carry a status_code attribute. We map by status code
        and class name so this works with both real SDK exceptions and mocks.
        """
        mro_names = {cls.__name__ for cls in type(exc).__mro__}

        # Class-name based checks first (highest priority)
        if "AuthenticationError" in mro_names:
            return AuthenticationError(str(exc), provider=self.provider_name)

        if "RateLimitError" in mro_names:
            return RateLimitError(str(exc), provider=self.provider_name)

        if "NotFoundError" in mro_names:
            return ModelNotFoundError(str(exc), provider=self.provider_name)

        if "TimeoutError" in mro_names or "APIConnectionError" in mro_names:
            return TimeoutError(str(exc), provider=self.provider_name)

        if "ServerError" in mro_names:
            return ServerError(str(exc), provider=self.provider_name)

        # Status code based fallback
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            if status_code == 401 or status_code == 403:
                return AuthenticationError(str(exc), provider=self.provider_name)
            if status_code == 429:
                return RateLimitError(str(exc), provider=self.provider_name)
            if status_code == 404:
                return ModelNotFoundError(str(exc), provider=self.provider_name)
            if status_code >= 500:
                return ServerError(str(exc), provider=self.provider_name)
            if status_code == 408 or status_code == 504:
                return TimeoutError(str(exc), provider=self.provider_name)
            # Other 4xx
            return ProviderError(str(exc), provider=self.provider_name, is_transient=False)

        # Fallback for unknown errors
        return ProviderError(str(exc), provider=self.provider_name)


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register_provider("gemini", GeminiProvider)
