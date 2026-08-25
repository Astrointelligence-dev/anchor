"""Shared helpers for OpenAI-compatible providers (OpenAI, LiteLLM, etc.).

These functions convert between Anchor's unified models and the OpenAI
Chat Completions wire format. They are stateless and operate purely on
the data passed in, so they can be called from any provider.
"""

from __future__ import annotations

import json
from typing import Any

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


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------

_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.STOP,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_USE,
}


def map_stop_reason(finish_reason: str | None) -> StopReason:
    """Map an OpenAI-style finish_reason string to a unified StopReason."""
    if finish_reason is None:
        return StopReason.STOP
    return _STOP_REASON_MAP.get(finish_reason, StopReason.STOP)


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert Anchor messages to OpenAI Chat Completions format.

    Key conventions:
    - System messages stay in the messages list (no extraction)
    - Tool results use role='tool' with tool_call_id
    - Assistant tool calls use JSON strings for arguments
    """
    converted: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            converted.append({"role": "system", "content": msg.content or ""})
            continue

        if msg.role == Role.TOOL:
            # Tool result -> role='tool' with tool_call_id
            if msg.tool_result is not None:
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_result.tool_call_id,
                        "content": msg.tool_result.content,
                    }
                )
            continue

        if msg.role == Role.ASSISTANT and msg.tool_calls:
            # Assistant message with tool calls
            oai_msg: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                oai_msg["content"] = msg.content
            oai_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        # OpenAI requires arguments as a JSON string
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
            converted.append(oai_msg)
            continue

        # Regular user / assistant messages
        role_str = "user" if msg.role == Role.USER else "assistant"
        if isinstance(msg.content, str):
            converted.append({"role": role_str, "content": msg.content})
        elif isinstance(msg.content, list):
            # Content blocks -- for now pass as text (full multimodal support
            # can be extended later)
            blocks: list[dict[str, Any]] = []
            for block in msg.content:
                if block.type == "text" and block.text is not None:
                    blocks.append({"type": "text", "text": block.text})
                elif block.type == "image_url" and block.image_url is not None:
                    blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": block.image_url},
                        }
                    )
                elif block.type == "image_base64" and block.image_base64 is not None:
                    media_type = block.media_type or "image/png"
                    blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{block.image_base64}"
                            },
                        }
                    )
            converted.append({"role": role_str, "content": blocks})

    return converted


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------


def convert_tool(tool: ToolSchema) -> dict[str, Any]:
    """Convert a ToolSchema to OpenAI tool definition format.

    OpenAI uses 'parameters' (not 'input_schema' like Anthropic).
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


_TOOL_CHOICE_MODES = {"auto": "auto", "any": "required", "none": "none"}


def convert_tool_choice(tool_choice: str | dict[str, Any]) -> str | dict[str, Any]:
    """Map the generic tool_choice (Anthropic shape) to OpenAI's."""
    if isinstance(tool_choice, str):
        mapped = _TOOL_CHOICE_MODES.get(tool_choice)
        if mapped is None:
            msg = (
                f"Invalid tool_choice '{tool_choice}': expected one of "
                f"{sorted(_TOOL_CHOICE_MODES)} or {{'type': 'tool', 'name': ...}}"
            )
            raise ValueError(msg)
        return mapped
    return {"type": "function", "function": {"name": tool_choice.get("name", "")}}


def build_call_kwargs(
    model: str,
    converted: list[dict[str, Any]],
    tools: list[ToolSchema] | None,
    *,
    stream: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the shared Chat Completions kwargs (openai/litellm family)."""
    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": converted,
        "max_tokens": kwargs.get("max_tokens", 4096),
    }
    if stream:
        call_kwargs["stream"] = True
    if tools:
        call_kwargs["tools"] = [convert_tool(t) for t in tools]
    if kwargs.get("temperature") is not None:
        call_kwargs["temperature"] = kwargs["temperature"]
    if kwargs.get("stop"):
        call_kwargs["stop"] = kwargs["stop"]
    if kwargs.get("tool_choice") is not None and tools:
        call_kwargs["tool_choice"] = convert_tool_choice(kwargs["tool_choice"])
    return call_kwargs


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_response(response: Any, provider_name: str) -> LLMResponse:
    """Parse an OpenAI-compatible response into an LLMResponse."""
    choice = response.choices[0]
    message = choice.message

    content = message.content if message.content else None

    tool_calls: list[ToolCall] = []
    if message.tool_calls:
        for tc in message.tool_calls:
            # arguments comes as a JSON string from OpenAI
            try:
                arguments = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                )
            )

    usage = Usage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )

    return LLMResponse(
        content=content,
        tool_calls=tool_calls if tool_calls else None,
        usage=usage,
        model=response.model,
        provider=provider_name,
        stop_reason=map_stop_reason(choice.finish_reason),
    )


# ---------------------------------------------------------------------------
# Stream chunk parsing
# ---------------------------------------------------------------------------


def parse_stream_chunk(chunk: Any) -> StreamChunk | None:
    """Parse a single OpenAI-compatible stream chunk into a StreamChunk, or None."""
    if not chunk.choices:
        return None

    choice = chunk.choices[0]
    delta = choice.delta
    finish_reason = choice.finish_reason

    # Handle finish reason first (can combine with usage)
    if finish_reason is not None:
        return StreamChunk(stop_reason=map_stop_reason(finish_reason))

    # Handle tool call deltas
    if delta.tool_calls:
        tc_delta = delta.tool_calls[0]
        func_delta = tc_delta.function
        return StreamChunk(
            tool_call_delta=ToolCallDelta(
                index=tc_delta.index,
                id=tc_delta.id if tc_delta.id else None,
                name=func_delta.name if func_delta.name else None,
                arguments_fragment=func_delta.arguments if func_delta.arguments else None,
            )
        )

    # Handle text content
    if delta.content is not None:
        return StreamChunk(content=delta.content)

    return None
