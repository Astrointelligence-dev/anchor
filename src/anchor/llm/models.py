"""Unified models for the multi-provider LLM layer.

These models are the interface between Anchor and any LLM provider.
Provider adapters convert these to/from provider-specific formats.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class Role(StrEnum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentBlock(BaseModel, frozen=True):
    """A single block of content within a message."""

    type: str  # "text", "image_url", "image_base64"
    text: str | None = None
    image_url: str | None = None
    image_base64: str | None = None
    media_type: str | None = None  # e.g. "image/png"


class ToolCall(BaseModel, frozen=True):
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolCallDelta(BaseModel, frozen=True):
    """Incremental tool call data during streaming."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments_fragment: str | None = None


class ToolResult(BaseModel, frozen=True):
    """Result of executing a tool call."""

    tool_call_id: str
    content: str
    is_error: bool = False


class Message(BaseModel, frozen=True):
    """A single message in a conversation."""

    role: Role
    content: str | list[ContentBlock] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None
    name: str | None = None


class Usage(BaseModel, frozen=True):
    """Token usage and cost for an LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float | None = None


class StopReason(StrEnum):
    """Why the model stopped generating."""

    STOP = "stop"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"


class LLMResponse(BaseModel, frozen=True):
    """Complete response from an LLM provider."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage
    model: str
    provider: str
    stop_reason: StopReason


class StreamChunk(BaseModel, frozen=True):
    """A single chunk from a streaming LLM response."""

    content: str | None = None
    tool_call_delta: ToolCallDelta | None = None
    usage: Usage | None = None
    stop_reason: StopReason | None = None


class ToolSchema(BaseModel, frozen=True):
    """Provider-agnostic tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]
    input_examples: tuple[dict[str, Any], ...] = ()
