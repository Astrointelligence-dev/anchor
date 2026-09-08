"""Tests for anchor.llm.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anchor.llm.models import (
    ContentBlock,
    LLMResponse,
    Message,
    Role,
    StopReason,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    ToolSchema,
    Usage,
)


class TestRole:
    def test_role_values(self):
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"


class TestContentBlock:
    def test_text_block(self):
        block = ContentBlock(type="text", text="hello")
        assert block.type == "text"
        assert block.text == "hello"
        assert block.image_url is None

    def test_image_url_block(self):
        block = ContentBlock(type="image_url", image_url="https://example.com/img.png")
        assert block.type == "image_url"
        assert block.image_url == "https://example.com/img.png"

    def test_image_base64_block(self):
        block = ContentBlock(
            type="image_base64",
            image_base64="aGVsbG8=",
            media_type="image/png",
        )
        assert block.media_type == "image/png"

    def test_frozen(self):
        block = ContentBlock(type="text", text="hello")
        with pytest.raises(ValidationError):
            block.text = "world"


class TestToolCall:
    def test_creation(self):
        tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "NYC"})
        assert tc.id == "call_1"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "NYC"}

    def test_frozen(self):
        tc = ToolCall(id="call_1", name="get_weather", arguments={})
        with pytest.raises(ValidationError):
            tc.name = "other"


class TestToolCallDelta:
    def test_first_delta(self):
        delta = ToolCallDelta(index=0, id="call_1", name="get_weather")
        assert delta.index == 0
        assert delta.id == "call_1"
        assert delta.arguments_fragment is None

    def test_argument_fragment(self):
        delta = ToolCallDelta(index=0, arguments_fragment='{"city":')
        assert delta.arguments_fragment == '{"city":'


class TestToolResult:
    def test_creation(self):
        tr = ToolResult(tool_call_id="call_1", content="sunny")
        assert tr.is_error is False

    def test_error_result(self):
        tr = ToolResult(tool_call_id="call_1", content="failed", is_error=True)
        assert tr.is_error is True


class TestMessage:
    def test_user_message_string(self):
        msg = Message(role=Role.USER, content="hello")
        assert msg.role == Role.USER
        assert msg.content == "hello"
        assert msg.tool_calls is None

    def test_user_message_content_blocks(self):
        blocks = [ContentBlock(type="text", text="hello")]
        msg = Message(role=Role.USER, content=blocks)
        assert isinstance(msg.content, list)

    def test_assistant_with_tool_calls(self):
        tc = ToolCall(id="c1", name="fn", arguments={})
        msg = Message(role=Role.ASSISTANT, content="thinking...", tool_calls=[tc])
        assert len(msg.tool_calls) == 1

    def test_tool_message(self):
        tr = ToolResult(tool_call_id="c1", content="result")
        msg = Message(role=Role.TOOL, tool_result=tr)
        assert msg.tool_result.content == "result"


class TestUsage:
    def test_creation(self):
        u = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert u.total_cost is None

    def test_with_cost(self):
        u = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, total_cost=0.001)
        assert u.total_cost == 0.001


class TestLLMResponse:
    def test_text_response(self):
        r = LLMResponse(
            content="hello",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="gpt-4o",
            provider="openai",
            stop_reason=StopReason.STOP,
        )
        assert r.content == "hello"
        assert r.tool_calls is None

    def test_tool_use_response(self):
        tc = ToolCall(id="c1", name="fn", arguments={"x": 1})
        r = LLMResponse(
            tool_calls=[tc],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            stop_reason=StopReason.TOOL_USE,
        )
        assert r.stop_reason == StopReason.TOOL_USE
        assert len(r.tool_calls) == 1


class TestStreamChunk:
    def test_content_chunk(self):
        c = StreamChunk(content="hello")
        assert c.content == "hello"
        assert c.usage is None

    def test_final_chunk(self):
        c = StreamChunk(
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            stop_reason=StopReason.STOP,
        )
        assert c.stop_reason == StopReason.STOP

    def test_tool_call_delta_chunk(self):
        delta = ToolCallDelta(index=0, id="c1", name="fn")
        c = StreamChunk(tool_call_delta=delta)
        assert c.tool_call_delta.name == "fn"


class TestToolSchema:
    def test_creation(self):
        ts = ToolSchema(
            name="get_weather",
            description="Get weather for a city",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        assert ts.name == "get_weather"
