"""Tests for LiteLLMProvider adapter.

Uses unittest.mock to avoid real API calls. Tests cover:
- provider_name attribute
- _resolve_api_key returns None (LiteLLM manages keys via env internally)
- Message conversion (system stays in list, tool_calls as JSON strings, tool results)
- Tool schema conversion (uses 'parameters' not 'input_schema')
- Response parsing (text, tool_calls with JSON arguments)
- Stream chunk parsing
- Error mapping
- Self-registration in provider registry
"""

from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    ToolResult,
    ToolSchema,
    Usage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_provider():
    """Import LiteLLMProvider lazily so we can control mock setup."""
    from anchor.llm.providers.litellm import LiteLLMProvider
    return LiteLLMProvider


def _make_provider(**kwargs):
    cls = _import_provider()
    defaults = {"model": "gpt-4o", "max_retries": 0}
    defaults.update(kwargs)
    return cls(**defaults)


def _make_tool_schema():
    return ToolSchema(
        name="get_weather",
        description="Get current weather",
        input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
    )


def _make_sdk_response(content=None, tool_calls=None, finish_reason="stop", usage=None, model="gpt-4o"):
    """Build a mock LiteLLM SDK response (OpenAI-compatible format)."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    if usage is None:
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.total_tokens = 15

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model = model
    return resp


# ---------------------------------------------------------------------------
# Test: provider_name
# ---------------------------------------------------------------------------

class TestProviderName:
    def test_provider_name_is_litellm(self):
        provider = _make_provider()
        assert provider.provider_name == "litellm"

    def test_model_id_includes_provider(self):
        provider = _make_provider()
        assert provider.model_id == "litellm/gpt-4o"

    def test_provider_name_with_routed_model(self):
        """LiteLLM routes using model strings like 'anthropic/claude-3-5-sonnet'."""
        provider = _make_provider(model="anthropic/claude-3-5-sonnet-20241022")
        assert provider.provider_name == "litellm"
        assert "anthropic/claude-3-5-sonnet-20241022" in provider.model_id


# ---------------------------------------------------------------------------
# Test: _resolve_api_key
# ---------------------------------------------------------------------------

class TestResolveApiKey:
    def test_resolve_api_key_returns_none(self):
        """LiteLLM handles API keys internally via env vars — we return None."""
        provider = _make_provider()
        assert provider._resolve_api_key() is None

    def test_api_key_is_none_by_default(self):
        """Without explicit api_key, _api_key should be None."""
        provider = _make_provider()
        assert provider._api_key is None

    def test_explicit_api_key_overrides(self):
        """An explicitly provided api_key should still be stored."""
        provider = _make_provider(api_key="test-key")
        assert provider._api_key == "test-key"


# ---------------------------------------------------------------------------
# Test: Message conversion
# ---------------------------------------------------------------------------

class TestMessageConversion:
    """Test _convert_messages() for LiteLLM (OpenAI-compatible) format."""

    def setup_method(self):
        self.provider = _make_provider()

    def test_system_message_stays_in_list(self):
        """LiteLLM (like OpenAI) keeps system messages in the messages list."""
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello"),
        ]
        converted = self.provider._convert_messages(messages)
        assert len(converted) == 2
        assert converted[0]["role"] == "system"
        assert converted[0]["content"] == "You are helpful."
        assert converted[1]["role"] == "user"

    def test_user_message_string_content(self):
        messages = [Message(role=Role.USER, content="Hello world")]
        converted = self.provider._convert_messages(messages)
        assert converted[0] == {"role": "user", "content": "Hello world"}

    def test_assistant_message_with_text(self):
        messages = [Message(role=Role.ASSISTANT, content="I can help.")]
        converted = self.provider._convert_messages(messages)
        assert converted[0] == {"role": "assistant", "content": "I can help."}

    def test_assistant_message_with_tool_calls(self):
        """Tool calls in assistant messages use JSON string for arguments."""
        messages = [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall(id="call_abc", name="get_weather", arguments={"location": "NYC"})
                ],
            )
        ]
        converted = self.provider._convert_messages(messages)
        msg = converted[0]
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        # Arguments must be a JSON string, not a dict
        assert isinstance(tc["function"]["arguments"], str)
        assert json.loads(tc["function"]["arguments"]) == {"location": "NYC"}

    def test_tool_result_message(self):
        """Tool results use 'tool' role with tool_call_id."""
        messages = [
            Message(
                role=Role.TOOL,
                tool_result=ToolResult(tool_call_id="call_abc", content="Sunny, 75F"),
            )
        ]
        converted = self.provider._convert_messages(messages)
        msg = converted[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_abc"
        assert msg["content"] == "Sunny, 75F"

    def test_multiple_tool_calls_on_assistant(self):
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="Using tools",
                tool_calls=[
                    ToolCall(id="call_1", name="search", arguments={"q": "foo"}),
                    ToolCall(id="call_2", name="weather", arguments={"city": "LA"}),
                ],
            )
        ]
        converted = self.provider._convert_messages(messages)
        msg = converted[0]
        assert len(msg["tool_calls"]) == 2
        assert msg["tool_calls"][0]["id"] == "call_1"
        assert msg["tool_calls"][1]["id"] == "call_2"

    def test_assistant_with_tool_calls_no_content(self):
        """When content is None on assistant with tool calls, no content key."""
        messages = [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall(id="call_x", name="search", arguments={"q": "test"})
                ],
            )
        ]
        converted = self.provider._convert_messages(messages)
        msg = converted[0]
        assert "tool_calls" in msg
        # content key should not be present when None
        assert "content" not in msg


# ---------------------------------------------------------------------------
# Test: Tool schema conversion
# ---------------------------------------------------------------------------

class TestToolSchemaConversion:
    def setup_method(self):
        self.provider = _make_provider()

    def test_converts_tool_schema_uses_parameters(self):
        """LiteLLM uses 'parameters' not 'input_schema'."""
        tool = _make_tool_schema()
        result = self.provider._convert_tool(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["description"] == "Get current weather"
        assert result["function"]["parameters"] == tool.input_schema
        # Must NOT use 'input_schema'
        assert "input_schema" not in result["function"]

    def test_converts_multiple_tools(self):
        tools = [_make_tool_schema(), _make_tool_schema()]
        results = [self.provider._convert_tool(t) for t in tools]
        assert len(results) == 2
        for r in results:
            assert r["type"] == "function"


# ---------------------------------------------------------------------------
# Test: Response parsing
# ---------------------------------------------------------------------------

class TestResponseParsing:
    def setup_method(self):
        self.provider = _make_provider()

    def test_parse_text_response(self):
        sdk_resp = _make_sdk_response(content="Hello there!", finish_reason="stop")
        result = self.provider._parse_response(sdk_resp)

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello there!"
        assert result.tool_calls is None
        assert result.stop_reason == StopReason.STOP
        assert result.provider == "litellm"
        assert result.model == "gpt-4o"

    def test_parse_tool_calls_response(self):
        """Tool call arguments come as JSON strings from LiteLLM."""
        tc = MagicMock()
        tc.id = "call_abc"
        func = MagicMock()
        func.name = "get_weather"
        func.arguments = json.dumps({"location": "NYC"})
        tc.function = func

        sdk_resp = _make_sdk_response(
            content=None, tool_calls=[tc], finish_reason="tool_calls"
        )
        result = self.provider._parse_response(sdk_resp)

        assert result.content is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        parsed_tc = result.tool_calls[0]
        assert parsed_tc.id == "call_abc"
        assert parsed_tc.name == "get_weather"
        # arguments must be parsed back to dict
        assert isinstance(parsed_tc.arguments, dict)
        assert parsed_tc.arguments == {"location": "NYC"}
        assert result.stop_reason == StopReason.TOOL_USE

    def test_parse_usage(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150
        sdk_resp = _make_sdk_response(content="ok", usage=usage)

        result = self.provider._parse_response(sdk_resp)
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150

    def test_stop_reason_length(self):
        sdk_resp = _make_sdk_response(content="truncated", finish_reason="length")
        result = self.provider._parse_response(sdk_resp)
        assert result.stop_reason == StopReason.MAX_TOKENS

    def test_stop_reason_tool_calls(self):
        tc = MagicMock()
        tc.id = "call_1"
        func = MagicMock()
        func.name = "search"
        func.arguments = json.dumps({"q": "test"})
        tc.function = func
        sdk_resp = _make_sdk_response(tool_calls=[tc], finish_reason="tool_calls")
        result = self.provider._parse_response(sdk_resp)
        assert result.stop_reason == StopReason.TOOL_USE

    def test_parse_multiple_tool_calls(self):
        tc1 = MagicMock()
        tc1.id = "call_1"
        func1 = MagicMock()
        func1.name = "search"
        func1.arguments = json.dumps({"q": "foo"})
        tc1.function = func1

        tc2 = MagicMock()
        tc2.id = "call_2"
        func2 = MagicMock()
        func2.name = "weather"
        func2.arguments = json.dumps({"city": "LA"})
        tc2.function = func2

        sdk_resp = _make_sdk_response(tool_calls=[tc1, tc2], finish_reason="tool_calls")
        result = self.provider._parse_response(sdk_resp)
        assert len(result.tool_calls) == 2

    def test_provider_name_in_response(self):
        sdk_resp = _make_sdk_response(content="hi")
        result = self.provider._parse_response(sdk_resp)
        assert result.provider == "litellm"


# ---------------------------------------------------------------------------
# Test: Stream chunk parsing
# ---------------------------------------------------------------------------

class TestStreamChunkParsing:
    def setup_method(self):
        self.provider = _make_provider()

    def _make_chunk(self, content=None, tool_calls=None, finish_reason=None):
        """Build a mock LiteLLM streaming chunk."""
        delta = MagicMock()
        delta.content = content
        delta.tool_calls = tool_calls

        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = finish_reason

        chunk = MagicMock()
        chunk.choices = [choice]
        return chunk

    def test_text_content_chunk(self):
        chunk = self._make_chunk(content="Hello")
        [result] = self.provider._parse_stream_chunks(chunk)
        assert result.content == "Hello"
        assert result.tool_call_delta is None

    def test_empty_content_returns_none(self):
        chunk = self._make_chunk(content=None)
        assert self.provider._parse_stream_chunks(chunk) == []

    def test_finish_reason_stop(self):
        chunk = self._make_chunk(content=None, finish_reason="stop")
        [result] = self.provider._parse_stream_chunks(chunk)
        assert result.stop_reason == StopReason.STOP

    def test_finish_reason_length(self):
        chunk = self._make_chunk(content=None, finish_reason="length")
        [result] = self.provider._parse_stream_chunks(chunk)
        assert result.stop_reason == StopReason.MAX_TOKENS

    def test_finish_reason_tool_calls(self):
        chunk = self._make_chunk(content=None, finish_reason="tool_calls")
        [result] = self.provider._parse_stream_chunks(chunk)
        assert result.stop_reason == StopReason.TOOL_USE

    def test_tool_call_delta_with_id_and_name(self):
        """First tool call chunk has id and name."""
        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.id = "call_abc"
        func_delta = MagicMock()
        func_delta.name = "get_weather"
        func_delta.arguments = ""
        tc_delta.function = func_delta

        chunk = self._make_chunk(tool_calls=[tc_delta])
        [result] = self.provider._parse_stream_chunks(chunk)
        assert result.tool_call_delta is not None
        assert result.tool_call_delta.index == 0
        assert result.tool_call_delta.id == "call_abc"
        assert result.tool_call_delta.name == "get_weather"

    def test_tool_call_delta_arguments_fragment(self):
        """Subsequent chunks have arguments fragments."""
        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.id = None
        func_delta = MagicMock()
        func_delta.name = None
        func_delta.arguments = '{"loc'
        tc_delta.function = func_delta

        chunk = self._make_chunk(tool_calls=[tc_delta])
        [result] = self.provider._parse_stream_chunks(chunk)
        assert result.tool_call_delta is not None
        assert result.tool_call_delta.arguments_fragment == '{"loc'

    def test_no_choices_returns_none(self):
        chunk = MagicMock()
        chunk.choices = []
        assert self.provider._parse_stream_chunks(chunk) == []


# ---------------------------------------------------------------------------
# Test: Error mapping
# ---------------------------------------------------------------------------

class TestErrorMapping:
    def setup_method(self):
        self.provider = _make_provider()

    def test_authentication_error_mapped(self):
        AuthErr = type("AuthenticationError", (Exception,), {})
        err = AuthErr("bad key")
        result = self.provider._map_error(err)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "litellm"
        assert result.is_transient is False

    def test_rate_limit_error_mapped(self):
        RateLimitErr = type("RateLimitError", (Exception,), {})
        err = RateLimitErr("rate limited")
        result = self.provider._map_error(err)
        assert isinstance(result, RateLimitError)
        assert result.is_transient is True

    def test_api_status_error_5xx_mapped_to_server_error(self):
        APIStatusErr = type("APIStatusError", (Exception,), {})
        err = APIStatusErr("server error")
        err.status_code = 500
        result = self.provider._map_error(err)
        assert isinstance(result, ServerError)
        assert result.is_transient is True

    def test_api_status_error_4xx_mapped_to_provider_error(self):
        APIStatusErr = type("APIStatusError", (Exception,), {})
        err = APIStatusErr("client error")
        err.status_code = 400
        result = self.provider._map_error(err)
        assert isinstance(result, ProviderError)
        assert result.is_transient is False

    def test_api_connection_error_mapped_to_timeout(self):
        APIConnErr = type("APIConnectionError", (Exception,), {})
        err = APIConnErr("connection failed")
        result = self.provider._map_error(err)
        assert isinstance(result, TimeoutError)
        assert result.is_transient is True

    def test_api_timeout_error_mapped(self):
        TimeoutErr = type("APITimeoutError", (Exception,), {})
        err = TimeoutErr("timed out")
        result = self.provider._map_error(err)
        assert isinstance(result, TimeoutError)
        assert result.is_transient is True

    def test_not_found_error_mapped(self):
        NotFoundErr = type("NotFoundError", (Exception,), {})
        err = NotFoundErr("model not found")
        result = self.provider._map_error(err)
        assert isinstance(result, ModelNotFoundError)
        assert result.is_transient is False

    def test_model_not_found_error_mapped(self):
        """LiteLLM has its own ModelNotFoundError type."""
        ModelNotFoundErr = type("ModelNotFoundError", (Exception,), {})
        err = ModelNotFoundErr("unknown model")
        result = self.provider._map_error(err)
        assert isinstance(result, ModelNotFoundError)
        assert result.is_transient is False

    def test_service_unavailable_error_mapped_to_server_error(self):
        ServiceUnavailableErr = type("ServiceUnavailableError", (Exception,), {})
        err = ServiceUnavailableErr("service unavailable")
        result = self.provider._map_error(err)
        assert isinstance(result, ServerError)
        assert result.is_transient is True

    def test_unknown_exception_mapped_to_provider_error(self):
        err = Exception("unexpected error")
        result = self.provider._map_error(err)
        assert isinstance(result, ProviderError)
        assert result.provider == "litellm"


# ---------------------------------------------------------------------------
# Test: _do_invoke integration
# ---------------------------------------------------------------------------

class TestDoInvoke:
    def setup_method(self):
        self.provider = _make_provider()

    @patch.dict(sys.modules, {"litellm": MagicMock()})
    def test_do_invoke_returns_llm_response(self):
        mock_litellm = sys.modules["litellm"]
        mock_litellm.completion.return_value = _make_sdk_response(content="Hello!")

        messages = [Message(role=Role.USER, content="Hi")]
        result = self.provider._do_invoke(messages, tools=None)

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.provider == "litellm"
        mock_litellm.completion.assert_called_once()

    @patch.dict(sys.modules, {"litellm": MagicMock()})
    def test_do_invoke_passes_model(self):
        mock_litellm = sys.modules["litellm"]
        mock_litellm.completion.return_value = _make_sdk_response(content="ok")

        messages = [Message(role=Role.USER, content="Hi")]
        self.provider._do_invoke(messages, tools=None)

        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    @patch.dict(sys.modules, {"litellm": MagicMock()})
    def test_do_invoke_passes_system_in_messages(self):
        """System message should appear in messages list."""
        mock_litellm = sys.modules["litellm"]
        mock_litellm.completion.return_value = _make_sdk_response(content="ok")

        messages = [
            Message(role=Role.SYSTEM, content="Be concise."),
            Message(role=Role.USER, content="Hi"),
        ]
        self.provider._do_invoke(messages, tools=None)

        call_kwargs = mock_litellm.completion.call_args[1]
        msgs = call_kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be concise."
        assert "system" not in call_kwargs

    @patch.dict(sys.modules, {"litellm": MagicMock()})
    def test_do_invoke_passes_tools(self):
        mock_litellm = sys.modules["litellm"]
        mock_litellm.completion.return_value = _make_sdk_response(content="ok")

        tools = [_make_tool_schema()]
        messages = [Message(role=Role.USER, content="Weather?")]
        self.provider._do_invoke(messages, tools=tools)

        call_kwargs = mock_litellm.completion.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["type"] == "function"
        assert call_kwargs["tools"][0]["function"]["name"] == "get_weather"

    @patch.dict(sys.modules, {"litellm": MagicMock()})
    def test_do_invoke_passes_temperature(self):
        mock_litellm = sys.modules["litellm"]
        mock_litellm.completion.return_value = _make_sdk_response(content="ok")

        messages = [Message(role=Role.USER, content="Hi")]
        self.provider._do_invoke(messages, tools=None, temperature=0.7)

        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    @patch.dict(sys.modules, {"litellm": MagicMock()})
    def test_do_invoke_maps_sdk_error(self):
        mock_litellm = sys.modules["litellm"]
        AuthErr = type("AuthenticationError", (Exception,), {})
        mock_litellm.completion.side_effect = AuthErr("bad key")

        messages = [Message(role=Role.USER, content="Hi")]
        with pytest.raises(AuthenticationError):
            self.provider._do_invoke(messages, tools=None)

    @patch.dict(sys.modules, {"litellm": MagicMock()})
    def test_do_invoke_no_stream_flag(self):
        """Regular invoke should not pass stream=True."""
        mock_litellm = sys.modules["litellm"]
        mock_litellm.completion.return_value = _make_sdk_response(content="ok")

        messages = [Message(role=Role.USER, content="Hi")]
        self.provider._do_invoke(messages, tools=None)

        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs.get("stream") is None or call_kwargs.get("stream") is False


# ---------------------------------------------------------------------------
# Test: _do_stream integration
# ---------------------------------------------------------------------------

class TestDoStream:
    def setup_method(self):
        self.provider = _make_provider()

    @patch.dict(sys.modules, {"litellm": MagicMock()})
    def test_do_stream_yields_text_chunks(self):
        mock_litellm = sys.modules["litellm"]

        def _make_chunk(content=None, finish_reason=None):
            delta = MagicMock()
            delta.content = content
            delta.tool_calls = None
            choice = MagicMock()
            choice.delta = delta
            choice.finish_reason = finish_reason
            chunk = MagicMock()
            chunk.choices = [choice]
            return chunk

        chunks_data = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" world"),
            _make_chunk(finish_reason="stop"),
        ]
        mock_litellm.completion.return_value = iter(chunks_data)

        messages = [Message(role=Role.USER, content="Hi")]
        result_chunks = list(self.provider._do_stream(messages, tools=None))

        content_chunks = [c for c in result_chunks if c.content is not None]
        assert len(content_chunks) == 2
        assert content_chunks[0].content == "Hello"
        assert content_chunks[1].content == " world"

        stop_chunks = [c for c in result_chunks if c.stop_reason is not None]
        assert len(stop_chunks) == 1
        assert stop_chunks[0].stop_reason == StopReason.STOP

    @patch.dict(sys.modules, {"litellm": MagicMock()})
    def test_do_stream_passes_stream_true(self):
        mock_litellm = sys.modules["litellm"]
        mock_litellm.completion.return_value = iter([])

        messages = [Message(role=Role.USER, content="Hi")]
        list(self.provider._do_stream(messages, tools=None))

        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["stream"] is True


# ---------------------------------------------------------------------------
# Test: _do_ainvoke integration
# ---------------------------------------------------------------------------

class TestDoAinvoke:
    def setup_method(self):
        self.provider = _make_provider()

    @pytest.mark.asyncio
    @patch.dict(sys.modules, {"litellm": MagicMock()})
    async def test_do_ainvoke_returns_llm_response(self):
        mock_litellm = sys.modules["litellm"]
        mock_litellm.acompletion = AsyncMock(
            return_value=_make_sdk_response(content="Async answer")
        )

        messages = [Message(role=Role.USER, content="Hi async")]
        result = await self.provider._do_ainvoke(messages, tools=None)

        assert isinstance(result, LLMResponse)
        assert result.content == "Async answer"
        assert result.provider == "litellm"

    @pytest.mark.asyncio
    @patch.dict(sys.modules, {"litellm": MagicMock()})
    async def test_do_ainvoke_calls_acompletion(self):
        """Async invoke should use acompletion, not completion."""
        mock_litellm = sys.modules["litellm"]
        mock_litellm.acompletion = AsyncMock(
            return_value=_make_sdk_response(content="ok")
        )
        mock_litellm.completion = MagicMock()

        messages = [Message(role=Role.USER, content="Hi")]
        await self.provider._do_ainvoke(messages, tools=None)

        mock_litellm.acompletion.assert_called_once()
        mock_litellm.completion.assert_not_called()

    @pytest.mark.asyncio
    @patch.dict(sys.modules, {"litellm": MagicMock()})
    async def test_do_ainvoke_maps_error(self):
        mock_litellm = sys.modules["litellm"]
        RateLimitErr = type("RateLimitError", (Exception,), {})
        mock_litellm.acompletion = AsyncMock(side_effect=RateLimitErr("rate limited"))

        messages = [Message(role=Role.USER, content="Hi")]
        with pytest.raises(RateLimitError):
            await self.provider._do_ainvoke(messages, tools=None)


# ---------------------------------------------------------------------------
# Test: _do_astream integration
# ---------------------------------------------------------------------------

class TestDoAstream:
    def setup_method(self):
        self.provider = _make_provider()

    @pytest.mark.asyncio
    @patch.dict(sys.modules, {"litellm": MagicMock()})
    async def test_do_astream_yields_text_chunks(self):
        mock_litellm = sys.modules["litellm"]

        def _make_chunk(content=None, finish_reason=None):
            delta = MagicMock()
            delta.content = content
            delta.tool_calls = None
            choice = MagicMock()
            choice.delta = delta
            choice.finish_reason = finish_reason
            chunk = MagicMock()
            chunk.choices = [choice]
            return chunk

        async def _async_gen():
            for c in [_make_chunk(content="Hi"), _make_chunk(finish_reason="stop")]:
                yield c

        mock_litellm.acompletion = AsyncMock(return_value=_async_gen())

        messages = [Message(role=Role.USER, content="Hi")]
        chunks = []
        async for chunk in self.provider._do_astream(messages, tools=None):
            chunks.append(chunk)

        content_chunks = [c for c in chunks if c.content is not None]
        assert len(content_chunks) == 1
        assert content_chunks[0].content == "Hi"

        stop_chunks = [c for c in chunks if c.stop_reason is not None]
        assert len(stop_chunks) == 1

    @pytest.mark.asyncio
    @patch.dict(sys.modules, {"litellm": MagicMock()})
    async def test_do_astream_passes_stream_true(self):
        mock_litellm = sys.modules["litellm"]

        async def _empty_gen():
            return
            yield  # make it an async generator

        mock_litellm.acompletion = AsyncMock(return_value=_empty_gen())

        messages = [Message(role=Role.USER, content="Hi")]
        async for _ in self.provider._do_astream(messages, tools=None):
            pass

        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert call_kwargs["stream"] is True


# ---------------------------------------------------------------------------
# Test: Self-registration
# ---------------------------------------------------------------------------

class TestSelfRegistration:
    def test_import_registers_provider(self):
        """Importing the module should register 'litellm' in the registry."""
        import anchor.llm.providers.litellm  # noqa: F401
        from anchor.llm.registry import _PROVIDERS
        assert "litellm" in _PROVIDERS

    def test_registry_maps_to_litellm_provider(self):
        """The registered class should be LiteLLMProvider."""
        import anchor.llm.providers.litellm  # noqa: F401
        from anchor.llm.providers.litellm import LiteLLMProvider
        from anchor.llm.registry import _PROVIDERS
        assert _PROVIDERS["litellm"] is LiteLLMProvider
