"""Tests for AnthropicProvider adapter.

Uses unittest.mock to avoid real API calls. Tests cover:
- provider_name attribute
- _resolve_api_key from env
- Message conversion (system extraction, content blocks, tool results)
- Tool schema conversion
- Response parsing (text, tool use, usage)
- Stream event parsing
- Error mapping
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anchor.llm.errors import (
    AuthenticationError,
    LLMTimeoutError,
    ModelNotFoundError,
    RateLimitError,
    ServerError,
)
from anchor.llm.models import (
    ContentBlock,
    LLMResponse,
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSchema,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_provider():
    """Import AnthropicProvider lazily so we can control mock setup."""
    from anchor.llm.providers.anthropic import AnthropicProvider
    return AnthropicProvider


def _make_provider(**kwargs):
    cls = _import_provider()
    defaults = {"model": "claude-3-5-sonnet-20241022", "api_key": "sk-test", "max_retries": 0}
    defaults.update(kwargs)
    return cls(**defaults)


def _make_tool_schema():
    return ToolSchema(
        name="get_weather",
        description="Get current weather",
        input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
    )


# ---------------------------------------------------------------------------
# Test: provider_name
# ---------------------------------------------------------------------------

class TestProviderName:
    def test_provider_name_is_anthropic(self):
        provider = _make_provider()
        assert provider.provider_name == "anthropic"

    def test_model_id_includes_provider(self):
        provider = _make_provider()
        assert provider.model_id == "anthropic/claude-3-5-sonnet-20241022"


# ---------------------------------------------------------------------------
# Test: _resolve_api_key
# ---------------------------------------------------------------------------

class TestResolveApiKey:
    def test_reads_from_env(self):
        cls = _import_provider()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            provider = cls(model="claude-3-5-sonnet-20241022", max_retries=0)
        assert provider._api_key == "env-key"

    def test_explicit_key_overrides_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            provider = _make_provider(api_key="explicit-key")
        assert provider._api_key == "explicit-key"

    def test_returns_none_when_env_not_set(self):
        cls = _import_provider()
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            provider = cls(model="some-model", max_retries=0)
        assert provider._api_key is None


# ---------------------------------------------------------------------------
# Test: Message conversion
# ---------------------------------------------------------------------------

class TestMessageConversion:
    """Test _convert_messages() and _extract_system()."""

    def setup_method(self):
        self.provider = _make_provider()

    def test_system_message_extracted(self):
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello"),
        ]
        system, converted = self.provider._extract_system_and_convert(messages)
        assert system == "You are helpful."
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_no_system_message(self):
        messages = [Message(role=Role.USER, content="Hello")]
        system, converted = self.provider._extract_system_and_convert(messages)
        assert system is None
        assert len(converted) == 1

    def test_user_message_string_content(self):
        messages = [Message(role=Role.USER, content="Hello world")]
        _, converted = self.provider._extract_system_and_convert(messages)
        assert converted[0] == {"role": "user", "content": "Hello world"}

    def test_user_message_content_blocks(self):
        messages = [
            Message(
                role=Role.USER,
                content=[ContentBlock(type="text", text="Hello")],
            )
        ]
        _, converted = self.provider._extract_system_and_convert(messages)
        assert converted[0]["role"] == "user"
        assert isinstance(converted[0]["content"], list)
        assert converted[0]["content"][0]["type"] == "text"
        assert converted[0]["content"][0]["text"] == "Hello"

    def test_assistant_message_with_text(self):
        messages = [Message(role=Role.ASSISTANT, content="I can help.")]
        _, converted = self.provider._extract_system_and_convert(messages)
        assert converted[0] == {"role": "assistant", "content": "I can help."}

    def test_assistant_message_with_tool_calls(self):
        messages = [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall(id="tc_1", name="get_weather", arguments={"location": "NYC"})
                ],
            )
        ]
        _, converted = self.provider._extract_system_and_convert(messages)
        msg = converted[0]
        assert msg["role"] == "assistant"
        content_blocks = msg["content"]
        assert len(content_blocks) == 1
        block = content_blocks[0]
        assert block["type"] == "tool_use"
        assert block["id"] == "tc_1"
        assert block["name"] == "get_weather"
        assert block["input"] == {"location": "NYC"}

    def test_tool_result_message(self):
        messages = [
            Message(
                role=Role.TOOL,
                tool_result=ToolResult(tool_call_id="tc_1", content="Sunny, 75F"),
            )
        ]
        _, converted = self.provider._extract_system_and_convert(messages)
        msg = converted[0]
        assert msg["role"] == "user"
        content_blocks = msg["content"]
        assert len(content_blocks) == 1
        block = content_blocks[0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "tc_1"
        assert block["content"] == "Sunny, 75F"


# ---------------------------------------------------------------------------
# Test: Tool schema conversion
# ---------------------------------------------------------------------------

class TestToolSchemaConversion:
    def setup_method(self):
        self.provider = _make_provider()

    def test_converts_tool_schema(self):
        tool = _make_tool_schema()
        result = self.provider._convert_tool(tool)
        assert result["name"] == "get_weather"
        assert result["description"] == "Get current weather"
        assert result["input_schema"] == tool.input_schema
        # Must use 'input_schema', not 'parameters'
        assert "input_schema" in result
        assert "parameters" not in result

    def test_converts_multiple_tools(self):
        tools = [_make_tool_schema(), _make_tool_schema()]
        results = [self.provider._convert_tool(t) for t in tools]
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Test: Response parsing
# ---------------------------------------------------------------------------

class TestResponseParsing:
    def setup_method(self):
        self.provider = _make_provider()

    def _make_sdk_response(self, content_blocks, stop_reason="end_turn", usage=None):
        """Build a mock Anthropic SDK response."""
        resp = MagicMock()
        resp.content = content_blocks
        resp.stop_reason = stop_reason
        resp.model = "claude-3-5-sonnet-20241022"
        if usage is None:
            usage = MagicMock()
            usage.input_tokens = 10
            usage.output_tokens = 5
        resp.usage = usage
        return resp

    def test_parse_text_response(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello there!"
        sdk_resp = self._make_sdk_response([text_block])

        result = self.provider._parse_response(sdk_resp)
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello there!"
        assert result.tool_calls is None
        assert result.stop_reason == StopReason.STOP
        assert result.provider == "anthropic"

    def test_parse_tool_use_response(self):
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tu_1"
        tool_block.name = "get_weather"
        tool_block.input = {"location": "NYC"}
        sdk_resp = self._make_sdk_response([tool_block], stop_reason="tool_use")

        result = self.provider._parse_response(sdk_resp)
        assert result.content is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "tu_1"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "NYC"}
        assert result.stop_reason == StopReason.TOOL_USE

    def test_parse_mixed_text_and_tool(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Using tool..."
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tu_2"
        tool_block.name = "search"
        tool_block.input = {"query": "python"}
        sdk_resp = self._make_sdk_response([text_block, tool_block], stop_reason="tool_use")

        result = self.provider._parse_response(sdk_resp)
        assert result.content == "Using tool..."
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1

    def test_parse_usage(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "ok"
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        sdk_resp = self._make_sdk_response([text_block], usage=usage)

        result = self.provider._parse_response(sdk_resp)
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150

    def test_stop_reason_max_tokens(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "truncated"
        sdk_resp = self._make_sdk_response([text_block], stop_reason="max_tokens")

        result = self.provider._parse_response(sdk_resp)
        assert result.stop_reason == StopReason.MAX_TOKENS

    def test_stop_reason_end_turn(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "done"
        sdk_resp = self._make_sdk_response([text_block], stop_reason="end_turn")

        result = self.provider._parse_response(sdk_resp)
        assert result.stop_reason == StopReason.STOP


# ---------------------------------------------------------------------------
# Test: Stream event parsing
# ---------------------------------------------------------------------------

class TestStreamEventParsing:
    def setup_method(self):
        self.provider = _make_provider()

    def test_text_delta_event(self):
        event = MagicMock()
        event.type = "content_block_delta"
        delta = MagicMock()
        delta.type = "text_delta"
        delta.text = "Hello"
        event.delta = delta
        event.index = 0

        chunk = self.provider._parse_stream_event(event)
        assert chunk is not None
        assert chunk.content == "Hello"
        assert chunk.tool_call_delta is None

    def test_tool_use_start_event(self):
        event = MagicMock()
        event.type = "content_block_start"
        block = MagicMock()
        block.type = "tool_use"
        block.id = "tu_1"
        block.name = "get_weather"
        event.content_block = block
        event.index = 0

        chunk = self.provider._parse_stream_event(event)
        assert chunk is not None
        assert chunk.tool_call_delta is not None
        assert chunk.tool_call_delta.index == 0
        assert chunk.tool_call_delta.id == "tu_1"
        assert chunk.tool_call_delta.name == "get_weather"

    def test_input_json_delta_event(self):
        event = MagicMock()
        event.type = "content_block_delta"
        delta = MagicMock()
        delta.type = "input_json_delta"
        delta.partial_json = '{"loc'
        event.delta = delta
        event.index = 1

        chunk = self.provider._parse_stream_event(event)
        assert chunk is not None
        assert chunk.tool_call_delta is not None
        assert chunk.tool_call_delta.index == 1
        assert chunk.tool_call_delta.arguments_fragment == '{"loc'

    def test_message_delta_stop_event(self):
        event = MagicMock()
        event.type = "message_delta"
        delta = MagicMock()
        delta.stop_reason = "end_turn"
        event.delta = delta
        usage = MagicMock()
        usage.output_tokens = 25
        event.usage = usage

        chunk = self.provider._parse_stream_event(event)
        assert chunk is not None
        assert chunk.stop_reason == StopReason.STOP

    def test_message_delta_tool_use_stop(self):
        event = MagicMock()
        event.type = "message_delta"
        delta = MagicMock()
        delta.stop_reason = "tool_use"
        event.delta = delta
        event.usage = MagicMock()
        event.usage.output_tokens = 10

        chunk = self.provider._parse_stream_event(event)
        assert chunk is not None
        assert chunk.stop_reason == StopReason.TOOL_USE

    def test_unknown_event_returns_none(self):
        event = MagicMock()
        event.type = "message_start"

        chunk = self.provider._parse_stream_event(event)
        assert chunk is None

    def test_text_content_block_start_returns_none(self):
        """content_block_start for text type has no useful data yet."""
        event = MagicMock()
        event.type = "content_block_start"
        block = MagicMock()
        block.type = "text"
        event.content_block = block
        event.index = 0

        chunk = self.provider._parse_stream_event(event)
        assert chunk is None


# ---------------------------------------------------------------------------
# Test: Error mapping
# ---------------------------------------------------------------------------

class TestErrorMapping:
    def setup_method(self):
        self.provider = _make_provider()

    def _make_anthropic_error(self, cls_name, status_code=None, message="error"):
        """Create a mock Anthropic SDK exception."""
        err = MagicMock(spec=Exception)
        err.args = (message,)
        err.__str__ = lambda self: message
        if status_code is not None:
            err.status_code = status_code
        return err

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_authentication_error_mapped(self, mock_anthropic):
        mock_anthropic.AuthenticationError = type("AuthenticationError", (Exception,), {})
        actual_err = mock_anthropic.AuthenticationError("auth failed")

        result = self.provider._map_error(actual_err)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "anthropic"

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_rate_limit_error_mapped(self, mock_anthropic):
        mock_anthropic.RateLimitError = type("RateLimitError", (Exception,), {})
        err = mock_anthropic.RateLimitError("rate limited")

        result = self.provider._map_error(err)
        assert isinstance(result, RateLimitError)
        assert result.is_transient is True

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_api_status_error_5xx_mapped_to_server_error(self, mock_anthropic):
        mock_anthropic.APIStatusError = type("APIStatusError", (Exception,), {})
        err = mock_anthropic.APIStatusError("server error")
        err.status_code = 500

        result = self.provider._map_error(err)
        assert isinstance(result, ServerError)
        assert result.is_transient is True

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_api_status_error_4xx_not_mapped_to_server_error(self, mock_anthropic):
        mock_anthropic.APIStatusError = type("APIStatusError", (Exception,), {})
        err = mock_anthropic.APIStatusError("client error")
        err.status_code = 400

        # A generic 4xx that isn't auth/rate-limit/not-found should still return ProviderError
        result = self.provider._map_error(err)
        from anchor.llm.errors import ProviderError
        assert isinstance(result, ProviderError)

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_connection_error_mapped_to_timeout(self, mock_anthropic):
        mock_anthropic.APIConnectionError = type("APIConnectionError", (Exception,), {})
        err = mock_anthropic.APIConnectionError("connection failed")

        result = self.provider._map_error(err)
        assert isinstance(result, LLMTimeoutError)
        assert result.is_transient is True

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_timeout_error_mapped(self, mock_anthropic):
        mock_anthropic.APITimeoutError = type("APITimeoutError", (Exception,), {})
        err = mock_anthropic.APITimeoutError("timed out")

        result = self.provider._map_error(err)
        assert isinstance(result, LLMTimeoutError)

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_not_found_error_mapped(self, mock_anthropic):
        mock_anthropic.NotFoundError = type("NotFoundError", (Exception,), {})
        err = mock_anthropic.NotFoundError("model not found")

        result = self.provider._map_error(err)
        assert isinstance(result, ModelNotFoundError)
        assert result.is_transient is False


# ---------------------------------------------------------------------------
# Test: _do_invoke integration (mocking SDK client)
# ---------------------------------------------------------------------------

class TestDoInvoke:
    def setup_method(self):
        self.provider = _make_provider()

    def _build_sdk_response(self, text="Answer"):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        usage = MagicMock()
        usage.input_tokens = 10
        usage.output_tokens = 5
        resp = MagicMock()
        resp.content = [text_block]
        resp.stop_reason = "end_turn"
        resp.model = "claude-3-5-sonnet-20241022"
        resp.usage = usage
        return resp

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_do_invoke_returns_llm_response(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = self._build_sdk_response("Hello!")

        messages = [Message(role=Role.USER, content="Hi")]
        result = self.provider._do_invoke(messages, tools=None)

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.provider == "anthropic"
        mock_client.messages.create.assert_called_once()

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_do_invoke_passes_system(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = self._build_sdk_response()

        messages = [
            Message(role=Role.SYSTEM, content="Be concise."),
            Message(role=Role.USER, content="Hi"),
        ]
        self.provider._do_invoke(messages, tools=None)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs.get("system") == "Be concise."

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_do_invoke_maps_sdk_error(self, mock_anthropic):
        mock_anthropic.AuthenticationError = type("AuthenticationError", (Exception,), {})
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = mock_anthropic.AuthenticationError("bad key")

        messages = [Message(role=Role.USER, content="Hi")]
        with pytest.raises(AuthenticationError):
            self.provider._do_invoke(messages, tools=None)

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_do_invoke_passes_tools(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = self._build_sdk_response()

        tools = [_make_tool_schema()]
        messages = [Message(role=Role.USER, content="What's the weather?")]
        self.provider._do_invoke(messages, tools=tools)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["name"] == "get_weather"
        assert "input_schema" in call_kwargs["tools"][0]


# ---------------------------------------------------------------------------
# Test: _do_stream integration
# ---------------------------------------------------------------------------

class TestDoStream:
    def setup_method(self):
        self.provider = _make_provider()

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_do_stream_yields_chunks(self, mock_anthropic):
        # Build fake events
        text_event = MagicMock()
        text_event.type = "content_block_delta"
        delta = MagicMock()
        delta.type = "text_delta"
        delta.text = "Hello"
        text_event.delta = delta
        text_event.index = 0

        stop_event = MagicMock()
        stop_event.type = "message_delta"
        stop_delta = MagicMock()
        stop_delta.stop_reason = "end_turn"
        stop_event.delta = stop_delta
        stop_event.usage = MagicMock()
        stop_event.usage.output_tokens = 5

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([text_event, stop_event]))
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.stream.return_value.__enter__ = MagicMock(return_value=mock_stream)
        mock_client.messages.stream.return_value.__exit__ = MagicMock(return_value=False)

        messages = [Message(role=Role.USER, content="Hi")]
        chunks = list(self.provider._do_stream(messages, tools=None))

        assert any(c.content == "Hello" for c in chunks)
        stop_chunks = [c for c in chunks if c.stop_reason is not None]
        assert len(stop_chunks) == 1
        assert stop_chunks[0].stop_reason == StopReason.STOP


# ---------------------------------------------------------------------------
# Test: _do_ainvoke integration
# ---------------------------------------------------------------------------

class TestDoAinvoke:
    def setup_method(self):
        self.provider = _make_provider()

    @patch("anchor.llm.providers.anthropic.anthropic")
    @pytest.mark.asyncio
    async def test_do_ainvoke_returns_llm_response(self, mock_anthropic):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Async answer"
        usage = MagicMock()
        usage.input_tokens = 10
        usage.output_tokens = 5
        resp = MagicMock()
        resp.content = [text_block]
        resp.stop_reason = "end_turn"
        resp.model = "claude-3-5-sonnet-20241022"
        resp.usage = usage

        mock_async_client = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_async_client
        mock_async_client.messages.create = AsyncMock(return_value=resp)

        messages = [Message(role=Role.USER, content="Hi async")]
        result = await self.provider._do_ainvoke(messages, tools=None)

        assert isinstance(result, LLMResponse)
        assert result.content == "Async answer"


# ---------------------------------------------------------------------------
# Test: Self-registration
# ---------------------------------------------------------------------------

class TestSelfRegistration:
    def test_import_registers_provider(self):
        """Importing the module should register 'anthropic' in the registry."""
        import anchor.llm.providers.anthropic  # noqa: F401
        from anchor.llm.registry import _PROVIDERS
        assert "anthropic" in _PROVIDERS
