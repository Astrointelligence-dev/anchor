"""Tests for GeminiProvider adapter.

Uses unittest.mock to avoid real API calls. Tests cover:
- provider_name attribute
- _resolve_api_key from GOOGLE_API_KEY and GEMINI_API_KEY env vars
- Message conversion (system extraction, user/assistant, tool calls, tool results)
- Tool schema conversion
- Response parsing (text, tool use, usage, finish reason)
- Stream chunk parsing
- Error mapping
- Self-registration
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anchor.llm.errors import (
    AuthenticationError,
    ModelNotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_provider():
    """Import GeminiProvider lazily so module-level mock setup works."""
    from anchor.llm.providers.gemini import GeminiProvider
    return GeminiProvider


def _make_provider(**kwargs):
    cls = _import_provider()
    defaults = {"model": "gemini-2.0-flash", "api_key": "gm-test-key", "max_retries": 0}
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
    def test_provider_name_is_gemini(self):
        provider = _make_provider()
        assert provider.provider_name == "gemini"

    def test_model_id_includes_provider(self):
        provider = _make_provider()
        assert provider.model_id == "gemini/gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Test: _resolve_api_key
# ---------------------------------------------------------------------------


class TestResolveApiKey:
    def test_reads_google_api_key_from_env(self):
        cls = _import_provider()
        env = {k: v for k, v in os.environ.items()
               if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
        env["GOOGLE_API_KEY"] = "google-key"
        with patch.dict(os.environ, env, clear=True):
            provider = cls(model="gemini-2.0-flash", max_retries=0)
        assert provider._api_key == "google-key"

    def test_reads_gemini_api_key_from_env(self):
        cls = _import_provider()
        env = {k: v for k, v in os.environ.items()
               if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
        env["GEMINI_API_KEY"] = "gemini-key"
        with patch.dict(os.environ, env, clear=True):
            provider = cls(model="gemini-2.0-flash", max_retries=0)
        assert provider._api_key == "gemini-key"

    def test_google_api_key_takes_priority_over_gemini_api_key(self):
        cls = _import_provider()
        env = {k: v for k, v in os.environ.items()
               if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
        env["GOOGLE_API_KEY"] = "google-key"
        env["GEMINI_API_KEY"] = "gemini-key"
        with patch.dict(os.environ, env, clear=True):
            provider = cls(model="gemini-2.0-flash", max_retries=0)
        assert provider._api_key == "google-key"

    def test_explicit_key_overrides_env(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
            provider = _make_provider(api_key="explicit-key")
        assert provider._api_key == "explicit-key"

    def test_returns_none_when_env_not_set(self):
        cls = _import_provider()
        env = {k: v for k, v in os.environ.items()
               if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            provider = cls(model="gemini-2.0-flash", max_retries=0)
        assert provider._api_key is None


# ---------------------------------------------------------------------------
# Test: Message conversion
# ---------------------------------------------------------------------------


class TestMessageConversion:
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

    def test_no_system_message(self):
        messages = [Message(role=Role.USER, content="Hello")]
        system, converted = self.provider._extract_system_and_convert(messages)
        assert system is None
        assert len(converted) == 1

    def test_user_message_role(self):
        messages = [Message(role=Role.USER, content="Hello world")]
        _, converted = self.provider._extract_system_and_convert(messages)
        msg = converted[0]
        assert msg["role"] == "user"

    def test_user_message_has_text_part(self):
        messages = [Message(role=Role.USER, content="Hello world")]
        _, converted = self.provider._extract_system_and_convert(messages)
        msg = converted[0]
        assert "parts" in msg
        assert len(msg["parts"]) == 1

    def test_assistant_message_role_is_model(self):
        """Gemini uses 'model' for assistant messages."""
        messages = [Message(role=Role.ASSISTANT, content="I can help.")]
        _, converted = self.provider._extract_system_and_convert(messages)
        msg = converted[0]
        assert msg["role"] == "model"

    def test_assistant_message_has_text_part(self):
        messages = [Message(role=Role.ASSISTANT, content="I can help.")]
        _, converted = self.provider._extract_system_and_convert(messages)
        msg = converted[0]
        assert "parts" in msg
        assert len(msg["parts"]) == 1

    def test_assistant_with_tool_calls(self):
        """Tool calls convert to function_call parts."""
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
        assert msg["role"] == "model"
        parts = msg["parts"]
        assert len(parts) == 1
        part = parts[0]
        assert part["function_call"]["name"] == "get_weather"
        assert part["function_call"]["args"] == {"location": "NYC"}

    def test_assistant_with_text_and_tool_calls(self):
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="Using tool...",
                tool_calls=[
                    ToolCall(id="tc_2", name="search", arguments={"query": "python"})
                ],
            )
        ]
        _, converted = self.provider._extract_system_and_convert(messages)
        msg = converted[0]
        parts = msg["parts"]
        # Should have text part + function_call part
        assert len(parts) == 2
        text_parts = [p for p in parts if "text" in p]
        func_parts = [p for p in parts if "function_call" in p]
        assert len(text_parts) == 1
        assert len(func_parts) == 1

    def test_tool_result_message_role_is_user(self):
        """Tool results are wrapped in a user message in Gemini format."""
        messages = [
            Message(
                role=Role.TOOL,
                tool_result=ToolResult(tool_call_id="tc_1", content="Sunny, 75F"),
            )
        ]
        _, converted = self.provider._extract_system_and_convert(messages)
        msg = converted[0]
        assert msg["role"] == "user"

    def test_tool_result_message_has_function_response_part(self):
        messages = [
            Message(
                role=Role.TOOL,
                tool_result=ToolResult(tool_call_id="tc_1", content="Sunny, 75F"),
            )
        ]
        _, converted = self.provider._extract_system_and_convert(messages)
        msg = converted[0]
        parts = msg["parts"]
        assert len(parts) == 1
        part = parts[0]
        assert "function_response" in part
        assert part["function_response"]["name"] == "tc_1"


# ---------------------------------------------------------------------------
# Test: Tool schema conversion
# ---------------------------------------------------------------------------


class TestToolSchemaConversion:
    def setup_method(self):
        self.provider = _make_provider()

    def test_converts_tool_name(self):
        tool = _make_tool_schema()
        result = self.provider._convert_tool(tool)
        assert result["name"] == "get_weather"

    def test_converts_tool_description(self):
        tool = _make_tool_schema()
        result = self.provider._convert_tool(tool)
        assert result["description"] == "Get current weather"

    def test_converts_tool_parameters(self):
        tool = _make_tool_schema()
        result = self.provider._convert_tool(tool)
        # Gemini uses 'parameters', not 'input_schema'
        assert "parameters" in result
        assert result["parameters"] == tool.input_schema


# ---------------------------------------------------------------------------
# Helpers: Build mock SDK responses
# ---------------------------------------------------------------------------


def _make_text_part(text: str) -> MagicMock:
    part = MagicMock()
    part.text = text
    part.function_call = None
    return part


def _make_function_call_part(name: str, args: dict) -> MagicMock:
    part = MagicMock()
    part.text = None
    fc = MagicMock()
    fc.name = name
    fc.args = args
    part.function_call = fc
    return part


def _make_candidate(parts, finish_reason="STOP") -> MagicMock:
    candidate = MagicMock()
    content = MagicMock()
    content.parts = parts
    candidate.content = content
    candidate.finish_reason = finish_reason
    return candidate


def _make_usage(prompt_tokens=10, completion_tokens=5) -> MagicMock:
    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = completion_tokens
    usage.total_token_count = prompt_tokens + completion_tokens
    return usage


def _make_sdk_response(parts, finish_reason="STOP", usage=None,
                       model="gemini-2.0-flash") -> MagicMock:
    resp = MagicMock()
    candidate = _make_candidate(parts, finish_reason)
    resp.candidates = [candidate]
    resp.usage_metadata = usage or _make_usage()
    resp.model = model
    # resp.text shortcut — may raise if no text parts; set via property or stub
    if any(getattr(p, "text", None) for p in parts):
        resp.text = " ".join(
            p.text for p in parts if getattr(p, "text", None)
        )
    else:
        resp.text = None
    return resp


# ---------------------------------------------------------------------------
# Test: Response parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def setup_method(self):
        self.provider = _make_provider()

    def test_parse_text_response(self):
        parts = [_make_text_part("Hello there!")]
        sdk_resp = _make_sdk_response(parts)

        result = self.provider._parse_response(sdk_resp)
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello there!"
        assert result.tool_calls is None
        assert result.stop_reason == StopReason.STOP
        assert result.provider == "gemini"

    def test_parse_tool_call_response(self):
        parts = [_make_function_call_part("get_weather", {"location": "NYC"})]
        sdk_resp = _make_sdk_response(parts, finish_reason="STOP")

        result = self.provider._parse_response(sdk_resp)
        assert result.content is None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "NYC"}

    def test_parse_mixed_text_and_tool_call(self):
        parts = [
            _make_text_part("Using tool..."),
            _make_function_call_part("search", {"query": "python"}),
        ]
        sdk_resp = _make_sdk_response(parts)

        result = self.provider._parse_response(sdk_resp)
        assert result.content == "Using tool..."
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1

    def test_parse_usage_tokens(self):
        parts = [_make_text_part("ok")]
        usage = _make_usage(prompt_tokens=100, completion_tokens=50)
        sdk_resp = _make_sdk_response(parts, usage=usage)

        result = self.provider._parse_response(sdk_resp)
        assert result.usage.prompt_tokens == 100
        assert result.usage.completion_tokens == 50
        assert result.usage.total_tokens == 150

    def test_stop_reason_stop_maps_to_stop(self):
        parts = [_make_text_part("done")]
        sdk_resp = _make_sdk_response(parts, finish_reason="STOP")

        result = self.provider._parse_response(sdk_resp)
        assert result.stop_reason == StopReason.STOP

    def test_stop_reason_max_tokens_maps_to_max_tokens(self):
        parts = [_make_text_part("truncated")]
        sdk_resp = _make_sdk_response(parts, finish_reason="MAX_TOKENS")

        result = self.provider._parse_response(sdk_resp)
        assert result.stop_reason == StopReason.MAX_TOKENS

    def test_stop_reason_tool_use(self):
        parts = [_make_function_call_part("tool", {})]
        sdk_resp = _make_sdk_response(parts, finish_reason="STOP")

        result = self.provider._parse_response(sdk_resp)
        # Function calls always set TOOL_USE regardless of finish_reason
        assert result.stop_reason == StopReason.TOOL_USE

    def test_tool_call_id_auto_generated(self):
        """Gemini doesn't return a tool call ID; provider must generate one."""
        parts = [_make_function_call_part("get_weather", {"location": "NYC"})]
        sdk_resp = _make_sdk_response(parts)

        result = self.provider._parse_response(sdk_resp)
        tc = result.tool_calls[0]
        assert tc.id is not None
        assert len(tc.id) > 0


# ---------------------------------------------------------------------------
# Test: Stream chunk parsing
# ---------------------------------------------------------------------------


class TestStreamChunkParsing:
    def setup_method(self):
        self.provider = _make_provider()

    def _make_stream_chunk(self, parts=None, finish_reason=None):
        chunk = MagicMock()
        if parts is not None:
            candidate = _make_candidate(parts, finish_reason or "STOP")
            chunk.candidates = [candidate]
        else:
            chunk.candidates = []
        # text shortcut
        if parts and any(getattr(p, "text", None) for p in parts):
            chunk.text = " ".join(p.text for p in parts if getattr(p, "text", None))
        else:
            chunk.text = None
        return chunk

    def test_text_chunk(self):
        parts = [_make_text_part("Hello")]
        raw_chunk = self._make_stream_chunk(parts)

        chunks = self.provider._parse_stream_chunks(raw_chunk, 0)
        assert chunks[0].content == "Hello"
        assert chunks[0].tool_call_delta is None

    def test_function_call_chunk(self):
        parts = [_make_function_call_part("get_weather", {"location": "NYC"})]
        raw_chunk = self._make_stream_chunk(parts)

        chunks = self.provider._parse_stream_chunks(raw_chunk, 0)
        delta = chunks[0].tool_call_delta
        assert delta is not None
        assert delta.name == "get_weather"
        # Gemini provides no call ids; the parser generates one.
        assert delta.id

    def test_function_call_chunk_arguments_are_json(self):
        parts = [
            _make_function_call_part("get_weather", {"location": "NYC", "days": 3}),
        ]
        raw_chunk = self._make_stream_chunk(parts)

        chunks = self.provider._parse_stream_chunks(raw_chunk, 0)
        delta = chunks[0].tool_call_delta
        # The agent loop json.loads the accumulated fragments --
        # str(dict) (single quotes) would break it.
        assert delta is not None
        assert json.loads(delta.arguments_fragment) == {
            "location": "NYC",
            "days": 3,
        }

    def test_empty_candidates_returns_empty(self):
        raw_chunk = self._make_stream_chunk(parts=None)
        assert self.provider._parse_stream_chunks(raw_chunk, 0) == []

    def test_finish_reason_no_longer_dropped_by_text(self):
        parts = [_make_text_part("end")]
        raw_chunk = self._make_stream_chunk(parts=parts, finish_reason="STOP")
        raw_chunk.candidates[0].finish_reason = "STOP"

        chunks = self.provider._parse_stream_chunks(raw_chunk, 0)
        # Text and the finish_reason of the same raw chunk both surface.
        assert chunks[0].content == "end"
        assert chunks[-1].stop_reason == StopReason.STOP

    def test_stream_tool_calls_round_trip_through_loop(self):
        from anchor.agent.agent import Agent, _RoundState

        parts = [
            _make_function_call_part("alpha", {"x": 1}),
            _make_function_call_part("beta", {"y": 2}),
        ]
        raw_chunk = self._make_stream_chunk(parts)
        raw_chunk.candidates[0].finish_reason = "STOP"

        chunks = self.provider._parse_stream_chunks(raw_chunk, 0)
        seen = sum(1 for c in chunks if c.tool_call_delta is not None)
        chunks = [self.provider._with_tool_use_override(c, seen) for c in chunks]

        state = _RoundState()
        for chunk in chunks:
            Agent._ingest_chunk(state, chunk)
        # Function calls mean TOOL_USE even though Gemini reports STOP;
        # without the override the loop silently never executes tools.
        assert state.stop_reason == StopReason.TOOL_USE
        calls = Agent._build_tool_calls(state.accumulators)
        assert [c.name for c in calls] == ["alpha", "beta"]
        assert [c.arguments for c in calls] == [{"x": 1}, {"y": 2}]
        assert calls[0].id
        assert calls[1].id
        assert calls[0].id != calls[1].id


# ---------------------------------------------------------------------------
# Test: Error mapping
# ---------------------------------------------------------------------------


class TestErrorMapping:
    def setup_method(self):
        self.provider = _make_provider()

    @patch("anchor.llm.providers.gemini.genai")
    def test_authentication_error_mapped(self, mock_genai):
        mock_genai.errors = MagicMock()
        AuthErr = type("ClientError", (Exception,), {})
        err = AuthErr("API key not valid")
        err.status_code = 401

        result = self.provider._map_error(err)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "gemini"

    @patch("anchor.llm.providers.gemini.genai")
    def test_rate_limit_error_mapped(self, mock_genai):
        RateLimitErr = type("ClientError", (Exception,), {})
        err = RateLimitErr("rate limited")
        err.status_code = 429

        result = self.provider._map_error(err)
        assert isinstance(result, RateLimitError)
        assert result.is_transient is True

    @patch("anchor.llm.providers.gemini.genai")
    def test_server_error_5xx_mapped(self, mock_genai):
        ServerErr = type("ServerError", (Exception,), {})
        err = ServerErr("server error")
        err.status_code = 500

        result = self.provider._map_error(err)
        assert isinstance(result, ServerError)
        assert result.is_transient is True

    @patch("anchor.llm.providers.gemini.genai")
    def test_not_found_mapped(self, mock_genai):
        NotFoundErr = type("NotFoundError", (Exception,), {})
        err = NotFoundErr("model not found")
        err.status_code = 404

        result = self.provider._map_error(err)
        assert isinstance(result, ModelNotFoundError)

    @patch("anchor.llm.providers.gemini.genai")
    def test_timeout_error_mapped(self, mock_genai):
        TimeoutErr = type("TimeoutError", (Exception,), {})
        err = TimeoutErr("timed out")

        result = self.provider._map_error(err)
        assert isinstance(result, TimeoutError)
        assert result.is_transient is True

    @patch("anchor.llm.providers.gemini.genai")
    def test_connection_error_mapped_to_timeout(self, mock_genai):
        ConnErr = type("APIConnectionError", (Exception,), {})
        err = ConnErr("connection failed")

        result = self.provider._map_error(err)
        assert isinstance(result, TimeoutError)

    @patch("anchor.llm.providers.gemini.genai")
    def test_unknown_error_maps_to_provider_error(self, mock_genai):
        from anchor.llm.errors import ProviderError
        err = ValueError("unexpected error")

        result = self.provider._map_error(err)
        assert isinstance(result, ProviderError)
        assert result.provider == "gemini"


# ---------------------------------------------------------------------------
# Test: _do_invoke integration
# ---------------------------------------------------------------------------


class TestDoInvoke:
    def setup_method(self):
        self.provider = _make_provider()

    def _build_sdk_response(self, text="Answer"):
        parts = [_make_text_part(text)]
        return _make_sdk_response(parts)

    @patch("anchor.llm.providers.gemini.genai")
    def test_do_invoke_returns_llm_response(self, mock_genai):
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.return_value = self._build_sdk_response("Hello!")

        messages = [Message(role=Role.USER, content="Hi")]
        result = self.provider._do_invoke(messages, tools=None)

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.provider == "gemini"
        mock_client.models.generate_content.assert_called_once()

    @patch("anchor.llm.providers.gemini.genai")
    def test_do_invoke_passes_system_instruction(self, mock_genai):
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.return_value = self._build_sdk_response()

        # Make GenerateContentConfig raise so we fall back to our simple namespace
        mock_genai.types.GenerateContentConfig.side_effect = TypeError("unavailable")

        messages = [
            Message(role=Role.SYSTEM, content="Be concise."),
            Message(role=Role.USER, content="Hi"),
        ]
        self.provider._do_invoke(messages, tools=None)

        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs.get("config")
        assert config is not None
        # system_instruction should be in the config (set on our fallback namespace obj)
        assert config.system_instruction == "Be concise."

    @patch("anchor.llm.providers.gemini.genai")
    def test_do_invoke_maps_sdk_error(self, mock_genai):
        AuthErr = type("AuthenticationError", (Exception,), {})
        mock_genai.Client.return_value = MagicMock()
        mock_genai.Client.return_value.models.generate_content.side_effect = AuthErr("bad key")

        messages = [Message(role=Role.USER, content="Hi")]
        with pytest.raises(Exception):
            self.provider._do_invoke(messages, tools=None)

    @patch("anchor.llm.providers.gemini.genai")
    def test_do_invoke_passes_tools(self, mock_genai):
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content.return_value = self._build_sdk_response()

        tools = [_make_tool_schema()]
        messages = [Message(role=Role.USER, content="What's the weather?")]
        self.provider._do_invoke(messages, tools=tools)

        call_kwargs = mock_client.models.generate_content.call_args[1]
        # Tools should be passed in the config or as a separate argument
        assert call_kwargs.get("tools") is not None or (
            call_kwargs.get("config") is not None
            and getattr(call_kwargs["config"], "tools", None) is not None
        )


# ---------------------------------------------------------------------------
# Test: _do_stream integration
# ---------------------------------------------------------------------------


class TestDoStream:
    def setup_method(self):
        self.provider = _make_provider()

    @patch("anchor.llm.providers.gemini.genai")
    def test_do_stream_yields_text_chunks(self, mock_genai):
        parts = [_make_text_part("Hello")]
        chunk1 = MagicMock()
        chunk1.candidates = [_make_candidate(parts, "STOP")]
        chunk1.text = "Hello"

        parts2 = [_make_text_part(" world")]
        chunk2 = MagicMock()
        chunk2.candidates = [_make_candidate(parts2, "STOP")]
        chunk2.text = " world"

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content_stream.return_value = iter([chunk1, chunk2])

        messages = [Message(role=Role.USER, content="Hi")]
        chunks = list(self.provider._do_stream(messages, tools=None))

        text_chunks = [c for c in chunks if c.content is not None]
        assert len(text_chunks) >= 1

    @patch("anchor.llm.providers.gemini.genai")
    def test_do_stream_maps_errors(self, mock_genai):
        from anchor.llm.errors import ProviderError
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.generate_content_stream.side_effect = Exception("api error")

        messages = [Message(role=Role.USER, content="Hi")]
        with pytest.raises(ProviderError):
            list(self.provider._do_stream(messages, tools=None))


# ---------------------------------------------------------------------------
# Test: _do_ainvoke integration
# ---------------------------------------------------------------------------


class TestDoAinvoke:
    def setup_method(self):
        self.provider = _make_provider()

    @patch("anchor.llm.providers.gemini.genai")
    @pytest.mark.asyncio
    async def test_do_ainvoke_returns_llm_response(self, mock_genai):
        parts = [_make_text_part("Async answer")]
        sdk_resp = _make_sdk_response(parts)

        mock_async_client = MagicMock()
        mock_genai.Client.return_value = mock_async_client
        mock_async_client.aio.models.generate_content = AsyncMock(return_value=sdk_resp)

        messages = [Message(role=Role.USER, content="Hi async")]
        result = await self.provider._do_ainvoke(messages, tools=None)

        assert isinstance(result, LLMResponse)
        assert result.content == "Async answer"
        assert result.provider == "gemini"


# ---------------------------------------------------------------------------
# Test: Self-registration
# ---------------------------------------------------------------------------


class TestSelfRegistration:
    def test_import_registers_provider(self):
        """Importing the module should register 'gemini' in the registry."""
        import anchor.llm.providers.gemini  # noqa: F401
        from anchor.llm.registry import _PROVIDERS
        assert "gemini" in _PROVIDERS
