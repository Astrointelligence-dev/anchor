"""Phase 5 provider tests: tool_choice, prompt caching, context management."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from anchor.llm.models import Message, Role, ToolSchema
from anchor.llm.providers._openai_compat import (
    build_call_kwargs,
)
from anchor.llm.providers._openai_compat import (
    convert_tool_choice as openai_tool_choice,
)
from anchor.llm.providers.anthropic import (
    AnthropicProvider,
    _context_betas,
    _StreamState,
)
from anchor.llm.providers.anthropic import (
    _convert_tool_choice as anthropic_tool_choice,
)
from anchor.llm.providers.gemini import (
    _convert_tool_choice as gemini_tool_choice,
)


def _make_provider(**kwargs) -> AnthropicProvider:
    return AnthropicProvider(
        model="claude-sonnet-5", api_key="key", max_retries=0, **kwargs,
    )


def _tool() -> ToolSchema:
    return ToolSchema(
        name="echo",
        description="echoes",
        input_schema={"type": "object", "properties": {}},
    )


def _messages() -> list[Message]:
    return [Message(role=Role.USER, content="hi")]


def _sdk_response(text: str = "hi") -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    usage = MagicMock()
    usage.input_tokens = 10
    usage.output_tokens = 5
    usage.cache_creation_input_tokens = 3
    usage.cache_read_input_tokens = 7
    response = MagicMock()
    response.content = [block]
    response.usage = usage
    response.model = "claude-sonnet-5"
    response.stop_reason = "end_turn"
    return response


# ---------------------------------------------------------------------------
# tool_choice mapping (all provider families)
# ---------------------------------------------------------------------------


class TestToolChoiceMapping:
    def test_anthropic_modes_and_passthrough(self) -> None:
        assert anthropic_tool_choice("auto") == {"type": "auto"}
        assert anthropic_tool_choice("any") == {"type": "any"}
        assert anthropic_tool_choice("none") == {"type": "none"}
        specific = {"type": "tool", "name": "echo"}
        assert anthropic_tool_choice(specific) == specific

    def test_anthropic_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid tool_choice"):
            anthropic_tool_choice("required")

    def test_openai_modes_and_function(self) -> None:
        assert openai_tool_choice("auto") == "auto"
        assert openai_tool_choice("any") == "required"
        assert openai_tool_choice("none") == "none"
        assert openai_tool_choice({"type": "tool", "name": "echo"}) == {
            "type": "function",
            "function": {"name": "echo"},
        }

    def test_openai_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid tool_choice"):
            openai_tool_choice("required")

    def test_gemini_modes_and_allowed_names(self) -> None:
        assert gemini_tool_choice("auto") == {
            "function_calling_config": {"mode": "AUTO"},
        }
        assert gemini_tool_choice("none") == {
            "function_calling_config": {"mode": "NONE"},
        }
        assert gemini_tool_choice({"type": "tool", "name": "echo"}) == {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["echo"],
            },
        }

    def test_openai_family_build_call_kwargs(self) -> None:
        call_kwargs = build_call_kwargs(
            "gpt-x", [], [_tool()], tool_choice="any",
        )
        assert call_kwargs["tool_choice"] == "required"

    def test_tool_choice_omitted_without_tools(self) -> None:
        call_kwargs = build_call_kwargs("gpt-x", [], None, tool_choice="any")
        assert "tool_choice" not in call_kwargs


# ---------------------------------------------------------------------------
# Anthropic: call kwargs, caching, beta routing
# ---------------------------------------------------------------------------


class TestAnthropicPhase5:
    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_tool_choice_in_call_kwargs_non_beta(self, mock_anthropic) -> None:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_response()

        provider = _make_provider()
        provider._do_invoke(_messages(), tools=[_tool()], tool_choice="any")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tool_choice"] == {"type": "any"}
        mock_client.beta.messages.create.assert_not_called()

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_prompt_caching_flag_adds_cache_control(self, mock_anthropic) -> None:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _sdk_response()

        provider = _make_provider(prompt_caching=True)
        response = provider._do_invoke(_messages(), tools=None)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["cache_control"] == {"type": "ephemeral"}
        # Cache hit visibility parsed from usage
        assert response.usage.cache_creation_tokens == 3
        assert response.usage.cache_read_tokens == 7

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_context_management_routes_to_beta(self, mock_anthropic) -> None:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.beta.messages.create.return_value = _sdk_response()

        provider = _make_provider()
        config = {"edits": [{"type": "clear_tool_uses_20250919"}]}
        provider._do_invoke(_messages(), tools=None, context_management=config)

        mock_client.messages.create.assert_not_called()
        call_kwargs = mock_client.beta.messages.create.call_args[1]
        assert call_kwargs["context_management"] == config
        assert call_kwargs["betas"] == ["context-management-2025-06-27"]

    def test_context_betas_selection(self) -> None:
        clear = {"edits": [{"type": "clear_tool_uses_20250919"}]}
        compact = {"edits": [{"type": "compact_20260112"}]}
        both = {"edits": [
            {"type": "clear_thinking_20251015"},
            {"type": "compact_20260112"},
        ]}
        assert _context_betas(clear) == ["context-management-2025-06-27"]
        assert _context_betas(compact) == ["compact-2026-01-12"]
        assert _context_betas(both) == [
            "context-management-2025-06-27",
            "compact-2026-01-12",
        ]
        assert _context_betas({"edits": []}) == ["context-management-2025-06-27"]

    @patch("anchor.llm.providers.anthropic.anthropic")
    def test_compaction_block_parsed_into_raw_content(self, mock_anthropic) -> None:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        compaction = MagicMock()
        compaction.type = "compaction"
        compaction.content = "summary of earlier turns"
        response = _sdk_response("after compaction")
        response.content = [compaction, *response.content]
        mock_client.beta.messages.create.return_value = response

        provider = _make_provider()
        result = provider._do_invoke(
            _messages(),
            tools=None,
            context_management={"edits": [{"type": "compact_20260112"}]},
        )

        assert result.raw_content == [
            {"type": "compaction", "content": "summary of earlier turns"},
        ]
        assert result.content == "after compaction"

    def test_raw_content_reemitted_verbatim(self) -> None:
        provider = _make_provider()
        block = {"type": "compaction", "content": "summary"}
        messages = [
            Message(role=Role.USER, content="hi"),
            Message(role=Role.ASSISTANT, content="ok", raw_content=[block]),
        ]
        _, converted = provider._extract_system_and_convert(messages)
        assistant = converted[-1]
        assert assistant["role"] == "assistant"
        assert assistant["content"][0] == block
        assert assistant["content"][1] == {"type": "text", "text": "ok"}

    def test_stream_compaction_block_assembled(self) -> None:
        provider = _make_provider()
        state = _StreamState()

        start = MagicMock()
        start.type = "content_block_start"
        start.index = 0
        start.content_block = MagicMock()
        start.content_block.type = "compaction"
        start.content_block.content = ""

        delta = MagicMock()
        delta.type = "content_block_delta"
        delta.index = 0
        delta.delta = MagicMock()
        delta.delta.type = "compaction_delta"
        delta.delta.content = "sum"

        stop = MagicMock()
        stop.type = "content_block_stop"

        assert provider._parse_stream_event(start, state=state) is None
        assert provider._parse_stream_event(delta, state=state) is None
        chunk = provider._parse_stream_event(stop, state=state)
        assert chunk is not None
        assert chunk.raw_block == {"type": "compaction", "content": "sum"}
        # State cleared — a later block_stop yields nothing.
        assert provider._parse_stream_event(stop, state=state) is None
