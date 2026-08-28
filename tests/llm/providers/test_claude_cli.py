"""Tests for ClaudeCLIProvider — Claude Code CLI as an LLM backend.

The claude_agent_sdk is faked in sys.modules so these run without the SDK
(and without the CLI). Tests cover:
- provider_name / model_id / no API key
- message flattening (system split, single-turn bare, tool turns dropped)
- image content blocks rejected instead of silently dropped
- option building: token-overhead guards, MCP tool bridge, tool_choice
- the defer hook: unknown call defers, answered call is allowed through
- delivery handler returns the caller's stored tool result
- result parsing: text, deferred tool call, usage/cost, stop reasons
- error mapping (result payload and SDK exceptions)
- invoke/stream over the sync bridge
- registry registration
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any

import pytest

from anchor.llm.errors import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    ServerError,
)
from anchor.llm.errors import TimeoutError as ProviderTimeoutError
from anchor.llm.models import (
    ContentBlock,
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolResult,
    ToolSchema,
)

# ---------------------------------------------------------------------------
# Fake SDK
# ---------------------------------------------------------------------------

class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class AssistantMessage:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class StreamEvent:
    def __init__(self, event: dict[str, Any]) -> None:
        self.event = event


class DeferredToolUse:
    def __init__(self, id: str, name: str, input: dict[str, Any]) -> None:  # noqa: A002
        self.id = id
        self.name = name
        self.input = input


class ResultMessage:
    def __init__(
        self,
        *,
        result: str = "",
        usage: dict[str, Any] | None = None,
        total_cost_usd: float | None = None,
        stop_reason: str | None = "end_turn",
        deferred_tool_use: Any = None,
        is_error: bool = False,
        api_error_status: int | None = None,
        subtype: str = "success",
    ) -> None:
        self.result = result
        self.usage = usage or {}
        self.total_cost_usd = total_cost_usd
        self.stop_reason = stop_reason
        self.deferred_tool_use = deferred_tool_use
        self.is_error = is_error
        self.api_error_status = api_error_status
        self.subtype = subtype


class ClaudeSDKError(Exception):
    pass


class CLIConnectionError(ClaudeSDKError):
    pass


class CLINotFoundError(CLIConnectionError):
    pass


class ProcessError(ClaudeSDKError):
    pass


class ResultError(ProcessError):
    def __init__(self, message: str, api_error_status: int | None = None) -> None:
        super().__init__(message)
        self.api_error_status = api_error_status


class ClaudeAgentOptions:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class HookMatcher:
    def __init__(self, matcher: str, hooks: list[Any]) -> None:
        self.matcher = matcher
        self.hooks = hooks


class SdkMcpTool:
    def __init__(self, name: str, description: str, schema: Any, handler: Any) -> None:
        self.name = name
        self.description = description
        self.schema = schema
        self.handler = handler


def _tool(name: str, description: str, input_schema: Any) -> Any:
    def decorator(handler: Any) -> SdkMcpTool:
        return SdkMcpTool(name, description, input_schema, handler)

    return decorator


def _create_sdk_mcp_server(*, name: str, version: str, tools: list[Any]) -> dict[str, Any]:
    return {"name": name, "version": version, "tools": tools}


@pytest.fixture
def fake_sdk(monkeypatch):
    """Install a fake claude_agent_sdk and let each test script its messages."""
    module = types.ModuleType("claude_agent_sdk")
    module.ClaudeAgentOptions = ClaudeAgentOptions
    module.HookMatcher = HookMatcher
    module.tool = _tool
    module.create_sdk_mcp_server = _create_sdk_mcp_server
    module.CLINotFoundError = CLINotFoundError
    module.CLIConnectionError = CLIConnectionError
    module.ProcessError = ProcessError
    module.ResultError = ResultError
    module.scripted = []
    module.raises = None
    module.calls = []

    async def query(*, prompt: str, options: Any):
        module.calls.append({"prompt": prompt, "options": options})
        if module.raises is not None:
            raise module.raises
        for message in module.scripted:
            yield message

    module.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_provider():
    from anchor.llm.providers.claude_cli import ClaudeCLIProvider

    return ClaudeCLIProvider


def _make_provider(**kwargs):
    cls = _import_provider()
    defaults = {"model": "sonnet", "max_retries": 0}
    defaults.update(kwargs)
    return cls(**defaults)


def _text_delta(text: str) -> dict[str, Any]:
    return {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}


def _weather_schema() -> ToolSchema:
    return ToolSchema(
        name="get_weather",
        description="Get current weather",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )


def _options(provider, messages, tools=None, *, partial=False, tool_choice=None):
    import claude_agent_sdk as sdk

    return provider._build_options(
        sdk, messages, tools, partial=partial, tool_choice=tool_choice
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_provider_name(self):
        assert _make_provider().provider_name == "claude_cli"

    def test_model_id(self):
        assert _make_provider(model="opus").model_id == "claude_cli/opus"

    def test_no_api_key_needed(self):
        provider = _make_provider()
        assert provider._resolve_api_key() is None
        assert provider._api_key is None

    def test_registered(self):
        from anchor.llm.registry import _PROVIDERS

        _import_provider()
        assert _PROVIDERS["claude_cli"] is _import_provider()


# ---------------------------------------------------------------------------
# Message flattening
# ---------------------------------------------------------------------------

class TestFlatten:
    def test_system_split_out_and_single_turn_is_bare(self):
        from anchor.llm.providers.claude_cli import _flatten

        system, prompt = _flatten(
            [
                Message(role=Role.SYSTEM, content="Be terse."),
                Message(role=Role.USER, content="Say: PONG"),
            ]
        )
        assert system == "Be terse."
        assert prompt == "Say: PONG"

    def test_multiple_system_messages_join(self):
        from anchor.llm.providers.claude_cli import _flatten

        system, _ = _flatten(
            [
                Message(role=Role.SYSTEM, content="A"),
                Message(role=Role.SYSTEM, content="B"),
                Message(role=Role.USER, content="hi"),
            ]
        )
        assert system == "A\n\nB"

    def test_multi_turn_is_role_labelled(self):
        from anchor.llm.providers.claude_cli import _flatten

        _, prompt = _flatten(
            [
                Message(role=Role.USER, content="one"),
                Message(role=Role.ASSISTANT, content="two"),
                Message(role=Role.USER, content="three"),
            ]
        )
        assert prompt == "User: one\n\nAssistant: two\n\nUser: three"

    def test_tool_turns_are_not_in_the_prompt(self):
        from anchor.llm.providers.claude_cli import _flatten

        _, prompt = _flatten(
            [
                Message(role=Role.USER, content="weather?"),
                Message(
                    role=Role.TOOL,
                    tool_result=ToolResult(tool_call_id="t1", content="25C"),
                ),
            ]
        )
        assert prompt == "weather?"
        assert "25C" not in prompt

    def test_no_system_returns_none(self):
        from anchor.llm.providers.claude_cli import _flatten

        system, _ = _flatten([Message(role=Role.USER, content="hi")])
        assert system is None

    def test_text_content_blocks_are_joined(self):
        from anchor.llm.providers.claude_cli import _flatten

        _, prompt = _flatten(
            [
                Message(
                    role=Role.USER,
                    content=[
                        ContentBlock(type="text", text="a"),
                        ContentBlock(type="text", text="b"),
                    ],
                )
            ]
        )
        assert prompt == "a\nb"

    def test_image_block_raises_instead_of_dropping_content(self):
        from anchor.llm.providers.claude_cli import _flatten

        with pytest.raises(ProviderError, match="text only"):
            _flatten(
                [
                    Message(
                        role=Role.USER,
                        content=[ContentBlock(type="image_url", image_url="http://x/y.png")],
                    )
                ]
            )


# ---------------------------------------------------------------------------
# Delivery maps
# ---------------------------------------------------------------------------

class TestDeliveryMaps:
    def test_pairs_call_with_result_by_id(self):
        from anchor.llm.providers.claude_cli import _canonical, _delivery_maps

        call = ToolCall(id="t1", name="get_weather", arguments={"city": "Tokyo"})
        by_key, by_name = _delivery_maps(
            [
                Message(role=Role.ASSISTANT, tool_calls=[call]),
                Message(
                    role=Role.TOOL,
                    tool_result=ToolResult(tool_call_id="t1", content="25C"),
                ),
            ]
        )
        assert by_key[("get_weather", _canonical({"city": "Tokyo"}))] == "25C"
        assert by_name["get_weather"] == "25C"

    def test_result_without_matching_call_is_ignored(self):
        from anchor.llm.providers.claude_cli import _delivery_maps

        by_key, by_name = _delivery_maps(
            [
                Message(
                    role=Role.TOOL,
                    tool_result=ToolResult(tool_call_id="orphan", content="x"),
                )
            ]
        )
        assert by_key == {}
        assert by_name == {}

    def test_canonical_is_key_order_independent(self):
        from anchor.llm.providers.claude_cli import _canonical

        assert _canonical({"a": 1, "b": 2}) == _canonical({"b": 2, "a": 1})


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class TestOptions:
    def test_token_overhead_guards_are_on(self, fake_sdk):
        provider = _make_provider()
        options, _ = _options(provider, [Message(role=Role.USER, content="hi")])
        # Without these the CLI loads its 177k-token default prompt plus the
        # user's CLAUDE.md, settings and MCP servers.
        assert options.tools == []
        assert options.setting_sources == []
        assert options.strict_mcp_config is True

    def test_no_mcp_server_without_tools(self, fake_sdk):
        provider = _make_provider()
        options, _ = _options(provider, [Message(role=Role.USER, content="hi")])
        assert not hasattr(options, "mcp_servers")
        assert not hasattr(options, "hooks")

    def test_tools_become_an_mcp_server_and_allowed_list(self, fake_sdk):
        provider = _make_provider()
        options, _ = _options(
            provider, [Message(role=Role.USER, content="hi")], [_weather_schema()]
        )
        assert list(options.mcp_servers) == ["anchor"]
        assert options.allowed_tools == ["mcp__anchor__get_weather"]
        assert options.hooks["PreToolUse"][0].matcher == "mcp__anchor__.*"

    def test_tool_choice_none_drops_the_tools(self, fake_sdk):
        provider = _make_provider()
        options, _ = _options(
            provider,
            [Message(role=Role.USER, content="hi")],
            [_weather_schema()],
            tool_choice="none",
        )
        assert not hasattr(options, "mcp_servers")

    def test_tool_choice_any_becomes_a_system_instruction(self, fake_sdk):
        provider = _make_provider()
        options, _ = _options(
            provider,
            [Message(role=Role.USER, content="hi")],
            [_weather_schema()],
            tool_choice="any",
        )
        assert "must call one of the available tools" in options.system_prompt

    def test_named_tool_choice_names_the_tool(self, fake_sdk):
        provider = _make_provider()
        options, _ = _options(
            provider,
            [Message(role=Role.SYSTEM, content="Be terse."),
             Message(role=Role.USER, content="hi")],
            [_weather_schema()],
            tool_choice={"type": "tool", "name": "get_weather"},
        )
        assert options.system_prompt.startswith("Be terse.")
        assert "must call the get_weather tool" in options.system_prompt

    def test_builtin_tool_list_is_passed_and_allowed(self, fake_sdk):
        provider = _make_provider(builtin_tools=["Read", "Grep"])
        options, _ = _options(provider, [Message(role=Role.USER, content="hi")])
        assert options.tools == ["Read", "Grep"]
        assert options.allowed_tools == ["Read", "Grep"]

    def test_builtin_preset(self, fake_sdk):
        provider = _make_provider(builtin_tools="claude_code")
        options, _ = _options(provider, [Message(role=Role.USER, content="hi")])
        assert options.tools == {"type": "preset", "preset": "claude_code"}

    def test_optional_knobs_only_set_when_given(self, fake_sdk):
        provider = _make_provider()
        options, _ = _options(provider, [Message(role=Role.USER, content="hi")])
        for attr in ("permission_mode", "cwd", "cli_path", "max_turns"):
            assert not hasattr(options, attr)

    def test_optional_knobs_are_forwarded(self, fake_sdk):
        provider = _make_provider(
            permission_mode="acceptEdits", cwd="/work/x", cli_path="/bin/claude",
            max_turns=3,
        )
        options, _ = _options(provider, [Message(role=Role.USER, content="hi")])
        assert options.permission_mode == "acceptEdits"
        assert options.cwd == "/work/x"
        assert options.cli_path == "/bin/claude"
        assert options.max_turns == 3


# ---------------------------------------------------------------------------
# The tool bridge
# ---------------------------------------------------------------------------

class TestToolBridge:
    def test_unanswered_call_is_deferred_back_to_anchor(self, fake_sdk):
        provider = _make_provider()
        options, _ = _options(
            provider, [Message(role=Role.USER, content="hi")], [_weather_schema()]
        )
        hook = options.hooks["PreToolUse"][0].hooks[0]
        out = asyncio.run(
            hook(
                {"tool_name": "mcp__anchor__get_weather", "tool_input": {"city": "Tokyo"}},
                "t1",
                None,
            )
        )
        assert out["hookSpecificOutput"]["permissionDecision"] == "defer"

    def test_answered_call_is_allowed_through(self, fake_sdk):
        provider = _make_provider()
        call = ToolCall(id="t1", name="get_weather", arguments={"city": "Tokyo"})
        messages = [
            Message(role=Role.USER, content="weather?"),
            Message(role=Role.ASSISTANT, tool_calls=[call]),
            Message(
                role=Role.TOOL, tool_result=ToolResult(tool_call_id="t1", content="25C")
            ),
        ]
        options, _ = _options(provider, messages, [_weather_schema()])
        hook = options.hooks["PreToolUse"][0].hooks[0]
        out = asyncio.run(
            hook(
                {"tool_name": "mcp__anchor__get_weather", "tool_input": {"city": "Tokyo"}},
                "t2",
                None,
            )
        )
        assert out == {}

    def test_handler_returns_the_stored_result(self, fake_sdk):
        provider = _make_provider()
        call = ToolCall(id="t1", name="get_weather", arguments={"city": "Tokyo"})
        messages = [
            Message(role=Role.ASSISTANT, tool_calls=[call]),
            Message(
                role=Role.TOOL, tool_result=ToolResult(tool_call_id="t1", content="25C")
            ),
        ]
        options, _ = _options(provider, messages, [_weather_schema()])
        handler = options.mcp_servers["anchor"]["tools"][0].handler
        out = asyncio.run(handler({"city": "Tokyo"}))
        assert out == {"content": [{"type": "text", "text": "25C"}]}

    def test_handler_falls_back_to_name_when_args_drift(self, fake_sdk):
        provider = _make_provider()
        call = ToolCall(id="t1", name="get_weather", arguments={"city": "Tokyo"})
        messages = [
            Message(role=Role.ASSISTANT, tool_calls=[call]),
            Message(
                role=Role.TOOL, tool_result=ToolResult(tool_call_id="t1", content="25C")
            ),
        ]
        options, _ = _options(provider, messages, [_weather_schema()])
        handler = options.mcp_servers["anchor"]["tools"][0].handler
        out = asyncio.run(handler({"city": "tokyo"}))
        assert out["content"][0]["text"] == "25C"

    def test_each_tool_keeps_its_own_handler(self, fake_sdk):
        provider = _make_provider()
        other = ToolSchema(name="other", description="d", input_schema={})
        call = ToolCall(id="t1", name="other", arguments={})
        messages = [
            Message(role=Role.ASSISTANT, tool_calls=[call]),
            Message(
                role=Role.TOOL, tool_result=ToolResult(tool_call_id="t1", content="OTHER")
            ),
        ]
        options, _ = _options(provider, messages, [_weather_schema(), other])
        handlers = {t.name: t.handler for t in options.mcp_servers["anchor"]["tools"]}
        assert asyncio.run(handlers["other"]({}))["content"][0]["text"] == "OTHER"
        assert asyncio.run(handlers["get_weather"]({}))["content"][0]["text"] == ""


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

class TestUsageAndStopReason:
    def test_usage_maps_tokens_cache_and_cost(self):
        from anchor.llm.providers.claude_cli import _usage

        usage = _usage(
            ResultMessage(
                usage={
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cache_creation_input_tokens": 7,
                    "cache_read_input_tokens": 3,
                },
                total_cost_usd=0.5,
            )
        )
        assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (10, 4, 14)
        assert usage.cache_creation_tokens == 7
        assert usage.cache_read_tokens == 3
        assert usage.total_cost == 0.5

    def test_usage_tolerates_missing_fields(self):
        from anchor.llm.providers.claude_cli import _usage

        usage = _usage(ResultMessage())
        assert usage.total_tokens == 0
        assert usage.total_cost is None

    def test_deferred_means_tool_use(self):
        from anchor.llm.providers.claude_cli import _stop_reason

        result = ResultMessage(deferred_tool_use=DeferredToolUse("t", "x", {}))
        assert _stop_reason(result) is StopReason.TOOL_USE

    def test_tool_deferred_stop_reason(self):
        from anchor.llm.providers.claude_cli import _stop_reason

        assert _stop_reason(ResultMessage(stop_reason="tool_deferred")) is StopReason.TOOL_USE

    def test_max_turns_is_max_tokens(self):
        from anchor.llm.providers.claude_cli import _stop_reason

        assert (
            _stop_reason(ResultMessage(subtype="error_max_turns")) is StopReason.MAX_TOKENS
        )

    def test_default_is_stop(self):
        from anchor.llm.providers.claude_cli import _stop_reason

        assert _stop_reason(ResultMessage()) is StopReason.STOP


class TestBuildResponse:
    def test_text_response(self, fake_sdk):
        provider = _make_provider()
        response = provider._build_response(
            ResultMessage(result="hello", usage={"input_tokens": 1, "output_tokens": 2}),
            [],
        )
        assert response.content == "hello"
        assert response.tool_calls is None
        assert response.stop_reason is StopReason.STOP
        assert response.provider == "claude_cli"
        assert response.model == "sonnet"

    def test_cli_result_text_wins_over_intermediate_narration(self, fake_sdk):
        provider = _make_provider()
        response = provider._build_response(
            ResultMessage(result="the answer"), ["thinking out loud"]
        )
        assert response.content == "the answer"

    def test_assistant_text_is_the_fallback(self, fake_sdk):
        provider = _make_provider()
        response = provider._build_response(ResultMessage(result=""), ["only text"])
        assert response.content == "only text"

    def test_deferred_call_becomes_a_tool_call(self, fake_sdk):
        provider = _make_provider()
        response = provider._build_response(
            ResultMessage(
                deferred_tool_use=DeferredToolUse(
                    "toolu_1", "mcp__anchor__get_weather", {"city": "Tokyo"}
                )
            ),
            [],
        )
        assert response.stop_reason is StopReason.TOOL_USE
        assert response.tool_calls == [
            ToolCall(id="toolu_1", name="get_weather", arguments={"city": "Tokyo"})
        ]
        assert response.content is None

    def test_missing_result_message_is_transient(self, fake_sdk):
        provider = _make_provider()
        with pytest.raises(ProviderError) as exc:
            provider._build_response(None, [])
        assert exc.value.is_transient is True

    def test_error_result_raises(self, fake_sdk):
        provider = _make_provider()
        with pytest.raises(ModelNotFoundError):
            provider._build_response(
                ResultMessage(is_error=True, api_error_status=404, result="nope"), []
            )


class TestFinalChunk:
    def test_carries_usage_and_stop_reason(self, fake_sdk):
        provider = _make_provider()
        chunk = provider._final_chunk(
            ResultMessage(usage={"input_tokens": 3, "output_tokens": 1})
        )
        assert chunk.usage.prompt_tokens == 3
        assert chunk.stop_reason is StopReason.STOP
        assert chunk.tool_call_delta is None

    def test_deferred_call_becomes_a_complete_delta(self, fake_sdk):
        provider = _make_provider()
        chunk = provider._final_chunk(
            ResultMessage(
                deferred_tool_use=DeferredToolUse(
                    "toolu_1", "mcp__anchor__get_weather", {"city": "Tokyo"}
                )
            )
        )
        delta = chunk.tool_call_delta
        assert delta.id == "toolu_1"
        assert delta.name == "get_weather"
        assert json.loads(delta.arguments_fragment) == {"city": "Tokyo"}


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

class TestErrorMapping:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (401, AuthenticationError),
            (403, AuthenticationError),
            (404, ModelNotFoundError),
            (429, RateLimitError),
            (500, ServerError),
            (503, ServerError),
            (None, ProviderError),
            (400, ProviderError),
        ],
    )
    def test_status_to_error(self, fake_sdk, status, expected):
        provider = _make_provider()
        assert type(provider._status_error(status, "boom")) is expected

    def test_missing_cli_explains_how_to_install(self, fake_sdk):
        provider = _make_provider()
        mapped = provider._map_error(CLINotFoundError("Claude Code not found"))
        assert "npm install -g" in str(mapped)

    def test_result_error_uses_its_status(self, fake_sdk):
        provider = _make_provider()
        assert isinstance(
            provider._map_error(ResultError("rate limited", api_error_status=429)),
            RateLimitError,
        )

    def test_process_error_is_transient(self, fake_sdk):
        provider = _make_provider()
        assert provider._map_error(ProcessError("exit 1")).is_transient is True

    def test_provider_error_passes_through(self, fake_sdk):
        provider = _make_provider()
        original = ProviderError("keep me", provider="claude_cli")
        assert provider._map_error(original) is original


# ---------------------------------------------------------------------------
# End to end over the fake SDK (exercises the sync bridge)
# ---------------------------------------------------------------------------

class TestInvokeAndStream:
    def test_invoke(self, fake_sdk):
        fake_sdk.scripted = [
            AssistantMessage([TextBlock("PONG")]),
            ResultMessage(usage={"input_tokens": 5, "output_tokens": 1}, total_cost_usd=0.1),
        ]
        response = _make_provider().invoke([Message(role=Role.USER, content="ping")])
        assert response.content == "PONG"
        assert response.usage.total_cost == 0.1
        assert fake_sdk.calls[0]["prompt"] == "ping"
        assert fake_sdk.calls[0]["options"].include_partial_messages is False

    def test_invoke_surfaces_a_deferred_tool_call(self, fake_sdk):
        fake_sdk.scripted = [
            ResultMessage(
                deferred_tool_use=DeferredToolUse(
                    "toolu_1", "mcp__anchor__get_weather", {"city": "Tokyo"}
                ),
                stop_reason="tool_deferred",
            )
        ]
        response = _make_provider().invoke(
            [Message(role=Role.USER, content="weather?")], tools=[_weather_schema()]
        )
        assert response.tool_calls[0].name == "get_weather"

    def test_stream_yields_text_then_a_final_chunk(self, fake_sdk):
        fake_sdk.scripted = [
            StreamEvent(_text_delta("1 ")),
            StreamEvent(_text_delta("2")),
            ResultMessage(usage={"input_tokens": 9, "output_tokens": 2}),
        ]
        chunks = list(_make_provider().stream([Message(role=Role.USER, content="count")]))
        assert "".join(c.content or "" for c in chunks) == "1 2"
        assert chunks[-1].usage.prompt_tokens == 9
        assert fake_sdk.calls[0]["options"].include_partial_messages is True

    def test_stream_ignores_non_text_events(self, fake_sdk):
        fake_sdk.scripted = [
            StreamEvent({"type": "message_start"}),
            StreamEvent(
                {"type": "content_block_delta", "delta": {"type": "input_json_delta"}}
            ),
            ResultMessage(),
        ]
        chunks = list(_make_provider().stream([Message(role=Role.USER, content="x")]))
        assert [c for c in chunks if c.content] == []

    def test_timeout_is_reported_as_transient(self, fake_sdk):
        async def slow(*, prompt, options):
            await asyncio.sleep(5)
            yield ResultMessage()

        fake_sdk.query = slow
        provider = _make_provider(timeout=0.05)
        with pytest.raises(ProviderTimeoutError, match="exceeded"):
            provider.invoke([Message(role=Role.USER, content="hi")])

    def test_sdk_exception_is_mapped(self, fake_sdk):
        fake_sdk.raises = CLINotFoundError("no cli")
        with pytest.raises(ProviderError, match="npm install"):
            _make_provider().invoke([Message(role=Role.USER, content="hi")])

    def test_async_invoke(self, fake_sdk):
        fake_sdk.scripted = [AssistantMessage([TextBlock("hi")]), ResultMessage()]
        response = asyncio.run(
            _make_provider().ainvoke([Message(role=Role.USER, content="hi")])
        )
        assert response.content == "hi"

    def test_async_stream(self, fake_sdk):
        fake_sdk.scripted = [
            StreamEvent(_text_delta("a")),
            ResultMessage(),
        ]

        async def collect():
            return [c async for c in _make_provider().astream(
                [Message(role=Role.USER, content="x")]
            )]

        chunks = asyncio.run(collect())
        assert "".join(c.content or "" for c in chunks) == "a"
