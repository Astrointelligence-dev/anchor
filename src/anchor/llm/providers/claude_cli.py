"""ClaudeCLIProvider — run Anchor on the Claude Code CLI's own credentials.

No ``ANTHROPIC_API_KEY``. Auth is whatever the ``claude`` CLI already has:
a Pro/Max subscription (OAuth) or a ``claude setup-token`` token.

The CLI is an agent, not a completion endpoint — it runs its own loop and
executes its own tools. Caller tools still round-trip: they are registered as
an in-process MCP server and a ``PreToolUse`` hook answers
``permissionDecision: "defer"``, which stops the run *without executing* and
reports the call back as ``deferred_tool_use``. Anchor's Agent stays the loop.

Closing the loop needs no CLI session: when the history already carries a
result for a call, the hook allows it through and the MCP handler returns that
stored result, so a replayed call resolves locally.

Set ``builtin_tools`` to also let the CLI's own tools (Read, Bash, ...) run
inside the run. That is opt-in for a reason — see the security note in
``ClaudeCLIProvider``.

Self-registers via register_provider() at module import time.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

from anchor.llm.base import BaseLLMProvider
from anchor.llm.errors import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderError,
    ProviderNotInstalledError,
    RateLimitError,
    ServerError,
)
from anchor.llm.errors import TimeoutError as ProviderTimeoutError
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

MCP_SERVER_NAME = "anchor"
MCP_TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"

_DEFER = {
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "defer",
    }
}


def _ensure_sdk() -> Any:
    """Import and return claude_agent_sdk, raising clearly if missing."""
    try:
        import claude_agent_sdk
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ProviderNotInstalledError(
            "claude_cli", "claude-agent-sdk", "claude-cli"
        ) from exc
    return claude_agent_sdk


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

_ROLE_LABEL = {Role.USER: "User", Role.ASSISTANT: "Assistant"}


def _canonical(args: dict[str, Any]) -> str:
    """Stable key for a tool call's arguments."""
    return json.dumps(args, sort_keys=True, default=str)


def _strip_namespace(name: str) -> str:
    return name[len(MCP_TOOL_PREFIX):] if name.startswith(MCP_TOOL_PREFIX) else name


def _message_text(msg: Message) -> str:
    """Text of a message. Non-text blocks are a hard error, not silent loss."""
    if msg.content is None:
        return ""
    if isinstance(msg.content, str):
        return msg.content
    parts: list[str] = []
    for block in msg.content:
        if block.type != "text" or block.text is None:
            raise ProviderError(
                f"claude_cli takes text only; got a {block.type!r} content block. "
                "Use an API-key provider for images.",
                provider="claude_cli",
            )
        parts.append(block.text)
    return "\n".join(parts)


def _flatten(messages: list[Message]) -> tuple[str | None, str]:
    """Split system out and fold the rest into one role-labelled prompt.

    ponytail: the CLI has no way to replay assistant turns (``--input-format
    stream-json`` treats every user message as a *new* turn), so history is
    flattened. Upgrade path is ``resume`` + a session prefix cache, which buys
    prompt-cache hits — not correctness.
    """
    system_parts: list[str] = []
    turns: list[tuple[Role, str]] = []

    for msg in messages:
        if msg.role is Role.SYSTEM:
            text = _message_text(msg)
            if text:
                system_parts.append(text)
            continue
        # Tool turns are delivered through the MCP handler, not the prompt.
        if msg.role is Role.TOOL:
            continue
        text = _message_text(msg)
        if text:
            turns.append((msg.role, text))

    system = "\n\n".join(system_parts) or None
    if len(turns) == 1:
        # Single turn — send it bare so the model sees a plain question.
        return system, turns[0][1]
    return system, "\n\n".join(f"{_ROLE_LABEL[r]}: {t}" for r, t in turns)


def _delivery_maps(
    messages: list[Message],
) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Pair executed tool calls with their results so a replay resolves locally."""
    calls: dict[str, tuple[str, str]] = {}
    by_key: dict[tuple[str, str], str] = {}
    by_name: dict[str, str] = {}

    for msg in messages:
        for call in msg.tool_calls or ():
            calls[call.id] = (call.name, _canonical(call.arguments))
        result = msg.tool_result
        if result is not None and result.tool_call_id in calls:
            name, args = calls[result.tool_call_id]
            by_key[(name, args)] = result.content
            by_name[name] = result.content
    return by_key, by_name


# ---------------------------------------------------------------------------
# Sync bridge — the SDK is async-only, LLMProvider is not
# ---------------------------------------------------------------------------

_DONE = object()


def _iter_sync(make_agen: Any) -> Iterator[Any]:
    """Drain an async generator from sync code on a private event loop.

    ponytail: unbounded queue, so the pump never blocks and the CLI
    subprocess always gets torn down even if the consumer stops early.
    Bound it if a response ever gets big enough to matter.
    """
    items: queue.Queue[Any] = queue.Queue()

    async def pump() -> None:
        try:
            async for item in make_agen():
                items.put(item)
        except BaseException as exc:  # re-raised on the caller's thread
            items.put(exc)
        finally:
            items.put(_DONE)

    threading.Thread(target=lambda: asyncio.run(pump()), daemon=True).start()
    while True:
        item = items.get()
        if item is _DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


class ClaudeCLIProvider(BaseLLMProvider):
    """Adapter for the Claude Code CLI, using its credentials instead of a key.

    Args:
        model: CLI model alias (``"sonnet"``, ``"opus"``) or full id.
        builtin_tools: let the CLI's *own* tools run inside the run —
            ``"claude_code"`` for the full preset, or a list of names
            (``["Read", "Grep"]``). Default ``None`` keeps the CLI a pure
            model. **Security:** enabling this runs those tools as your OS
            user, so a prompt injection in retrieved context becomes code
            execution. Only turn it on for trusted input, and prefer the
            narrowest tool list that works.
        permission_mode: CLI permission mode; only meaningful with
            ``builtin_tools``.
        setting_sources: which CLI setting sources to load. Default ``[]`` —
            without it the user's CLAUDE.md, MCP servers and settings leak in
            and cost tens of thousands of tokens per call.
        cwd: working directory for the CLI process.
        cli_path: explicit path to the ``claude`` binary.
        max_turns: cap on CLI turns; unset lets the CLI decide.

    ``max_tokens``, ``temperature`` and ``stop`` are accepted and ignored —
    the CLI exposes no equivalent. ``tool_choice="none"`` is exact (tools are
    not registered); ``"any"`` and a named tool degrade to a system-prompt
    instruction.
    """

    provider_name = "claude_cli"

    def __init__(
        self,
        model: str = "sonnet",
        *,
        builtin_tools: str | list[str] | None = None,
        permission_mode: str | None = None,
        setting_sources: list[str] | None = None,
        cwd: str | None = None,
        cli_path: str | None = None,
        max_turns: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        self._builtin_tools = builtin_tools
        self._permission_mode = permission_mode
        self._setting_sources = [] if setting_sources is None else setting_sources
        self._cwd = cwd
        self._cli_path = cli_path
        self._max_turns = max_turns

    def _resolve_api_key(self) -> str | None:
        """No key: the CLI carries its own credentials."""
        return None

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    def _builtin(self) -> Any:
        if self._builtin_tools is None:
            return []
        if self._builtin_tools == "claude_code":
            return {"type": "preset", "preset": "claude_code"}
        return list(self._builtin_tools)

    def _sdk_tools(
        self,
        sdk: Any,
        schemas: list[ToolSchema],
        by_key: dict[tuple[str, str], str],
        by_name: dict[str, str],
    ) -> list[Any]:
        """Wrap caller schemas as SDK MCP tools that deliver stored results."""

        def make_handler(name: str) -> Any:
            async def handler(args: dict[str, Any]) -> dict[str, Any]:
                text = by_key.get((name, _canonical(args)), by_name.get(name, ""))
                return {"content": [{"type": "text", "text": text}]}

            return handler

        return [
            sdk.tool(s.name, s.description, s.input_schema)(make_handler(s.name))
            for s in schemas
        ]

    def _defer_hooks(
        self,
        sdk: Any,
        by_key: dict[tuple[str, str], str],
        by_name: dict[str, str],
    ) -> dict[str, Any]:
        """Defer caller tools back to Anchor; allow the ones we can answer."""

        async def defer_hook(
            input_data: dict[str, Any], tool_use_id: str | None, context: Any
        ) -> dict[str, Any]:
            name = _strip_namespace(str(input_data.get("tool_name", "")))
            args = input_data.get("tool_input") or {}
            if (name, _canonical(args)) in by_key or name in by_name:
                return {}  # allow — the handler returns the caller's result
            return _DEFER

        return {
            "PreToolUse": [
                sdk.HookMatcher(matcher=f"{MCP_TOOL_PREFIX}.*", hooks=[defer_hook])
            ]
        }

    def _build_options(
        self,
        sdk: Any,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        *,
        partial: bool,
        tool_choice: str | dict[str, Any] | None,
    ) -> tuple[Any, str]:
        system, prompt = _flatten(messages)

        if tool_choice == "none":
            tools = None  # exact: the model cannot call anything
        elif tools and tool_choice:
            system = "\n\n".join(
                p for p in (system, _forced_tool_instruction(tool_choice)) if p
            )

        options = sdk.ClaudeAgentOptions(
            model=self._model,
            system_prompt=system,
            tools=self._builtin(),
            setting_sources=self._setting_sources,
            strict_mcp_config=True,
            include_partial_messages=partial,
        )
        allowed: list[str] = []
        # An explicit builtin list is the opt-in, so grant it: without this the
        # CLI asks for permission it cannot get in print mode and the tool dies.
        if isinstance(self._builtin_tools, list):
            allowed += list(self._builtin_tools)
        if tools:
            by_key, by_name = _delivery_maps(messages)
            options.mcp_servers = {
                MCP_SERVER_NAME: sdk.create_sdk_mcp_server(
                    name=MCP_SERVER_NAME,
                    version="1.0.0",
                    tools=self._sdk_tools(sdk, tools, by_key, by_name),
                )
            }
            allowed += [f"{MCP_TOOL_PREFIX}{s.name}" for s in tools]
            options.hooks = self._defer_hooks(sdk, by_key, by_name)
        if allowed:
            options.allowed_tools = allowed
        if self._permission_mode:
            options.permission_mode = self._permission_mode
        if self._cwd:
            options.cwd = self._cwd
        if self._cli_path:
            options.cli_path = self._cli_path
        if self._max_turns:
            options.max_turns = self._max_turns
        return options, prompt

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------

    async def _aiter_messages(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        *,
        partial: bool,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Yield raw SDK messages for one run. The only place that spawns a CLI."""
        sdk = _ensure_sdk()
        options, prompt = self._build_options(
            sdk, messages, tools, partial=partial,
            tool_choice=kwargs.get("tool_choice"),
        )
        try:
            async with asyncio.timeout(self._timeout):
                async for msg in sdk.query(prompt=prompt, options=options):
                    yield msg
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                f"claude CLI exceeded {self._timeout}s", provider=self.provider_name
            ) from exc
        except Exception as exc:
            raise self._map_error(exc) from exc

    # ------------------------------------------------------------------
    # invoke / stream
    # ------------------------------------------------------------------

    async def _do_ainvoke(
        self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs: Any
    ) -> LLMResponse:
        text_parts: list[str] = []
        result: Any = None
        async for msg in self._aiter_messages(
            messages, tools, partial=False, **kwargs
        ):
            kind = type(msg).__name__
            if kind == "AssistantMessage":
                text_parts.extend(_assistant_text(msg))
            elif kind == "ResultMessage":
                result = msg
        return self._build_response(result, text_parts)

    async def _do_astream(
        self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs: Any
    ) -> AsyncIterator[StreamChunk]:
        async for msg in self._aiter_messages(messages, tools, partial=True, **kwargs):
            kind = type(msg).__name__
            if kind == "StreamEvent":
                text = _delta_text(msg)
                if text:
                    yield StreamChunk(content=text)
            elif kind == "ResultMessage":
                yield self._final_chunk(msg)

    def _do_invoke(
        self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs: Any
    ) -> LLMResponse:
        text_parts: list[str] = []
        result: Any = None
        for msg in _iter_sync(
            lambda: self._aiter_messages(messages, tools, partial=False, **kwargs)
        ):
            kind = type(msg).__name__
            if kind == "AssistantMessage":
                text_parts.extend(_assistant_text(msg))
            elif kind == "ResultMessage":
                result = msg
        return self._build_response(result, text_parts)

    def _do_stream(
        self, messages: list[Message], tools: list[ToolSchema] | None, **kwargs: Any
    ) -> Iterator[StreamChunk]:
        for msg in _iter_sync(
            lambda: self._aiter_messages(messages, tools, partial=True, **kwargs)
        ):
            kind = type(msg).__name__
            if kind == "StreamEvent":
                text = _delta_text(msg)
                if text:
                    yield StreamChunk(content=text)
            elif kind == "ResultMessage":
                yield self._final_chunk(msg)

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _build_response(self, result: Any, text_parts: list[str]) -> LLMResponse:
        if result is None:
            raise ProviderError(
                "claude CLI ended without a result message",
                provider=self.provider_name,
                is_transient=True,
            )
        self._raise_if_error(result)
        deferred = getattr(result, "deferred_tool_use", None)
        # The CLI's own result text is the answer. Accumulated assistant text is
        # the fallback: it is empty on a deferred run, and in agentic mode it
        # also carries the intermediate narration we do not want here.
        content = (getattr(result, "result", "") or None) or (
            "".join(text_parts) or None
        )
        return LLMResponse(
            content=content,
            tool_calls=[_deferred_call(deferred)] if deferred is not None else None,
            usage=_usage(result),
            model=self._model,
            provider=self.provider_name,
            stop_reason=_stop_reason(result),
        )

    def _final_chunk(self, result: Any) -> StreamChunk:
        self._raise_if_error(result)
        deferred = getattr(result, "deferred_tool_use", None)
        delta = None
        if deferred is not None:
            delta = ToolCallDelta(
                index=0,
                id=deferred.id,
                name=_strip_namespace(deferred.name),
                arguments_fragment=json.dumps(deferred.input or {}),
            )
        return StreamChunk(
            tool_call_delta=delta,
            usage=_usage(result),
            stop_reason=_stop_reason(result),
        )

    def _raise_if_error(self, result: Any) -> None:
        if not getattr(result, "is_error", False):
            return
        raise self._status_error(
            getattr(result, "api_error_status", None),
            getattr(result, "result", None) or "claude CLI reported an error",
        )

    def _status_error(self, status: int | None, message: str) -> ProviderError:
        if status == 401 or status == 403:
            return AuthenticationError(message, provider=self.provider_name)
        if status == 404:
            return ModelNotFoundError(message, provider=self.provider_name)
        if status == 429:
            return RateLimitError(message, provider=self.provider_name)
        if status is not None and status >= 500:
            return ServerError(message, provider=self.provider_name)
        return ProviderError(message, provider=self.provider_name)

    def _map_error(self, exc: Exception) -> ProviderError:
        """Map an SDK exception to our hierarchy (class names, so mocks work)."""
        if isinstance(exc, ProviderError):
            return exc
        names = {cls.__name__ for cls in type(exc).__mro__}
        if "CLINotFoundError" in names:
            return ProviderError(
                f"{exc}. Install it with: npm install -g @anthropic-ai/claude-code",
                provider=self.provider_name,
            )
        if "ResultError" in names:
            return self._status_error(
                getattr(exc, "api_error_status", None), str(exc)
            )
        if "CLIConnectionError" in names or "ProcessError" in names:
            return ProviderError(str(exc), provider=self.provider_name, is_transient=True)
        return ProviderError(str(exc), provider=self.provider_name)


# ---------------------------------------------------------------------------
# SDK payload helpers (module-level: no provider state, easy to test)
# ---------------------------------------------------------------------------


def _forced_tool_instruction(tool_choice: str | dict[str, Any]) -> str:
    """Best-effort tool_choice. The CLI has no native equivalent."""
    if isinstance(tool_choice, dict) and tool_choice.get("name"):
        return f"You must call the {tool_choice['name']} tool before answering."
    if tool_choice == "any":
        return "You must call one of the available tools before answering."
    return ""


def _assistant_text(msg: Any) -> list[str]:
    return [
        block.text
        for block in getattr(msg, "content", None) or ()
        if type(block).__name__ == "TextBlock" and getattr(block, "text", None)
    ]


def _delta_text(msg: Any) -> str | None:
    event = getattr(msg, "event", None) or {}
    if event.get("type") != "content_block_delta":
        return None
    delta = event.get("delta") or {}
    if delta.get("type") != "text_delta":
        return None
    return delta.get("text") or None


def _deferred_call(deferred: Any) -> ToolCall:
    return ToolCall(
        id=deferred.id,
        name=_strip_namespace(deferred.name),
        arguments=deferred.input or {},
    )


def _usage(result: Any) -> Usage:
    raw = getattr(result, "usage", None) or {}
    prompt = int(raw.get("input_tokens", 0) or 0)
    completion = int(raw.get("output_tokens", 0) or 0)
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        total_cost=getattr(result, "total_cost_usd", None),
        cache_creation_tokens=int(raw.get("cache_creation_input_tokens", 0) or 0),
        cache_read_tokens=int(raw.get("cache_read_input_tokens", 0) or 0),
    )


def _stop_reason(result: Any) -> StopReason:
    if getattr(result, "deferred_tool_use", None) is not None:
        return StopReason.TOOL_USE
    raw = getattr(result, "stop_reason", None)
    if raw == "tool_deferred":
        return StopReason.TOOL_USE
    if raw == "max_tokens" or getattr(result, "subtype", None) == "error_max_turns":
        return StopReason.MAX_TOKENS
    return StopReason.STOP


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register_provider("claude_cli", ClaudeCLIProvider)
