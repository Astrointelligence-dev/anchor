"""Agent class that wraps ContextPipeline + LLMProvider + tool loop."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from anchor._callbacks import fire_callbacks
from anchor.formatters.anthropic import AnthropicFormatter
from anchor.llm.base import LLMProvider
from anchor.llm.models import (
    Message,
    Role,
    StopReason,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    ToolResult,
    ToolSchema,
)
from anchor.llm.registry import create_provider
from anchor.memory.manager import MemoryManager
from anchor.models.budget import TokenBudget
from anchor.models.context import ContextResult
from anchor.pipeline.pipeline import ContextPipeline
from anchor.protocols.tokenizer import Tokenizer

from .hooks import AgentCallback, PostToolHook, PreToolHook
from .models import AgentTool, RoundUsage, TurnDiagnostics
from .skills.activate import _make_activate_skill_tool
from .skills.models import Skill
from .skills.registry import SkillRegistry
from .skills.resources import _make_read_skill_file_tool, _make_run_skill_script_tool
from .subagent import (
    SubagentDefinition,
    _is_subagent_tool,
    _make_subagent_tool,
    _make_task_tool,
    _subagent_listing,
)
from .tool_search import _make_search_tools_tool

logger = logging.getLogger(__name__)

# Maximum character length for tool input/result recorded in memory.
_TOOL_MEMORY_TRUNCATE = 200

_FINAL_ROUND_NOTICE = (
    "[system] Final round: the tool budget is exhausted. Respond with "
    "your best final answer now — do not call tools."
)


class _WhitespaceTokenizer:
    """Minimal tokenizer that counts whitespace-separated words.

    Fallback when tiktoken (an optional extra) is unavailable.
    """

    __slots__ = ()

    def count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


def _default_tokenizer() -> Tokenizer:
    """Real tokenizer when tiktoken is installed, whitespace fallback otherwise."""
    try:
        from anchor.tokens.counter import get_default_counter

        return get_default_counter()
    except Exception:  # tiktoken missing or encoding data unavailable
        return _WhitespaceTokenizer()


class _RoundState:
    """Mutable accumulation state for one streamed round."""

    __slots__ = (
        "accumulators",
        "completion_tokens",
        "prompt_tokens",
        "stop_reason",
        "text",
    )

    def __init__(self) -> None:
        self.text = ""
        self.accumulators: dict[int, dict[str, Any]] = {}
        self.stop_reason: StopReason | None = None
        self.prompt_tokens = 0
        self.completion_tokens = 0


class Agent:
    """High-level agent combining context pipeline with an LLM provider.

    Provides streaming chat with automatic tool use, memory management,
    and agentic RAG -- all powered by the anchor pipeline.

    Usage::

        agent = (
            Agent(model="claude-haiku-4-5-20251001")
            .with_system_prompt("You are a helpful assistant.")
            .with_memory(memory)
            .with_tools(memory_tools(memory))
        )

        for chunk in agent.chat("Hello!"):
            print(chunk, end="", flush=True)
    """

    __slots__ = (
        "_activate_tool",
        "_agent_callbacks",
        "_allow_skill_scripts",
        "_deferred_loaded",
        "_last_result",
        "_last_turn",
        "_llm",
        "_max_response_tokens",
        "_max_rounds",
        "_mcp_configs",
        "_mcp_pool",
        "_mcp_tools",
        "_memory",
        "_pipeline",
        "_post_hooks",
        "_pre_hooks",
        "_read_file_tool",
        "_script_tool",
        "_search_tool",
        "_skill_registry",
        "_subagents",
        "_system_prompt",
        "_task_tool",
        "_tokenizer",
        "_tool_timeout",
        "_tools",
    )

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        api_key: str | None = None,
        llm: LLMProvider | None = None,
        fallbacks: list[str] | None = None,
        max_tokens: int = 16384,
        max_response_tokens: int = 1024,
        max_rounds: int = 10,
        allow_skill_scripts: bool = False,
        tool_timeout: float | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if llm is not None:
            self._llm: LLMProvider = llm
        else:
            self._llm = create_provider(model, api_key=api_key, fallbacks=fallbacks)

        self._max_response_tokens = max_response_tokens
        self._max_rounds = max_rounds
        self._system_prompt = ""
        self._tools: list[AgentTool] = []
        self._memory: MemoryManager | None = None
        self._last_result: ContextResult | None = None
        self._last_turn: TurnDiagnostics | None = None
        self._skill_registry = SkillRegistry()
        self._activate_tool: AgentTool | None = None
        self._read_file_tool: AgentTool | None = None
        self._script_tool: AgentTool | None = None
        self._allow_skill_scripts = allow_skill_scripts
        self._mcp_configs: list[Any] = []  # MCPServerConfig instances
        self._mcp_pool: Any = None  # MCPClientPool (lazy)
        self._mcp_tools: list[AgentTool] = []
        self._tool_timeout = tool_timeout
        self._pre_hooks: list[PreToolHook] = []
        self._post_hooks: list[PostToolHook] = []
        self._agent_callbacks: list[AgentCallback] = []
        self._deferred_loaded: set[str] = set()
        self._search_tool: AgentTool | None = None
        self._subagents: dict[str, tuple[SubagentDefinition, Agent]] = {}
        self._task_tool: AgentTool | None = None

        self._tokenizer: Tokenizer = tokenizer or _default_tokenizer()
        self._pipeline = ContextPipeline(
            max_tokens=max_tokens, tokenizer=self._tokenizer,
        )
        self._pipeline.with_formatter(AnthropicFormatter(enable_caching=True))

    # -- Fluent configuration (all return self) --

    def with_system_prompt(self, prompt: str) -> Agent:
        """Set the system prompt. Returns self for chaining."""
        self._system_prompt = prompt
        self._pipeline._system_items.clear()
        self._pipeline.add_system_prompt(prompt)
        return self

    def with_memory(self, memory: MemoryManager) -> Agent:
        """Attach memory for conversation history and facts. Returns self for chaining."""
        self._memory = memory
        self._pipeline.with_memory(memory)
        return self

    def with_tools(self, tools: list[AgentTool]) -> Agent:
        """Add tools (additive). Returns self for chaining."""
        self._tools.extend(tools)
        return self

    def with_budget(self, budget: TokenBudget) -> Agent:
        """Attach a token budget to the context pipeline. Returns self for chaining."""
        self._pipeline.with_budget(budget)
        return self

    def with_hooks(
        self,
        *,
        pre_tool_use: list[PreToolHook] | None = None,
        post_tool_use: list[PostToolHook] | None = None,
    ) -> Agent:
        """Add veto hooks around tool execution (additive). Returns self.

        Pre-hooks may deny a call (the reason is fed back to the model)
        or rewrite its input; post-hooks may replace the output. A
        pre-hook that raises fails closed: the call is denied.
        """
        self._pre_hooks.extend(pre_tool_use or [])
        self._post_hooks.extend(post_tool_use or [])
        return self

    def with_callbacks(self, callbacks: list[AgentCallback]) -> Agent:
        """Add observer callbacks for loop events (additive). Returns self.

        Callbacks are fire-and-forget: exceptions are swallowed and logged.
        """
        self._agent_callbacks.extend(callbacks)
        return self

    def with_skill(self, skill: Skill) -> Agent:
        """Register a skill. Returns self for chaining."""
        self._check_tool_collisions(skill)
        self._skill_registry.register(skill)
        self._ensure_skill_meta_tools()
        return self

    def with_skills(self, skills: list[Skill]) -> Agent:
        """Register multiple skills. Returns self for chaining."""
        for skill in skills:
            self._check_tool_collisions(skill)
            self._skill_registry.register(skill)
        self._ensure_skill_meta_tools()
        return self

    def with_skills_directory(self, path: str | Path) -> Agent:
        """Load all SKILL.md skills from a directory. Returns self for chaining."""
        self._skill_registry.load_from_directory(Path(path))
        self._ensure_skill_meta_tools()
        return self

    def with_mcp_servers(
        self,
        servers: list[str | Any],
    ) -> Agent:
        """Connect to external MCP servers. Returns self for chaining.

        Accepts MCPServerConfig objects or convenience strings
        (URLs for HTTP, commands for STDIO).
        """
        from anchor.mcp.tools import parse_server_string

        for server in servers:
            if isinstance(server, str):
                self._mcp_configs.append(parse_server_string(server))
            else:
                self._mcp_configs.append(server)
        return self

    def with_skill_from_path(self, path: str | Path) -> Agent:
        """Load one SKILL.md skill from a directory. Returns self for chaining."""
        self._skill_registry.load_from_path(Path(path))
        self._ensure_skill_meta_tools()
        return self

    def with_subagents(self, definitions: list[SubagentDefinition]) -> Agent:
        """Register subagents and the ``task`` meta-tool. Returns self.

        Each definition becomes an isolated sub-``Agent`` (own system
        prompt, restricted tools, no memory — MULTI_AGENT.md rule 1).
        The model delegates via ``task(agent_name, task)``; a discovery
        listing is appended to the system prompt.
        """
        for definition in definitions:
            if definition.name in self._subagents:
                msg = f"Duplicate subagent name: '{definition.name}'"
                raise ValueError(msg)
            if any(_is_subagent_tool(t) for t in definition.tools):
                msg = (
                    f"Subagent '{definition.name}' has subagent tools "
                    "(MULTI_AGENT.md: no nesting)"
                )
                raise ValueError(msg)
            if definition.model is None:
                sub = Agent(
                    llm=self._llm,
                    max_rounds=definition.max_rounds,
                    tokenizer=self._tokenizer,
                )
            else:
                sub = Agent(
                    model=definition.model,
                    max_rounds=definition.max_rounds,
                    tokenizer=self._tokenizer,
                )
            if definition.system_prompt:
                sub.with_system_prompt(definition.system_prompt)
            if definition.tools:
                sub.with_tools(list(definition.tools))
            self._subagents[definition.name] = (definition, sub)

        if self._subagents and self._task_tool is None:
            self._task_tool = _make_task_tool(self._subagents)
        return self

    def as_tool(
        self,
        name: str,
        description: str,
        *,
        output_model: type[BaseModel] | None = None,
        max_output_retries: int = 1,
    ) -> AgentTool:
        """Expose this agent as a subagent tool for an orchestrator.

        The tool takes a single ``task`` string — the subagent starts
        with a clean context (MULTI_AGENT.md rule 1). With
        ``output_model``, the subagent must return schema-valid JSON;
        invalid output is retried ``max_output_retries`` times with the
        validation error (rule 3). Agents that already have subagent
        tools cannot be wrapped (rule 5: no nesting).
        """
        return _make_subagent_tool(
            self,
            name,
            description,
            output_model=output_model,
            max_output_retries=max_output_retries,
        )

    # -- Accessors --

    @property
    def memory(self) -> MemoryManager | None:
        """The attached memory manager, if any."""
        return self._memory

    @property
    def pipeline(self) -> ContextPipeline:
        """The underlying context pipeline."""
        return self._pipeline

    @property
    def last_result(self) -> ContextResult | None:
        """The ContextResult from the most recent ``chat()`` call."""
        return self._last_result

    @property
    def last_turn(self) -> TurnDiagnostics | None:
        """Per-round token accounting for the most recent turn."""
        return self._last_turn

    # -- Internal helpers --

    def _all_active_tools(self) -> list[AgentTool]:
        """Return direct tools + skill tools + meta-tools + MCP tools."""
        tools: list[AgentTool] = list(self._tools)
        tools.extend(self._skill_registry.active_tools())
        if self._activate_tool is not None:
            tools.append(self._activate_tool)
        if self._read_file_tool is not None:
            tools.append(self._read_file_tool)
        if self._script_tool is not None:
            tools.append(self._script_tool)
        if self._task_tool is not None:
            tools.append(self._task_tool)
        if self._search_tool is not None:
            tools.append(self._search_tool)
        tools.extend(self._mcp_tools)
        return tools

    def _sendable_tools(self) -> list[AgentTool]:
        """Tools whose schemas are sent this round.

        Deferred tools are excluded until loaded through the
        ``search_tools`` meta-tool, which is auto-created as soon as
        any deferred tool exists.
        """
        tools = self._all_active_tools()
        if self._search_tool is None and any(t.defer_loading for t in tools):
            self._search_tool = _make_search_tools_tool(self)
            tools.append(self._search_tool)
        return [
            t
            for t in tools
            if not t.defer_loading or t.name in self._deferred_loaded
        ]

    def _check_tool_collisions(self, skill: Skill) -> None:
        """Raise if a skill tool name collides with a direct agent tool."""
        direct_names = {t.name for t in self._tools}
        for tool in skill.tools:
            if tool.name in direct_names:
                msg = (
                    f"Tool name collision: skill '{skill.name}' provides "
                    f"'{tool.name}', already registered via with_tools()"
                )
                raise ValueError(msg)

    def _ensure_skill_meta_tools(self) -> None:
        """Create skill meta-tools as they become relevant.

        ``activate_skill`` appears once an on-demand skill is registered;
        ``read_skill_file`` once any skill has an on-disk directory;
        ``run_skill_script`` additionally requires ``allow_skill_scripts=True``
        and at least one bundled script.
        """
        registry = self._skill_registry
        if registry.on_demand_skills() and self._activate_tool is None:
            self._activate_tool = _make_activate_skill_tool(registry)

        has_paths = any(s.path is not None for s in registry.all_skills())
        if has_paths and self._read_file_tool is None:
            self._read_file_tool = _make_read_skill_file_tool(registry)

        has_scripts = any(s.script_files() for s in registry.all_skills())
        if self._allow_skill_scripts and has_scripts and self._script_tool is None:
            self._script_tool = _make_run_skill_script_tool(registry)

    def _system_suffix(self) -> str:
        """Static text appended to the system prompt each turn.

        Contains always-skill instructions, the on-demand skill
        discovery listing, and the subagent listing. Built once per
        turn (not per round) and stable across activation state —
        prompt-cache friendly. The skill listing is capped at roughly
        1% of the pipeline's token budget.
        """
        parts: list[str] = []
        always = self._skill_registry.always_instructions()
        if always:
            parts.append(always)
        # ~4 chars/token heuristic; 1% of the context budget for the listing.
        max_chars = max(400, (self._pipeline.max_tokens * 4) // 100)
        discovery = self._skill_registry.skill_discovery_prompt(max_chars=max_chars)
        if discovery:
            parts.append(discovery)
        subagents = _subagent_listing(self._subagents)
        if subagents:
            parts.append(subagents)
        return "\n\n".join(parts)

    def _fire(self, method: str, *args: Any) -> None:
        """Notify observer callbacks; exceptions are swallowed and logged."""
        if self._agent_callbacks:
            fire_callbacks(self._agent_callbacks, method, *args, logger=logger)

    # -- Tool execution --

    def _find_tool(self, name: str) -> AgentTool | None:
        for tool in self._all_active_tools():
            if tool.name == name:
                return tool
        return None

    def _apply_pre_hooks(
        self, name: str, tool_input: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        """Run pre-hooks. Returns (possibly-updated input, deny reason)."""
        for hook in self._pre_hooks:
            try:
                result = hook(name, tool_input)
            except Exception as exc:  # fail closed: a broken gate stays shut
                logger.exception("Pre-tool hook failed for '%s'", name)
                reason = f"pre-tool hook raised {type(exc).__name__}: {exc}"
                return tool_input, reason
            if result is None:
                continue
            if result.decision == "deny":
                return tool_input, result.reason or "denied by pre-tool hook"
            if result.updated_input is not None:
                tool_input = result.updated_input
        return tool_input, None

    def _apply_post_hooks(
        self, name: str, tool_input: dict[str, Any], output: str,
    ) -> str:
        """Run post-hooks; a raising post-hook keeps the current output."""
        for hook in self._post_hooks:
            try:
                result = hook(name, tool_input, output)
            except Exception:
                logger.exception("Post-tool hook failed for '%s'", name)
                continue
            if result is not None and result.updated_output is not None:
                output = result.updated_output
        return output

    def _resolve_call(
        self, name: str, tool_input: dict[str, Any],
    ) -> tuple[AgentTool | None, dict[str, Any], str | None]:
        """Look up the tool, validate input, and run pre-hooks.

        Returns ``(tool, updated_input, error_text)``; when
        ``error_text`` is set the call must not execute.
        """
        tool = self._find_tool(name)
        if tool is None:
            return None, tool_input, f"Unknown tool: {name}"
        valid, err = tool.validate_input(tool_input)
        if not valid:
            logger.warning("Tool '%s' input validation failed: %s", name, err)
            return tool, tool_input, f"Error: invalid input for tool '{name}': {err}"
        tool_input, deny = self._apply_pre_hooks(name, tool_input)
        if deny is not None:
            return tool, tool_input, f"Error: tool '{name}' denied: {deny}"
        return tool, tool_input, None

    def _error_result(
        self, tc: ToolCall, tool_input: dict[str, Any], error_text: str,
    ) -> ToolResult:
        self._fire("on_tool_error", tc.name, tool_input, error_text)
        self._record_tool_call(tc.name, tool_input, error_text)
        return ToolResult(tool_call_id=tc.id, content=error_text, is_error=True)

    def _ok_result(
        self, tc: ToolCall, tool_input: dict[str, Any], output: str,
    ) -> ToolResult:
        output = self._apply_post_hooks(tc.name, tool_input, output)
        self._fire("on_tool_end", tc.name, tool_input, output)
        self._record_tool_call(tc.name, tool_input, output)
        return ToolResult(tool_call_id=tc.id, content=output)

    def _execute_call(self, tc: ToolCall) -> ToolResult:
        """Execute one tool call synchronously."""
        tool, tool_input, err = self._resolve_call(tc.name, tc.arguments)
        if tool is None or err is not None:
            return self._error_result(tc, tool_input, err or f"Unknown tool: {tc.name}")
        self._fire("on_tool_start", tc.name, tool_input)
        try:
            # ponytail: no timeout on sync tools — cancelling a sync call
            # needs a worker thread; the async path enforces tool.timeout.
            output = tool.fn(**tool_input)
        except Exception as exc:
            logger.exception("Tool '%s' failed", tc.name)
            error_text = (
                f"Error: tool '{tc.name}' failed: {type(exc).__name__}: {exc}"
            )
            return self._error_result(tc, tool_input, error_text)
        return self._ok_result(tc, tool_input, output)

    async def _aexecute_call(self, tc: ToolCall) -> ToolResult:
        """Execute one tool call asynchronously (MCP/subagent-aware)."""
        tool, tool_input, err = self._resolve_call(tc.name, tc.arguments)
        if tool is None or err is not None:
            return self._error_result(tc, tool_input, err or f"Unknown tool: {tc.name}")
        self._fire("on_tool_start", tc.name, tool_input)
        async_caller = getattr(tool, "_mcp_async_caller", None) or getattr(
            tool, "_anchor_async_caller", None,
        )
        timeout = tool.timeout if tool.timeout is not None else self._tool_timeout
        try:
            if async_caller is not None:
                original_name = getattr(tool, "_mcp_original_name", tc.name)
                coro = async_caller(original_name, tool_input)
                if timeout is not None:
                    output = await asyncio.wait_for(coro, timeout)
                else:
                    output = await coro
            else:
                # ponytail: sync tools run inline (blocking, no timeout);
                # switch to asyncio.to_thread if cancellation ever matters.
                output = tool.fn(**tool_input)
        except TimeoutError:
            error_text = f"Error: tool '{tc.name}' timed out after {timeout}s."
            return self._error_result(tc, tool_input, error_text)
        except Exception as exc:
            logger.exception("Tool '%s' failed", tc.name)
            error_text = (
                f"Error: tool '{tc.name}' failed: {type(exc).__name__}: {exc}"
            )
            return self._error_result(tc, tool_input, error_text)
        return self._ok_result(tc, tool_input, output)

    async def _aexecute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """Async tool execution by name; returns the result text."""
        result = await self._aexecute_call(
            ToolCall(id="direct", name=name, arguments=tool_input),
        )
        return result.content

    def _run_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls sequentially and return ToolResult objects."""
        return [self._execute_call(tc) for tc in tool_calls]

    async def _arun_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls concurrently, preserving call order."""
        results = await asyncio.gather(
            *(self._aexecute_call(tc) for tc in tool_calls),
            return_exceptions=True,
        )
        ordered: list[ToolResult] = []
        for tc, result in zip(tool_calls, results, strict=True):
            if isinstance(result, BaseException):
                error_text = (
                    f"Error: tool '{tc.name}' failed: "
                    f"{type(result).__name__}: {result}"
                )
                ordered.append(
                    ToolResult(tool_call_id=tc.id, content=error_text, is_error=True),
                )
            else:
                ordered.append(result)
        return ordered

    def _record_tool_call(
        self, name: str, tool_input: dict[str, Any], result: str,
    ) -> None:
        """Record a tool call in memory for conversation history."""
        if self._memory is not None:
            input_str = json.dumps(tool_input)[:_TOOL_MEMORY_TRUNCATE]
            result_str = result[:_TOOL_MEMORY_TRUNCATE]
            tool_summary = (
                f"[Tool: {name}] Input: {input_str} → Result: {result_str}"
            )
            self._memory.add_tool_message(tool_summary)

    @staticmethod
    def _build_tool_calls(
        accumulators: dict[int, dict[str, Any]],
    ) -> list[ToolCall]:
        """Convert accumulated tool call deltas into ToolCall objects."""
        calls: list[ToolCall] = []
        for _idx in sorted(accumulators):
            acc = accumulators[_idx]
            args = json.loads(acc["args_json"]) if acc["args_json"] else {}
            calls.append(ToolCall(id=acc["id"], name=acc["name"], arguments=args))
        return calls

    def _formatted_to_messages(
        self, formatted: dict[str, Any],
    ) -> tuple[list[Message], str | None]:
        """Convert pipeline formatted output to list[Message].

        Returns (messages, system_text) where system_text is extracted
        for providers that handle system messages separately.
        """
        messages: list[Message] = []
        system_text: str | None = None

        # Handle Anthropic format (system is separate)
        if "system" in formatted:
            system_parts = formatted["system"]
            if isinstance(system_parts, list):
                texts = [b["text"] for b in system_parts if b.get("text")]
                if texts:
                    system_text = " ".join(texts)
            elif isinstance(system_parts, str):
                system_text = system_parts

        for msg in formatted.get("messages", []):
            role_str = msg.get("role", "user")
            content = msg.get("content")

            if isinstance(content, str):
                messages.append(Message(role=Role(role_str), content=content))
            elif isinstance(content, list):
                # Content blocks (from tool_use / tool_result responses)
                # These should not appear in the initial pipeline output,
                # but handle for completeness.
                text_parts = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                if text_parts:
                    messages.append(
                        Message(role=Role(role_str), content=" ".join(text_parts)),
                    )
            elif content is None:
                messages.append(Message(role=Role(role_str)))

        return messages, system_text

    # -- Turn/round helpers (shared by chat and achat) --

    def _prepare_turn(
        self, ctx_result: ContextResult, message: str,
    ) -> tuple[list[Message], str]:
        """Extract base messages and the full system text for a turn."""
        self._last_result = ctx_result
        formatted = ctx_result.formatted_output
        if not isinstance(formatted, dict):
            msg = "Agent requires a dict-based formatter output"
            raise TypeError(msg)
        base_messages, base_system = self._formatted_to_messages(formatted)

        # Without memory the formatter emits no conversation, so the
        # user's message would never reach the model — guarantee it.
        if not any(m.role == Role.USER for m in base_messages):
            base_messages.append(Message(role=Role.USER, content=message))

        # Suffix text (skills + subagents) is static per turn: compute
        # once, keep the prompt stable across rounds so provider prompt
        # caching can hit.
        suffix = self._system_suffix()
        if base_system and suffix:
            full_system = f"{base_system}\n\n{suffix}"
        else:
            full_system = base_system or suffix or ""
        return list(base_messages), full_system

    def _round_request(
        self, full_system: str, messages: list[Message],
    ) -> tuple[list[Message], list[ToolSchema] | None, int]:
        """Assemble one round's request. Returns (messages, schemas, schema tokens)."""
        tools = self._sendable_tools()
        schemas = [t.to_tool_schema() for t in tools] if tools else None
        llm_messages: list[Message] = []
        if full_system:
            llm_messages.append(Message(role=Role.SYSTEM, content=full_system))
        llm_messages.extend(messages)
        schema_tokens = 0
        if schemas:
            schema_tokens = sum(
                self._tokenizer.count_tokens(
                    f"{s.name} {s.description} {json.dumps(s.input_schema)}",
                )
                for s in schemas
            )
        return llm_messages, schemas, schema_tokens

    @staticmethod
    def _ingest_chunk(state: _RoundState, chunk: StreamChunk) -> str | None:
        """Fold one stream chunk into *state*; return text to yield."""
        out: str | None = None
        if chunk.content:
            state.text += chunk.content
            out = chunk.content
        if chunk.tool_call_delta:
            delta: ToolCallDelta = chunk.tool_call_delta
            acc = state.accumulators.setdefault(
                delta.index, {"id": None, "name": None, "args_json": ""},
            )
            if delta.id:
                acc["id"] = delta.id
            if delta.name:
                acc["name"] = delta.name
            if delta.arguments_fragment:
                acc["args_json"] += delta.arguments_fragment
        if chunk.usage:
            state.prompt_tokens = max(state.prompt_tokens, chunk.usage.prompt_tokens)
            state.completion_tokens = max(
                state.completion_tokens, chunk.usage.completion_tokens,
            )
        if chunk.stop_reason:
            state.stop_reason = chunk.stop_reason
        return out

    def _round_usage(
        self,
        index: int,
        state: _RoundState,
        schema_tokens: int,
        tool_results: list[ToolResult],
    ) -> RoundUsage:
        result_tokens = sum(
            self._tokenizer.count_tokens(r.content) for r in tool_results
        )
        return RoundUsage(
            round=index,
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            tool_schema_tokens=schema_tokens,
            tool_result_tokens=result_tokens,
        )

    def _maybe_final_round_notice(
        self, round_index: int, messages: list[Message],
    ) -> None:
        """Warn the model one round before the limit so it can wrap up."""
        if round_index and round_index == self._max_rounds - 1:
            messages.append(Message(role=Role.USER, content=_FINAL_ROUND_NOTICE))

    def _finish_turn(
        self,
        rounds: list[RoundUsage],
        stopped_by: str,
        final_text: str,
    ) -> None:
        self._last_turn = TurnDiagnostics(
            rounds=tuple(rounds),
            stopped_by=stopped_by,  # type: ignore[arg-type]
        )
        if self._memory is not None and final_text:
            self._memory.add_assistant_message(final_text)

    @staticmethod
    def _stop_cause(stop_reason: StopReason | None) -> str:
        return "max_tokens" if stop_reason == StopReason.MAX_TOKENS else "stop"

    # -- Chat --

    def chat(self, message: str) -> Iterator[str]:
        """Send a message and stream the response.

        Handles the full tool-use loop: if the model calls tools,
        they are executed and results fed back until the model
        produces a final text response or ``max_rounds`` is reached.

        Yields text chunks as they arrive from the API.
        """
        if self._mcp_configs:
            msg = (
                "MCP servers require async execution. "
                "Use agent.achat() instead of agent.chat()."
            )
            raise TypeError(msg)
        if self._memory is not None:
            self._memory.add_user_message(message)

        messages, full_system = self._prepare_turn(
            self._pipeline.build(message), message,
        )
        final_text = ""
        rounds: list[RoundUsage] = []
        stopped_by = "max_rounds"

        for round_index in range(self._max_rounds):
            self._fire("on_round_start", round_index)
            self._maybe_final_round_notice(round_index, messages)
            llm_messages, schemas, schema_tokens = self._round_request(
                full_system, messages,
            )
            state = _RoundState()

            for chunk in self._llm.stream(
                llm_messages,
                tools=schemas,
                max_tokens=self._max_response_tokens,
            ):
                out = self._ingest_chunk(state, chunk)
                if out:
                    final_text += out
                    yield out

            if state.stop_reason != StopReason.TOOL_USE:
                rounds.append(self._round_usage(round_index, state, schema_tokens, []))
                stopped_by = self._stop_cause(state.stop_reason)
                self._fire("on_round_end", round_index)
                break

            if round_index == self._max_rounds - 1:
                # Final round still asked for tools: don't execute work
                # whose results the model will never see.
                rounds.append(self._round_usage(round_index, state, schema_tokens, []))
                self._fire("on_round_end", round_index)
                break

            tool_calls = self._build_tool_calls(state.accumulators)
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=state.text or None,
                    tool_calls=tool_calls,
                ),
            )
            tool_results = self._run_tools(tool_calls)
            for result in tool_results:
                messages.append(Message(role=Role.TOOL, tool_result=result))
            rounds.append(
                self._round_usage(round_index, state, schema_tokens, tool_results),
            )
            self._fire("on_round_end", round_index)

        self._finish_turn(rounds, stopped_by, final_text)

    async def achat(self, message: str) -> AsyncGenerator[str, None]:
        """Send a message and stream the response asynchronously.

        Async mirror of :meth:`chat`. Uses ``pipeline.abuild()``, async
        iteration over the streaming API, and executes independent tool
        calls concurrently.

        Yields text chunks as they arrive from the API.
        """
        if self._memory is not None:
            self._memory.add_user_message(message)

        # Lazy MCP connection
        if self._mcp_configs and self._mcp_pool is None:
            from anchor.mcp.client import MCPClientPool
            pool = MCPClientPool(self._mcp_configs)
            try:
                await pool.connect_all()
                self._mcp_tools = await pool.all_agent_tools()
                self._mcp_pool = pool
            except Exception:
                await pool.disconnect_all()
                raise

        messages, full_system = self._prepare_turn(
            await self._pipeline.abuild(message), message,
        )
        final_text = ""
        rounds: list[RoundUsage] = []
        stopped_by = "max_rounds"

        for round_index in range(self._max_rounds):
            self._fire("on_round_start", round_index)
            self._maybe_final_round_notice(round_index, messages)
            llm_messages, schemas, schema_tokens = self._round_request(
                full_system, messages,
            )
            state = _RoundState()

            async for chunk in self._llm.astream(
                llm_messages,
                tools=schemas,
                max_tokens=self._max_response_tokens,
            ):
                out = self._ingest_chunk(state, chunk)
                if out:
                    final_text += out
                    yield out

            if state.stop_reason != StopReason.TOOL_USE:
                rounds.append(self._round_usage(round_index, state, schema_tokens, []))
                stopped_by = self._stop_cause(state.stop_reason)
                self._fire("on_round_end", round_index)
                break

            if round_index == self._max_rounds - 1:
                # Final round still asked for tools: don't execute work
                # whose results the model will never see.
                rounds.append(self._round_usage(round_index, state, schema_tokens, []))
                self._fire("on_round_end", round_index)
                break

            tool_calls = self._build_tool_calls(state.accumulators)
            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=state.text or None,
                    tool_calls=tool_calls,
                ),
            )
            tool_results = await self._arun_tools(tool_calls)
            for result in tool_results:
                messages.append(Message(role=Role.TOOL, tool_result=result))
            rounds.append(
                self._round_usage(round_index, state, schema_tokens, tool_results),
            )
            self._fire("on_round_end", round_index)

        self._finish_turn(rounds, stopped_by, final_text)

    async def aclose(self) -> None:
        """Clean up MCP connections and other async resources."""
        if self._mcp_pool is not None:
            await self._mcp_pool.disconnect_all()
            self._mcp_pool = None
            self._mcp_tools = []

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.aclose()
