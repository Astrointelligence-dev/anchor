"""Agent class that wraps ContextPipeline + LLMProvider + tool loop."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncGenerator, Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

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

from .events import (
    _EVENT_SINK,
    AgentEvent,
    CompactionFinished,
    CompactionStarted,
    RoundFinished,
    RoundStarted,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    TurnStarted,
    UsageLimitReached,
)
from .hooks import (
    AgentCallback,
    ApprovalCallback,
    ApprovalDecision,
    ApprovalRequest,
    PostToolHook,
    PreToolHook,
)
from .memory_tool import MEMORY_INSTRUCTIONS, FileMemoryBackend, memory_tool
from .models import AgentTool, RoundUsage, TurnDiagnostics, UsageLimits
from .skills.activate import _make_activate_skill_tool
from .skills.models import Skill
from .skills.registry import SkillRegistry
from .skills.resources import _make_read_skill_file_tool, _make_run_skill_script_tool
from .subagent import (
    SubagentDefinition,
    _is_subagent_tool,
    _make_subagent_tool,
    _make_task_tool,
    _run_async,
    _run_sync,
    _subagent_listing,
)
from .tool_search import _make_search_tools_tool

logger = logging.getLogger(__name__)

# Maximum character length for tool input/result recorded in memory.
_TOOL_MEMORY_TRUNCATE = 200

_FINAL_ROUND_NOTICE = (
    "[system] Final round: the budget for this turn is exhausted. "
    "Respond with your best final answer now — do not call tools."
)

_FINAL_ROUND_NOTICE_OUTPUT = (
    "[system] Final round: the budget for this turn is exhausted. "
    "Record your final answer by calling the final_result tool now."
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
        "cache_creation_tokens",
        "cache_read_tokens",
        "completion_tokens",
        "prompt_tokens",
        "raw_blocks",
        "stop_reason",
        "text",
    )

    def __init__(self) -> None:
        self.text = ""
        self.accumulators: dict[int, dict[str, Any]] = {}
        self.stop_reason: StopReason | None = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0
        self.raw_blocks: list[dict[str, Any]] = []


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
        "_approval_callback",
        "_compact_fn",
        "_compaction_keep",
        "_compaction_trigger",
        "_context_management",
        "_deferred_loaded",
        "_last_output",
        "_last_result",
        "_last_turn",
        "_llm",
        "_max_output_retries",
        "_max_response_tokens",
        "_max_rounds",
        "_mcp_configs",
        "_mcp_pool",
        "_mcp_tools",
        "_memory",
        "_memory_tool",
        "_output_failures",
        "_output_mode",
        "_output_model",
        "_output_tool",
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
        "_usage_limits",
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
        self._context_management: dict[str, Any] | None = None
        self._compaction_trigger: int | None = None
        self._compaction_keep = 4
        self._compact_fn: Callable[[str], str] | None = None
        self._usage_limits: UsageLimits | None = None
        self._approval_callback: ApprovalCallback | None = None
        self._output_model: type[BaseModel] | None = None
        self._output_mode = "tool"
        self._output_tool: AgentTool | None = None
        self._max_output_retries = 1
        self._output_failures = 0
        self._last_output: str | None = None
        self._memory_tool: AgentTool | None = None

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
        for t in tools:
            if (
                (t.name == "task" and self._task_tool is not None)
                or (t.name == "search_tools" and self._search_tool is not None)
                or (t.name == "final_result" and self._output_tool is not None)
                or (t.name == "memory" and self._memory_tool is not None)
            ):
                msg = (
                    f"Tool name collision: '{t.name}' is already registered "
                    "as an agent meta-tool"
                )
                raise ValueError(msg)
        self._tools.extend(tools)
        return self

    def with_budget(self, budget: TokenBudget) -> Agent:
        """Attach a token budget to the context pipeline. Returns self for chaining."""
        self._pipeline.with_budget(budget)
        return self

    def with_usage_limits(self, limits: UsageLimits) -> Agent:
        """Enforce per-turn usage limits in the tool loop. Returns self.

        Crossing a limit grants the model one wrap-up round (final-round
        notice + ``tool_choice="none"``) and ends the turn with
        ``stopped_by="usage_limit"`` — no exception. Complements the
        pipeline's :class:`TokenBudget`, which governs what enters the
        context window; this caps what the whole turn may spend.
        """
        self._usage_limits = limits
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

    def with_approval(self, callback: ApprovalCallback) -> Agent:
        """Set the inline human-in-the-loop approval callback. Returns self.

        Called for tools marked ``requires_approval=True`` and for
        calls a pre-hook answered ``"ask"`` — the tool call pauses
        until the callback returns an :class:`ApprovalDecision` (deny
        becomes an ``is_error`` tool result carrying the reason; an
        approval may rewrite the input). An async callback requires
        the async loop. Without a callback, approval-gated calls fail
        closed. Parallel tool calls run their approval callbacks
        concurrently — serialize inside the callback (e.g. a lock or a
        queue) if your UI needs one prompt at a time.
        """
        self._approval_callback = callback
        return self

    def with_output_model(
        self,
        output_model: type[BaseModel],
        *,
        mode: str = "tool",
        max_output_retries: int = 1,
    ) -> Agent:
        """Require schema-validated structured output. Returns self.

        ``mode="tool"`` (default, portable): a synthetic ``final_result``
        tool carries the schema; ``tool_choice="any"`` keeps the model
        from stopping in plain text; calling it with valid arguments
        ends the turn (``TurnFinished.output`` / ``agent.last_output`` /
        :meth:`run`). Invalid arguments come back as an error tool
        result — the loop's own retry mechanic — bounded by
        ``max_output_retries``, then ``ValueError``.

        ``mode="prompted"``: the schema is appended to the prompt and
        the reply is validated, with self-contained retry turns (the
        subagent mechanic). Use :meth:`run`/:meth:`arun`.
        """
        if mode not in ("tool", "prompted"):
            msg = f"Unknown output mode: {mode!r} (expected 'tool' or 'prompted')"
            raise ValueError(msg)
        self._output_tool = None  # reconfiguration replaces prior state
        if mode == "tool":
            if any(t.name == "final_result" for t in self._all_active_tools()):
                msg = (
                    "Tool name collision: 'final_result' is reserved for "
                    "the structured-output tool"
                )
                raise ValueError(msg)
            self._output_tool = self._make_output_tool(output_model)
        self._output_model = output_model
        self._output_mode = mode
        self._max_output_retries = max_output_retries
        return self

    def _make_output_tool(self, output_model: type[BaseModel]) -> AgentTool:
        def record(**kwargs: Any) -> str:
            try:
                parsed = output_model.model_validate(kwargs)
            except ValidationError as exc:
                self._output_failures += 1
                raise ValueError(str(exc)) from exc
            self._last_output = parsed.model_dump_json()
            return "Final answer recorded."

        return AgentTool(
            name="final_result",
            description=(
                "Record your final answer. Call this exactly once, when "
                "you are done, with the complete answer as arguments."
            ),
            input_schema=output_model.model_json_schema(),
            fn=record,
        )

    def with_callbacks(self, callbacks: list[AgentCallback]) -> Agent:
        """Add observer callbacks for loop events (additive). Returns self.

        Callbacks are fire-and-forget: exceptions are swallowed and logged.
        """
        self._agent_callbacks.extend(callbacks)
        return self

    def with_context_management(self, config: dict[str, Any]) -> Agent:
        """Attach an Anthropic ``context_management`` config. Returns self.

        Passed through with every request; the Anthropic provider routes
        to the beta API with the required flags, other providers ignore
        it. ``compaction`` blocks returned by the API round-trip
        verbatim within the turn. For a provider-agnostic alternative,
        see :meth:`with_compaction`.
        """
        self._context_management = config
        return self

    def with_compaction(
        self,
        trigger_tokens: int,
        *,
        keep_last: int = 4,
        compact_fn: Callable[[str], str] | None = None,
    ) -> Agent:
        """Enable client-side compaction of the tool loop. Returns self.

        Works with every provider. When the turn's working messages
        exceed ``trigger_tokens``, older messages are summarized into a
        single ``[Conversation summary]`` user message; the last
        ``keep_last`` messages are kept intact. ``compact_fn`` receives
        the flattened transcript and returns the summary; the default
        uses ``TierCompactor`` with this agent's own LLM.
        """
        self._compaction_trigger = trigger_tokens
        self._compaction_keep = keep_last
        self._compact_fn = compact_fn
        return self

    def with_memory_tool(
        self, backend: FileMemoryBackend | str | Path,
    ) -> Agent:
        """Attach the client-side memory tool (memory_20250818-compatible).

        Accepts a :class:`FileMemoryBackend` or a base path. Registers
        the ``memory`` tool and appends the memory protocol to the
        system prompt (multi-provider — no automatic API injection
        here). Returns self.
        """
        if any(t.name == "memory" for t in self._all_active_tools()):
            msg = "Tool name collision: 'memory' is already registered"
            raise ValueError(msg)
        if not isinstance(backend, FileMemoryBackend):
            backend = FileMemoryBackend(backend)
        self._memory_tool = memory_tool(backend)
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
            if any(t.name == "task" for t in self._all_active_tools()):
                msg = (
                    "Tool name collision: 'task' is reserved for the "
                    "subagent dispatcher"
                )
                raise ValueError(msg)
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

    @property
    def last_output(self) -> str | None:
        """Normalized structured-output JSON from the most recent turn."""
        return self._last_output

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
        if self._output_tool is not None:
            tools.append(self._output_tool)
        if self._memory_tool is not None:
            tools.append(self._memory_tool)
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
            if any(t.name == "search_tools" for t in tools):
                msg = (
                    "Tool name collision: 'search_tools' is reserved for "
                    "the deferred-tool search meta-tool"
                )
                raise ValueError(msg)
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
        if self._memory_tool is not None:
            parts.append(MEMORY_INSTRUCTIONS)
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
    ) -> tuple[dict[str, Any], str | None, bool]:
        """Run pre-hooks. Returns (input, deny reason, ask).

        ``ask`` routes the call to the approval callback; a later
        hook's deny still wins over an earlier ask.
        """
        ask = False
        for hook in self._pre_hooks:
            try:
                result = hook(name, tool_input)
            except Exception as exc:  # fail closed: a broken gate stays shut
                logger.exception("Pre-tool hook failed for '%s'", name)
                reason = f"pre-tool hook raised {type(exc).__name__}: {exc}"
                return tool_input, reason, ask
            if result is None:
                continue
            if result.decision == "deny":
                return tool_input, result.reason or "denied by pre-tool hook", ask
            if result.decision == "ask":
                ask = True
            if result.updated_input is not None:
                tool_input = result.updated_input
        return tool_input, None, ask

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
    ) -> tuple[AgentTool | None, dict[str, Any], str | None, bool]:
        """Look up the tool, validate input, and run pre-hooks.

        Returns ``(tool, updated_input, error_text, needs_approval)``;
        when ``error_text`` is set the call must not execute.
        """
        tool = self._find_tool(name)
        if tool is None:
            return None, tool_input, f"Unknown tool: {name}", False
        valid, err = tool.validate_input(tool_input)
        if not valid:
            logger.warning("Tool '%s' input validation failed: %s", name, err)
            return (
                tool, tool_input,
                f"Error: invalid input for tool '{name}': {err}", False,
            )
        tool_input, deny, ask = self._apply_pre_hooks(name, tool_input)
        if deny is not None:
            return tool, tool_input, f"Error: tool '{name}' denied: {deny}", False
        return tool, tool_input, None, ask or tool.requires_approval

    def _approval_request(
        self, tc: ToolCall, tool_input: dict[str, Any],
    ) -> tuple[Any, str | None]:
        """Invoke the approval callback. Returns (raw result, deny text).

        The raw result may be an awaitable (async callback) — the
        async caller awaits it; the sync caller rejects it. No
        callback configured fails closed.
        """
        if self._approval_callback is None:
            return None, (
                f"Error: tool '{tc.name}' denied: approval required but no "
                "approval callback is configured (Agent.with_approval)"
            )
        request = ApprovalRequest(
            tool_call_id=tc.id, name=tc.name, tool_input=tool_input,
        )
        try:
            return self._approval_callback(request), None
        except Exception as exc:  # fail closed, like pre-hooks
            logger.exception("Approval callback failed for '%s'", tc.name)
            return None, (
                f"Error: tool '{tc.name}' denied: approval callback raised "
                f"{type(exc).__name__}: {exc}"
            )

    @staticmethod
    def _apply_decision(
        tc: ToolCall, tool_input: dict[str, Any], decision: ApprovalDecision,
    ) -> tuple[dict[str, Any], str | None]:
        """Turn an ApprovalDecision into (input, deny text)."""
        if not isinstance(decision, ApprovalDecision):
            return tool_input, (
                f"Error: tool '{tc.name}' denied: approval callback returned "
                f"{type(decision).__name__}, expected ApprovalDecision"
            )
        if not decision.approved:
            reason = decision.reason or "denied by approval callback"
            return tool_input, f"Error: tool '{tc.name}' denied: {reason}"
        if decision.updated_input is not None:
            return decision.updated_input, None
        return tool_input, None

    def _approve(
        self, tc: ToolCall, tool_input: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        """Sync approval resolution: (input, deny text)."""
        raw, deny = self._approval_request(tc, tool_input)
        if deny is not None:
            return tool_input, deny
        if inspect.isawaitable(raw):
            if inspect.iscoroutine(raw):
                raw.close()
            msg = (
                "Async approval callback requires async execution. "
                "Use agent.astream()/agent.achat()."
            )
            raise TypeError(msg)
        return self._apply_decision(tc, tool_input, raw)

    async def _aapprove(
        self, tc: ToolCall, tool_input: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        """Async approval resolution: (input, deny text).

        Deliberately outside the tool timeout: an approval may stay
        pending indefinitely — the turn waits.
        """
        raw, deny = self._approval_request(tc, tool_input)
        if deny is not None:
            return tool_input, deny
        try:
            decision = await raw if inspect.isawaitable(raw) else raw
        except Exception as exc:  # fail closed
            logger.exception("Approval callback failed for '%s'", tc.name)
            return tool_input, (
                f"Error: tool '{tc.name}' denied: approval callback "
                f"raised {type(exc).__name__}: {exc}"
            )
        return self._apply_decision(tc, tool_input, decision)

    @staticmethod
    def _tool_failure_text(name: str, exc: BaseException) -> str:
        return f"Error: tool '{name}' failed: {type(exc).__name__}: {exc}"

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
        tool, tool_input, err, needs_approval = self._resolve_call(
            tc.name, tc.arguments,
        )
        if tool is None or err is not None:
            return self._error_result(tc, tool_input, err or f"Unknown tool: {tc.name}")
        if needs_approval:
            tool_input, deny = self._approve(tc, tool_input)
            if deny is not None:
                return self._error_result(tc, tool_input, deny)
        self._fire("on_tool_start", tc.name, tool_input)
        try:
            # ponytail: no timeout on sync tools — cancelling a sync call
            # needs a worker thread; the async path enforces tool.timeout.
            output = tool.fn(**tool_input)
        except Exception as exc:
            logger.exception("Tool '%s' failed", tc.name)
            return self._error_result(
                tc, tool_input, self._tool_failure_text(tc.name, exc),
            )
        return self._ok_result(tc, tool_input, output)

    async def _aexecute_call(self, tc: ToolCall) -> ToolResult:
        """Execute one tool call asynchronously (MCP/subagent-aware)."""
        tool, tool_input, err, needs_approval = self._resolve_call(
            tc.name, tc.arguments,
        )
        if tool is None or err is not None:
            return self._error_result(tc, tool_input, err or f"Unknown tool: {tc.name}")
        if needs_approval:
            tool_input, deny = await self._aapprove(tc, tool_input)
            if deny is not None:
                return self._error_result(tc, tool_input, deny)
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
            return self._error_result(
                tc, tool_input, self._tool_failure_text(tc.name, exc),
            )
        return self._ok_result(tc, tool_input, output)

    async def _aexecute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """Async tool execution by name; returns the result text."""
        result = await self._aexecute_call(
            ToolCall(id="direct", name=name, arguments=tool_input),
        )
        return result.content

    async def _ensure_mcp(self) -> None:
        """Lazily connect configured MCP servers and load their tools."""
        if not self._mcp_configs or self._mcp_pool is not None:
            return
        from anchor.mcp.client import MCPClientPool
        pool = MCPClientPool(self._mcp_configs)
        try:
            await pool.connect_all()
            mcp_tools = await pool.all_agent_tools()
            # _find_tool is first-match-wins: a silent shadow would be
            # a debugging nightmare — fail loudly at connect instead.
            taken = {t.name for t in self._all_active_tools()}
            for t in mcp_tools:
                if t.name in taken:
                    msg = (
                        f"Tool name collision: MCP tool '{t.name}' shadows an "
                        "existing tool (set prefix_tools=True or rename)"
                    )
                    raise ValueError(msg)
                taken.add(t.name)
            self._mcp_tools = mcp_tools
            self._mcp_pool = pool
        except Exception:
            await pool.disconnect_all()
            raise

    def _require_mcp_pool(self) -> Any:
        if self._mcp_pool is None:
            msg = (
                "No connected MCP servers. Configure with_mcp_servers() and "
                "run a turn (or await agent._ensure_mcp()) first."
            )
            raise RuntimeError(msg)
        return self._mcp_pool

    async def mcp_prompts(self) -> dict[str, list[Any]]:
        """MCP prompt templates per server (app-facing; never automatic)."""
        await self._ensure_mcp()
        return await self._require_mcp_pool().all_prompts()

    async def mcp_get_prompt(
        self, server: str, name: str, arguments: dict[str, Any] | None = None,
    ) -> str:
        """Render an MCP prompt template as message text.

        Reference-client pattern: prompts are user-controlled — the
        application decides to send the result as a user message.
        """
        await self._ensure_mcp()
        return await self._require_mcp_pool().get_prompt(server, name, arguments)

    async def mcp_resources(self) -> dict[str, list[Any]]:
        """MCP resources per server (app-facing; never automatic)."""
        await self._ensure_mcp()
        return await self._require_mcp_pool().all_resources()

    async def mcp_read_resource(self, server: str, uri: str) -> str:
        """Read an MCP resource by URI — application-controlled context."""
        await self._ensure_mcp()
        return await self._require_mcp_pool().read_resource(server, uri)

    def _should_run_tools(self, state: _RoundState, *, final_round: bool) -> bool:
        """Whether this round's tool calls execute.

        A final round that still asked for tools runs none: the model
        would never see their results. Exception: a final round whose
        every call is the terminal ``final_result`` — capturing it ends
        the turn, so the model never needs a result back.
        """
        if state.stop_reason != StopReason.TOOL_USE:
            return False
        if not final_round:
            return True
        if self._output_tool is None or not state.accumulators:
            return False
        return all(
            acc.get("name") == "final_result"
            for acc in state.accumulators.values()
        )

    def _close_round(
        self,
        round_index: int,
        state: _RoundState,
        schema_tokens: int,
        tool_results: list[ToolResult],
        llm_messages: list[Message],
        *,
        run_tools: bool,
        wrap_up: bool,
        stopped_by: str,
    ) -> tuple[RoundUsage, str]:
        """Account the round and fire ``on_round_end``.

        Returns ``(usage, stopped_by)``; ``stopped_by`` changes only
        when the round ends the turn: ``usage_limit`` after a wrap-up
        round, otherwise the model's own stop cause.
        """
        if not run_tools:
            if wrap_up:
                stopped_by = "usage_limit"
            elif state.stop_reason != StopReason.TOOL_USE:
                stopped_by = self._stop_cause(state.stop_reason)
        usage = self._round_usage(
            round_index, state, schema_tokens, tool_results, llm_messages,
        )
        self._fire("on_round_end", round_index)
        return usage, stopped_by

    def _stream_tool_phase(
        self,
        state: _RoundState,
        messages: list[Message],
        tool_results: list[ToolResult],
    ) -> Iterator[AgentEvent]:
        """Append the assistant turn, run its tools, append the results."""
        tool_calls = self._build_tool_calls(state.accumulators)
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=state.text or None,
                tool_calls=tool_calls,
                raw_content=state.raw_blocks or None,
            ),
        )
        yield from self._stream_tools(tool_calls, tool_results)
        for result in tool_results:
            messages.append(Message(role=Role.TOOL, tool_result=result))

    async def _astream_tool_phase(
        self,
        state: _RoundState,
        messages: list[Message],
        tool_results: list[ToolResult],
    ) -> AsyncGenerator[AgentEvent, None]:
        """Async mirror of :meth:`_stream_tool_phase`."""
        tool_calls = self._build_tool_calls(state.accumulators)
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=state.text or None,
                tool_calls=tool_calls,
                raw_content=state.raw_blocks or None,
            ),
        )
        async for event in self._astream_tools(tool_calls, tool_results):
            yield event
        for result in tool_results:
            messages.append(Message(role=Role.TOOL, tool_result=result))

    @staticmethod
    def _tool_finished(tc: ToolCall, result: ToolResult) -> ToolFinished:
        return ToolFinished(
            tool_call_id=tc.id,
            name=tc.name,
            result=result.content,
            is_error=result.is_error,
        )

    def _stream_tools(
        self, tool_calls: list[ToolCall], results: list[ToolResult],
    ) -> Iterator[AgentEvent]:
        """Execute calls sequentially, yielding started/finished events.

        Completed ``ToolResult`` objects are appended to *results*.
        Events a subagent forwards during the call are buffered and
        yielded before its ``ToolFinished`` (a sync consumer is blocked
        while the tool runs, so live delivery is impossible anyway).
        """
        for tc in tool_calls:
            yield ToolStarted(
                tool_call_id=tc.id, name=tc.name, tool_input=tc.arguments,
            )
            forwarded: list[AgentEvent] = []
            token = _EVENT_SINK.set((forwarded.append, tc.id))
            try:
                result = self._execute_call(tc)
            finally:
                _EVENT_SINK.reset(token)
            results.append(result)
            yield from forwarded
            yield self._tool_finished(tc, result)

    async def _astream_tools(
        self, tool_calls: list[ToolCall], results: list[ToolResult],
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute calls concurrently, yielding finished events live.

        All ``ToolStarted`` events are emitted up front (the calls do
        start together); each ``ToolFinished`` is emitted as its call
        completes, interleaved with events subagents forward through
        the per-task event sink. *results* receives the ToolResults in
        call order. An exception escaping a call is converted through
        ``_error_result`` (firing ``on_tool_error``); cancellation
        propagates and pending calls are cancelled.
        """
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

        async def _one(index: int, tc: ToolCall) -> tuple[int, ToolResult]:
            # Each task owns a context copy, so the sink is per-call.
            _EVENT_SINK.set((queue.put_nowait, tc.id))
            try:
                return index, await self._aexecute_call(tc)
            except Exception as exc:
                logger.exception("Tool '%s' failed", tc.name)
                return index, self._error_result(
                    tc, tc.arguments, self._tool_failure_text(tc.name, exc),
                )

        for tc in tool_calls:
            yield ToolStarted(
                tool_call_id=tc.id, name=tc.name, tool_input=tc.arguments,
            )
        tasks = [
            asyncio.ensure_future(_one(i, tc)) for i, tc in enumerate(tool_calls)
        ]
        ordered: list[ToolResult | None] = [None] * len(tool_calls)
        async for event in self._merge_tool_events(
            tool_calls, tasks, queue, ordered,
        ):
            yield event
        results.extend(r for r in ordered if r is not None)

    async def _merge_tool_events(
        self,
        tool_calls: list[ToolCall],
        tasks: list[asyncio.Task[tuple[int, ToolResult]]],
        queue: asyncio.Queue[AgentEvent],
        ordered: list[ToolResult | None],
    ) -> AsyncGenerator[AgentEvent, None]:
        """Interleave forwarded child events with tool completions.

        Waits on the running tool tasks and the event queue at once, so
        a subagent's events surface while sibling calls are still
        running. Cancels everything still pending on the way out.
        """
        pending: set[asyncio.Task[tuple[int, ToolResult]]] = set(tasks)
        getter: asyncio.Task[AgentEvent] | None = None
        try:
            while pending:
                if getter is None:
                    getter = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait(
                    {*pending, getter}, return_when=asyncio.FIRST_COMPLETED,
                )
                if getter.done():
                    yield getter.result()
                    getter = None
                    while not queue.empty():
                        yield queue.get_nowait()
                for task in done & pending:
                    pending.discard(task)
                    index, result = task.result()
                    ordered[index] = result
                    # A call's forwarded events precede its ToolFinished.
                    while not queue.empty():
                        yield queue.get_nowait()
                    yield self._tool_finished(tool_calls[index], result)
        finally:
            leftover = [t for t in tasks if not t.done()]
            if getter is not None:
                leftover.append(getter)
            for fut in leftover:
                fut.cancel()
            if leftover:
                # Await the cancellations so in-flight tool calls (MCP
                # sessions, subagent turns) finish their cleanup before
                # aclose() returns — no orphaned pending tasks.
                await asyncio.gather(*leftover, return_exceptions=True)

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
        self,
        full_system: str,
        messages: list[Message],
        *,
        final_round: bool = False,
    ) -> tuple[list[Message], list[ToolSchema] | None, int, dict[str, Any]]:
        """Assemble one round's request.

        Returns ``(messages, schemas, schema_tokens, call_extra)`` where
        ``call_extra`` carries per-call provider options (tool_choice,
        context_management).
        """
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
        call_extra: dict[str, Any] = {}
        if self._context_management:
            call_extra["context_management"] = self._context_management
        if final_round and schemas:
            if self._output_tool is not None:
                # A final round with structured output pending must
                # still record the answer — force the output tool.
                call_extra["tool_choice"] = {
                    "type": "tool", "name": "final_result",
                }
            else:
                # Belt over the final-round notice: providers that
                # support tool_choice force a text answer; others just
                # ignore it.
                call_extra["tool_choice"] = "none"
        elif self._output_tool is not None and schemas:
            # Structured output: the model may not stop in plain text —
            # it must call a tool, ultimately final_result.
            call_extra["tool_choice"] = "any"
        return llm_messages, schemas, schema_tokens, call_extra

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
            state.cache_creation_tokens = max(
                state.cache_creation_tokens, chunk.usage.cache_creation_tokens,
            )
            state.cache_read_tokens = max(
                state.cache_read_tokens, chunk.usage.cache_read_tokens,
            )
        if chunk.raw_block:
            state.raw_blocks.append(chunk.raw_block)
        if chunk.stop_reason:
            state.stop_reason = chunk.stop_reason
        return out

    def _stream_model_round(
        self,
        llm_messages: list[Message],
        schemas: list[ToolSchema] | None,
        call_extra: dict[str, Any],
        state: _RoundState,
    ) -> Iterator[TextDelta]:
        """Stream one model call, folding chunks into *state*."""
        for chunk in self._llm.stream(
            llm_messages,
            tools=schemas,
            max_tokens=self._max_response_tokens,
            **call_extra,
        ):
            out = self._ingest_chunk(state, chunk)
            if out:
                yield TextDelta(text=out)

    async def _astream_model_round(
        self,
        llm_messages: list[Message],
        schemas: list[ToolSchema] | None,
        call_extra: dict[str, Any],
        state: _RoundState,
    ) -> AsyncGenerator[TextDelta, None]:
        """Async mirror of :meth:`_stream_model_round`."""
        async for chunk in self._llm.astream(
            llm_messages,
            tools=schemas,
            max_tokens=self._max_response_tokens,
            **call_extra,
        ):
            out = self._ingest_chunk(state, chunk)
            if out:
                yield TextDelta(text=out)

    def _round_usage(
        self,
        index: int,
        state: _RoundState,
        schema_tokens: int,
        tool_results: list[ToolResult],
        llm_messages: list[Message],
    ) -> RoundUsage:
        result_tokens = sum(
            self._tokenizer.count_tokens(r.content) for r in tool_results
        )
        # Providers that report no usage on the stream (all but
        # Anthropic today) get tokenizer estimates, so usage limits and
        # accounting hold across providers.
        prompt = state.prompt_tokens
        if prompt == 0:
            prompt = self._messages_tokens(llm_messages) + schema_tokens
        completion = state.completion_tokens
        if completion == 0:
            completion = self._tokenizer.count_tokens(state.text) + sum(
                self._tokenizer.count_tokens(acc["args_json"])
                for acc in state.accumulators.values()
            )
        return RoundUsage(
            round=index,
            prompt_tokens=prompt,
            completion_tokens=completion,
            tool_schema_tokens=schema_tokens,
            tool_result_tokens=result_tokens,
            cache_creation_tokens=state.cache_creation_tokens,
            cache_read_tokens=state.cache_read_tokens,
            tool_calls=len(tool_results),
        )

    def _is_final_round(self, round_index: int) -> bool:
        return round_index == self._max_rounds - 1

    def _maybe_final_round_notice(
        self, final_round: bool, messages: list[Message],
    ) -> None:
        """Warn the model its next answer must be the last so it wraps up.

        Only meaningful when tools exist — a tool-less agent has no
        tool budget to exhaust (relevant for ``max_rounds=1``).
        """
        if final_round and self._all_active_tools():
            notice = (
                _FINAL_ROUND_NOTICE_OUTPUT
                if self._output_tool is not None
                else _FINAL_ROUND_NOTICE
            )
            messages.append(Message(role=Role.USER, content=notice))

    def _check_usage_limits(
        self, rounds: list[RoundUsage],
    ) -> UsageLimitReached | None:
        """Return a breach event when a per-turn limit is crossed."""
        limits = self._usage_limits
        if limits is None:
            return None
        if limits.total_tokens_limit is not None:
            used = sum(r.total_tokens for r in rounds)
            if used > limits.total_tokens_limit:
                return UsageLimitReached(
                    kind="total_tokens",
                    used=used,
                    limit=limits.total_tokens_limit,
                )
        if limits.tool_calls_limit is not None:
            calls = sum(r.tool_calls for r in rounds)
            if calls > limits.tool_calls_limit:
                return UsageLimitReached(
                    kind="tool_calls", used=calls, limit=limits.tool_calls_limit,
                )
        return None

    # -- Client-side compaction (provider-agnostic) --

    def _message_text(self, msg: Message) -> str:
        parts: list[str] = []
        if isinstance(msg.content, str):
            parts.append(msg.content)
        if msg.tool_calls:
            parts.extend(json.dumps(tc.arguments) for tc in msg.tool_calls)
        if msg.tool_result is not None:
            parts.append(msg.tool_result.content)
        return "\n".join(parts)

    def _messages_tokens(self, messages: list[Message]) -> int:
        return sum(
            self._tokenizer.count_tokens(self._message_text(m)) for m in messages
        )

    def _split_for_compaction(
        self, messages: list[Message],
    ) -> tuple[list[Message], list[Message]] | None:
        """Head to summarize / tail to keep, or None when not needed."""
        if self._compaction_trigger is None:
            return None
        if self._messages_tokens(messages) <= self._compaction_trigger:
            return None
        split = len(messages) - self._compaction_keep
        # Never sever a tool_use/tool_result pair: pull the split left
        # so the kept tail starts at a non-TOOL message.
        while split > 0 and messages[split].role == Role.TOOL:
            split -= 1
        if split <= 0:
            return None
        head = list(messages[:split])
        # Summarizing a head smaller than the summary target shrinks
        # nothing (and would re-summarize its own summary every round).
        if self._messages_tokens(head) <= self._compaction_target_tokens():
            return None
        return head, list(messages[split:])

    def _compaction_transcript(self, head: list[Message]) -> str:
        return "\n".join(
            f"{m.role.value}: {self._message_text(m)}" for m in head
        )

    def _compaction_target_tokens(self) -> int:
        return max(64, (self._compaction_trigger or 1024) // 4)

    @staticmethod
    def _apply_compaction(
        messages: list[Message], summary: str, tail: list[Message],
    ) -> None:
        messages[:] = [
            Message(role=Role.USER, content=f"[Conversation summary]\n{summary}"),
            *tail,
        ]

    def _stream_compact(self, messages: list[Message]) -> Iterator[AgentEvent]:
        split = self._split_for_compaction(messages)
        if split is None:
            return
        yield CompactionStarted()
        tokens_before = self._messages_tokens(messages)
        head, tail = split
        transcript = self._compaction_transcript(head)
        if self._compact_fn is not None:
            summary = self._compact_fn(transcript)
        else:
            from anchor.memory.compactor import TierCompactor
            summary = TierCompactor(self._llm, tokenizer=self._tokenizer).summarize(
                transcript, 1, self._compaction_target_tokens(),
            )
        self._apply_compaction(messages, summary, tail)
        yield CompactionFinished(
            tokens_before=tokens_before,
            tokens_after=self._messages_tokens(messages),
        )

    async def _astream_compact(
        self, messages: list[Message],
    ) -> AsyncGenerator[AgentEvent, None]:
        split = self._split_for_compaction(messages)
        if split is None:
            return
        yield CompactionStarted()
        tokens_before = self._messages_tokens(messages)
        head, tail = split
        transcript = self._compaction_transcript(head)
        if self._compact_fn is not None:
            summary = self._compact_fn(transcript)
        else:
            from anchor.memory.compactor import TierCompactor
            compactor = TierCompactor(self._llm, tokenizer=self._tokenizer)
            summary = await compactor.asummarize(
                transcript, 1, self._compaction_target_tokens(),
            )
        self._apply_compaction(messages, summary, tail)
        yield CompactionFinished(
            tokens_before=tokens_before,
            tokens_after=self._messages_tokens(messages),
        )

    def _finish_turn(
        self,
        rounds: list[RoundUsage],
        stopped_by: str,
        final_text: str,
    ) -> TurnDiagnostics:
        self._last_turn = TurnDiagnostics(
            rounds=tuple(rounds),
            stopped_by=stopped_by,  # type: ignore[arg-type]
        )
        if self._memory is not None and final_text:
            self._memory.add_assistant_message(final_text)
        return self._last_turn

    @staticmethod
    def _stop_cause(stop_reason: StopReason | None) -> str:
        return "max_tokens" if stop_reason == StopReason.MAX_TOKENS else "stop"

    def _round_verdict(
        self,
        messages: list[Message],
        *,
        run_tools: bool,
        stopped_by: str,
    ) -> tuple[bool, str]:
        """Post-round control: ``(stop_loop, stopped_by)``.

        Structured output (tool mode) hooks in here: a captured
        ``final_result`` stops the loop; a model that stopped in plain
        text with output still pending gets a retry nudge appended to
        *messages*; exhausted retries raise ``ValueError``. Output
        captured during a usage-limit wrap-up keeps
        ``stopped_by="usage_limit"`` — the budget cut stays visible.
        """
        if self._last_output is not None:
            if stopped_by == "usage_limit":
                return True, "usage_limit"
            return True, "stop"
        output_pending = (
            self._output_model is not None and self._output_mode == "tool"
        )
        if output_pending and self._output_failures > self._max_output_retries:
            msg = (
                "structured output failed validation after "
                f"{self._max_output_retries} retries"
            )
            raise ValueError(msg)
        if not run_tools:
            if output_pending and stopped_by not in ("usage_limit",):
                self._output_failures += 1
                if self._output_failures > self._max_output_retries:
                    msg = (
                        "model ended the turn without calling final_result "
                        f"after {self._max_output_retries} retries"
                    )
                    raise ValueError(msg)
                messages.append(Message(
                    role=Role.USER,
                    content=(
                        "[system] You must record your final answer by "
                        "calling the final_result tool — do not answer in "
                        "plain text."
                    ),
                ))
                return False, stopped_by
            return True, stopped_by
        return False, stopped_by

    # -- Chat --

    def stream(self, message: str) -> Iterator[AgentEvent]:
        """Send a message and stream typed agent events.

        The full tool-use loop as one ordered event stream:
        ``TurnStarted``, per-round ``RoundStarted``/``RoundFinished``
        (the latter with :class:`RoundUsage`), ``TextDelta``,
        ``ToolStarted``/``ToolFinished`` (correlated by
        ``tool_call_id``), compaction events, and a terminal
        ``TurnFinished`` with the final text and diagnostics.
        :meth:`chat` is the text-only projection of this stream.
        """
        if self._mcp_configs:
            msg = (
                "MCP servers require async execution. "
                "Use agent.astream()/agent.achat() instead."
            )
            raise TypeError(msg)
        if self._memory is not None:
            self._memory.add_user_message(message)

        self._last_turn = None
        self._last_output = None
        self._output_failures = 0
        messages, full_system = self._prepare_turn(
            self._pipeline.build(message), message,
        )
        final_text = ""
        rounds: list[RoundUsage] = []
        stopped_by = "max_rounds"
        round_open: int | None = None
        wrap_up = False
        try:
            yield TurnStarted()
            for round_index in range(self._max_rounds):
                final_round = wrap_up or self._is_final_round(round_index)
                self._fire("on_round_start", round_index)
                round_open = round_index
                yield RoundStarted(round=round_index, max_rounds=self._max_rounds)
                yield from self._stream_compact(messages)
                self._maybe_final_round_notice(final_round, messages)
                llm_messages, schemas, schema_tokens, call_extra = (
                    self._round_request(
                        full_system, messages, final_round=final_round,
                    )
                )
                state = _RoundState()

                for delta in self._stream_model_round(
                    llm_messages, schemas, call_extra, state,
                ):
                    final_text += delta.text
                    yield delta

                run_tools = self._should_run_tools(state, final_round=final_round)
                tool_results: list[ToolResult] = []
                if run_tools:
                    yield from self._stream_tool_phase(
                        state, messages, tool_results,
                    )
                usage, stopped_by = self._close_round(
                    round_index, state, schema_tokens, tool_results,
                    llm_messages,
                    run_tools=run_tools, wrap_up=wrap_up, stopped_by=stopped_by,
                )
                round_open = None
                rounds.append(usage)
                yield RoundFinished(round=round_index, usage=usage)
                stop_loop, stopped_by = self._round_verdict(
                    messages, run_tools=run_tools, stopped_by=stopped_by,
                )
                if stop_loop:
                    break
                breach = self._check_usage_limits(rounds)
                if breach is not None:
                    # Set eagerly so an abandoned wrap-up still persists
                    # the true cause; the wrap-up's _close_round
                    # re-derives the same value on the happy path.
                    wrap_up = True
                    stopped_by = "usage_limit"
                    yield breach
        finally:
            # Runs exactly once on every path. On abandonment or a
            # mid-turn exception this closes the open round and persists
            # diagnostics plus the partial text the consumer already saw.
            if round_open is not None:
                self._fire("on_round_end", round_open)
            diagnostics = self._finish_turn(rounds, stopped_by, final_text)
        yield TurnFinished(
            text=final_text, diagnostics=diagnostics, output=self._last_output,
        )

    async def astream(self, message: str) -> AsyncGenerator[AgentEvent, None]:
        """Send a message and stream typed agent events asynchronously.

        Async mirror of :meth:`stream`. Uses ``pipeline.abuild()``,
        async iteration over the streaming API, and executes
        independent tool calls concurrently — ``ToolFinished`` events
        are emitted live as each call completes. :meth:`achat` is the
        text-only projection of this stream.
        """
        if self._memory is not None:
            self._memory.add_user_message(message)
        await self._ensure_mcp()

        self._last_turn = None
        self._last_output = None
        self._output_failures = 0
        messages, full_system = self._prepare_turn(
            await self._pipeline.abuild(message), message,
        )
        final_text = ""
        rounds: list[RoundUsage] = []
        stopped_by = "max_rounds"
        round_open: int | None = None
        wrap_up = False
        try:
            yield TurnStarted()
            for round_index in range(self._max_rounds):
                final_round = wrap_up or self._is_final_round(round_index)
                self._fire("on_round_start", round_index)
                round_open = round_index
                yield RoundStarted(round=round_index, max_rounds=self._max_rounds)
                async for event in self._astream_compact(messages):
                    yield event
                self._maybe_final_round_notice(final_round, messages)
                llm_messages, schemas, schema_tokens, call_extra = (
                    self._round_request(
                        full_system, messages, final_round=final_round,
                    )
                )
                state = _RoundState()

                async for delta in self._astream_model_round(
                    llm_messages, schemas, call_extra, state,
                ):
                    final_text += delta.text
                    yield delta

                run_tools = self._should_run_tools(state, final_round=final_round)
                tool_results: list[ToolResult] = []
                if run_tools:
                    async for event in self._astream_tool_phase(
                        state, messages, tool_results,
                    ):
                        yield event
                usage, stopped_by = self._close_round(
                    round_index, state, schema_tokens, tool_results,
                    llm_messages,
                    run_tools=run_tools, wrap_up=wrap_up, stopped_by=stopped_by,
                )
                round_open = None
                rounds.append(usage)
                yield RoundFinished(round=round_index, usage=usage)
                stop_loop, stopped_by = self._round_verdict(
                    messages, run_tools=run_tools, stopped_by=stopped_by,
                )
                if stop_loop:
                    break
                breach = self._check_usage_limits(rounds)
                if breach is not None:
                    # Set eagerly so an abandoned wrap-up still persists
                    # the true cause; the wrap-up's _close_round
                    # re-derives the same value on the happy path.
                    wrap_up = True
                    stopped_by = "usage_limit"
                    yield breach
        finally:
            # Runs exactly once on every path. On abandonment or a
            # mid-turn exception this closes the open round and persists
            # diagnostics plus the partial text the consumer already saw.
            if round_open is not None:
                self._fire("on_round_end", round_open)
            diagnostics = self._finish_turn(rounds, stopped_by, final_text)
        yield TurnFinished(
            text=final_text, diagnostics=diagnostics, output=self._last_output,
        )

    def chat(self, message: str) -> Iterator[str]:
        """Send a message and stream the response text.

        Text-only projection of :meth:`stream` — the full tool-use
        loop runs identically; only top-level ``TextDelta`` events
        surface, as plain strings.
        """
        for event in self.stream(message):
            if isinstance(event, TextDelta) and event.parent_tool_call_id is None:
                yield event.text

    async def achat(self, message: str) -> AsyncGenerator[str, None]:
        """Send a message and stream the response text asynchronously.

        Text-only projection of :meth:`astream`.
        """
        async for event in self.astream(message):
            if isinstance(event, TextDelta) and event.parent_tool_call_id is None:
                yield event.text

    def _require_output_model(self) -> type[BaseModel]:
        if self._output_model is None:
            msg = "run() requires with_output_model()"
            raise ValueError(msg)
        if self._output_mode == "prompted" and self._memory is not None:
            # Same rationale as the subagent clean-context guard: the
            # prompted mechanic persists the schema, invalid replies,
            # and retry prompts as real conversation turns.
            msg = (
                "prompted output mode requires an agent without memory "
                "(use tool mode, or drop with_memory)"
            )
            raise ValueError(msg)
        return self._output_model

    def run(self, message: str) -> BaseModel:
        """Run a turn and return the validated structured output.

        Requires :meth:`with_output_model`. Sync mirror of :meth:`arun`.
        """
        output_model = self._require_output_model()
        if self._output_mode == "prompted":
            normalized = _run_sync(
                self, message, output_model, self._max_output_retries,
            )
            return output_model.model_validate_json(normalized)
        for _ in self.stream(message):
            pass
        if self._last_output is None:
            msg = "turn ended without structured output"
            raise ValueError(msg)
        return output_model.model_validate_json(self._last_output)

    async def arun(self, message: str) -> BaseModel:
        """Async mirror of :meth:`run`."""
        output_model = self._require_output_model()
        if self._output_mode == "prompted":
            normalized = await _run_async(
                self, message, output_model, self._max_output_retries,
            )
            return output_model.model_validate_json(normalized)
        async for _ in self.astream(message):
            pass
        if self._last_output is None:
            msg = "turn ended without structured output"
            raise ValueError(msg)
        return output_model.model_validate_json(self._last_output)

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
