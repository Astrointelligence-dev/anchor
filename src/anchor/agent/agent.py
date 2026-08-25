"""Agent class that wraps ContextPipeline + LLMProvider + tool loop."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from pathlib import Path
from typing import Any

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
)
from anchor.llm.registry import create_provider
from anchor.memory.manager import MemoryManager
from anchor.models.context import ContextResult
from anchor.pipeline.pipeline import ContextPipeline

from .models import AgentTool
from .skills.activate import _make_activate_skill_tool
from .skills.models import Skill
from .skills.registry import SkillRegistry
from .skills.resources import _make_read_skill_file_tool, _make_run_skill_script_tool

logger = logging.getLogger(__name__)

# Maximum character length for tool input/result recorded in memory.
_TOOL_MEMORY_TRUNCATE = 200


class _WhitespaceTokenizer:
    """Minimal tokenizer that counts whitespace-separated words.

    Used internally by Agent to avoid tiktoken's network dependency.
    """

    __slots__ = ()

    def count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


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
        "_allow_skill_scripts",
        "_last_result",
        "_llm",
        "_max_response_tokens",
        "_max_rounds",
        "_mcp_configs",
        "_mcp_pool",
        "_mcp_tools",
        "_memory",
        "_pipeline",
        "_read_file_tool",
        "_script_tool",
        "_skill_registry",
        "_system_prompt",
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
        self._skill_registry = SkillRegistry()
        self._activate_tool: AgentTool | None = None
        self._read_file_tool: AgentTool | None = None
        self._script_tool: AgentTool | None = None
        self._allow_skill_scripts = allow_skill_scripts
        self._mcp_configs: list[Any] = []  # MCPServerConfig instances
        self._mcp_pool: Any = None  # MCPClientPool (lazy)
        self._mcp_tools: list[AgentTool] = []

        tokenizer = _WhitespaceTokenizer()
        self._pipeline = ContextPipeline(max_tokens=max_tokens, tokenizer=tokenizer)
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

    # -- Internal helpers --

    def _all_active_tools(self) -> list[AgentTool]:
        """Return direct tools + skill tools + skill meta-tools + MCP tools."""
        tools: list[AgentTool] = list(self._tools)
        tools.extend(self._skill_registry.active_tools())
        if self._activate_tool is not None:
            tools.append(self._activate_tool)
        if self._read_file_tool is not None:
            tools.append(self._read_file_tool)
        if self._script_tool is not None:
            tools.append(self._script_tool)
        tools.extend(self._mcp_tools)
        return tools

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

    def _skill_system_suffix(self) -> str:
        """Static skill text appended to the system prompt each turn.

        Contains always-skill instructions plus the on-demand discovery
        listing. Built once per turn (not per round) and stable across
        activation state — prompt-cache friendly. The listing is capped at
        roughly 1% of the pipeline's token budget.
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
        return "\n\n".join(parts)

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """Look up and execute a tool by name.

        Validates input against the tool's schema before calling.
        Searches direct tools, active skill tools, and the meta-tool.
        """
        for tool in self._all_active_tools():
            if tool.name == name:
                valid, err = tool.validate_input(tool_input)
                if not valid:
                    logger.warning(
                        "Tool '%s' input validation failed: %s", name, err,
                    )
                    return f"Error: invalid input for tool '{name}': {err}"
                try:
                    return tool.fn(**tool_input)
                except Exception:
                    logger.exception("Tool '%s' failed", name)
                    return f"Error: tool '{name}' failed."
        return f"Unknown tool: {name}"

    async def _aexecute_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        """Async tool execution — supports both regular and MCP tools."""
        for tool in self._all_active_tools():
            if tool.name == name:
                # Check if this is an MCP tool with async caller
                async_caller = getattr(tool, "_mcp_async_caller", None)
                if async_caller is not None:
                    original_name = getattr(tool, "_mcp_original_name", name)
                    # Validate input before calling MCP tool
                    valid, err = tool.validate_input(tool_input)
                    if not valid:
                        return f"Error: invalid input for tool '{name}': {err}"
                    try:
                        return await async_caller(original_name, tool_input)
                    except Exception as exc:
                        logger.exception("MCP tool '%s' failed", name)
                        return f"Error: MCP tool '{name}' failed: {exc}"

                # Regular tool — run sync
                valid, err = tool.validate_input(tool_input)
                if not valid:
                    return f"Error: invalid input for tool '{name}': {err}"
                try:
                    return tool.fn(**tool_input)
                except Exception:
                    logger.exception("Tool '%s' failed", name)
                    return f"Error: tool '{name}' failed."
        return f"Unknown tool: {name}"

    async def _arun_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls asynchronously."""
        results: list[ToolResult] = []
        for tc in tool_calls:
            result_text = await self._aexecute_tool(tc.name, tc.arguments)
            self._record_tool_call(tc.name, tc.arguments, result_text)
            results.append(ToolResult(tool_call_id=tc.id, content=result_text))
        return results

    def _run_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tool calls and return ToolResult objects."""
        results: list[ToolResult] = []
        for tc in tool_calls:
            result_text = self._execute_tool(tc.name, tc.arguments)
            self._record_tool_call(tc.name, tc.arguments, result_text)
            results.append(ToolResult(tool_call_id=tc.id, content=result_text))
        return results

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

        ctx_result = self._pipeline.build(message)
        self._last_result = ctx_result
        formatted = ctx_result.formatted_output
        if not isinstance(formatted, dict):
            msg = "Agent requires a dict-based formatter output"
            raise TypeError(msg)

        base_messages, base_system = self._formatted_to_messages(formatted)

        # Skill text (always-skill instructions + discovery listing) is static
        # per turn: compute once, join with newlines, keep the prompt stable
        # across rounds so provider prompt caching can hit.
        skill_suffix = self._skill_system_suffix()
        if base_system and skill_suffix:
            full_system = f"{base_system}\n\n{skill_suffix}"
        else:
            full_system = base_system or skill_suffix

        # Working message list for the tool loop
        messages: list[Message] = list(base_messages)

        final_text = ""

        for _round in range(self._max_rounds):
            # Per-round tool recomputation (skills may be activated mid-convo)
            all_tools = self._all_active_tools()
            tool_schemas = (
                [t.to_tool_schema() for t in all_tools] if all_tools else None
            )

            # Build messages with system message prepended
            llm_messages: list[Message] = []
            if full_system:
                llm_messages.append(Message(role=Role.SYSTEM, content=full_system))
            llm_messages.extend(messages)

            # Stream from provider
            accumulated_text = ""
            tool_call_accumulators: dict[int, dict[str, Any]] = {}
            stop_reason: StopReason | None = None

            for chunk in self._llm.stream(
                llm_messages,
                tools=tool_schemas,
                max_tokens=self._max_response_tokens,
            ):
                if chunk.content:
                    accumulated_text += chunk.content
                    final_text += chunk.content
                    yield chunk.content
                if chunk.tool_call_delta:
                    delta = chunk.tool_call_delta
                    acc = tool_call_accumulators.setdefault(
                        delta.index, {"id": None, "name": None, "args_json": ""},
                    )
                    if delta.id:
                        acc["id"] = delta.id
                    if delta.name:
                        acc["name"] = delta.name
                    if delta.arguments_fragment:
                        acc["args_json"] += delta.arguments_fragment
                if chunk.stop_reason:
                    stop_reason = chunk.stop_reason

            if stop_reason != StopReason.TOOL_USE:
                break

            # Build tool calls from accumulated deltas
            tool_calls = self._build_tool_calls(tool_call_accumulators)

            # Add assistant message with tool calls to conversation
            messages.append(Message(
                role=Role.ASSISTANT,
                content=accumulated_text or None,
                tool_calls=tool_calls,
            ))

            # Execute tools and add results
            tool_results = self._run_tools(tool_calls)
            for result in tool_results:
                messages.append(Message(role=Role.TOOL, tool_result=result))

        if self._memory is not None and final_text:
            self._memory.add_assistant_message(final_text)

    async def achat(self, message: str) -> AsyncGenerator[str, None]:
        """Send a message and stream the response asynchronously.

        Async mirror of :meth:`chat`. Uses ``pipeline.abuild()`` and
        async iteration over the streaming API.

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

        ctx_result = await self._pipeline.abuild(message)
        self._last_result = ctx_result
        formatted = ctx_result.formatted_output
        if not isinstance(formatted, dict):
            msg = "Agent requires a dict-based formatter output"
            raise TypeError(msg)

        base_messages, base_system = self._formatted_to_messages(formatted)

        # Skill text (always-skill instructions + discovery listing) is static
        # per turn: compute once, join with newlines, keep the prompt stable
        # across rounds so provider prompt caching can hit.
        skill_suffix = self._skill_system_suffix()
        if base_system and skill_suffix:
            full_system = f"{base_system}\n\n{skill_suffix}"
        else:
            full_system = base_system or skill_suffix

        # Working message list for the tool loop
        messages: list[Message] = list(base_messages)

        final_text = ""

        for _round in range(self._max_rounds):
            # Per-round tool recomputation (skills may be activated mid-convo)
            all_tools = self._all_active_tools()
            tool_schemas = (
                [t.to_tool_schema() for t in all_tools] if all_tools else None
            )

            # Build messages with system message prepended
            llm_messages: list[Message] = []
            if full_system:
                llm_messages.append(Message(role=Role.SYSTEM, content=full_system))
            llm_messages.extend(messages)

            # Stream from provider
            accumulated_text = ""
            tool_call_accumulators: dict[int, dict[str, Any]] = {}
            stop_reason: StopReason | None = None

            async for chunk in self._llm.astream(
                llm_messages,
                tools=tool_schemas,
                max_tokens=self._max_response_tokens,
            ):
                if chunk.content:
                    accumulated_text += chunk.content
                    final_text += chunk.content
                    yield chunk.content
                if chunk.tool_call_delta:
                    delta = chunk.tool_call_delta
                    acc = tool_call_accumulators.setdefault(
                        delta.index, {"id": None, "name": None, "args_json": ""},
                    )
                    if delta.id:
                        acc["id"] = delta.id
                    if delta.name:
                        acc["name"] = delta.name
                    if delta.arguments_fragment:
                        acc["args_json"] += delta.arguments_fragment
                if chunk.stop_reason:
                    stop_reason = chunk.stop_reason

            if stop_reason != StopReason.TOOL_USE:
                break

            # Build tool calls from accumulated deltas
            tool_calls = self._build_tool_calls(tool_call_accumulators)

            # Add assistant message with tool calls to conversation
            messages.append(Message(
                role=Role.ASSISTANT,
                content=accumulated_text or None,
                tool_calls=tool_calls,
            ))

            # Execute tools and add results
            tool_results = await self._arun_tools(tool_calls)
            for result in tool_results:
                messages.append(Message(role=Role.TOOL, tool_result=result))

        if self._memory is not None and final_text:
            self._memory.add_assistant_message(final_text)

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
