"""Subagents: agent-as-tool with context isolation and condensed returns.

Implements MULTI_AGENT.md:

- **Rule 1 — clean context.** A subagent receives only the ``task``
  string the orchestrator writes; nothing of the parent conversation.
- **Rule 3 — asymmetric returns.** With an ``output_model``, the
  subagent must answer with schema-valid JSON; invalid output gets one
  self-contained retry carrying the validation error.
- **Rule 5 — no nesting.** Wrapping an agent that itself has subagent
  tools raises at registration time.

Two layers over the same runner: :func:`_make_subagent_tool` (the
``Agent.as_tool()`` primitive — one tool per subagent) and
:func:`_make_task_tool` (the declarative registry's single ``task``
meta-tool dispatching by name, mirroring ``activate_skill``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError

from anchor._text import strip_markdown_fences
from anchor.agent.events import TurnFinished, _forward
from anchor.agent.models import _USAGE_POOL, AgentTool, UsageLimits
from anchor.agent.tool_decorator import tool

if TYPE_CHECKING:
    from anchor.agent.agent import Agent

_SUBAGENT_MARKER = "_anchor_subagent"


class SubagentDefinition(BaseModel):
    """Declarative description of a subagent for ``Agent.with_subagents``."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str
    system_prompt: str = ""
    model: str | None = None  # None inherits the orchestrator's provider
    tools: tuple[AgentTool, ...] = ()
    output_model: type[BaseModel] | None = None
    max_rounds: int = 6
    # Narrower per-turn limits for this child; the run's shared pool
    # still applies on top — the effective budget only ever narrows.
    usage_limits: UsageLimits | None = None


def _is_subagent_tool(t: AgentTool) -> bool:
    return bool(getattr(t, _SUBAGENT_MARKER, False))


def _guard_no_nesting(sub: Agent) -> None:
    """MULTI_AGENT.md rule 5: subagents do not spawn subagents."""
    if any(_is_subagent_tool(t) for t in sub._all_active_tools()):
        msg = (
            "Subagents cannot have their own subagents (MULTI_AGENT.md: "
            "no nesting). Split the work into sibling subagents on the "
            "orchestrator instead."
        )
        raise ValueError(msg)


def _guard_clean_context(sub: Agent) -> None:
    """MULTI_AGENT.md rule 1: each spawn starts with a clean context.

    Attached memory would leak one task's conversation into the next
    (and interleave writes under parallel dispatch).
    """
    if sub.memory is not None:
        msg = (
            "Subagents cannot have memory attached (MULTI_AGENT.md: clean "
            "context per spawn). Wrap an agent without with_memory()."
        )
        raise ValueError(msg)


def _schema_instruction(output_model: type[BaseModel]) -> str:
    schema = json.dumps(output_model.model_json_schema())
    return (
        "\n\nRespond ONLY with a single JSON object matching this schema "
        f"(no prose, no code fences):\n{schema}"
    )


def _validate_output(
    text: str, output_model: type[BaseModel],
) -> tuple[str | None, str | None]:
    """Return ``(normalized_json, None)`` or ``(None, error)``."""
    try:
        parsed = output_model.model_validate_json(strip_markdown_fences(text))
    except ValidationError as exc:
        return None, str(exc)
    return parsed.model_dump_json(), None


def _retry_prompt(
    previous: str, error: str, output_model: type[BaseModel],
) -> str:
    """Self-contained correction prompt — works for stateless subagents."""
    return (
        "Your previous reply was not valid JSON for the required schema.\n"
        f"Previous reply:\n{previous}\n\n"
        f"Validation error:\n{error}\n"
        "Reply again with ONLY the corrected JSON object."
        f"{_schema_instruction(output_model)}"
    )


def _task_prompt(task: str, output_model: type[BaseModel] | None) -> str:
    if output_model is None:
        return task
    return f"{task}{_schema_instruction(output_model)}"


def _one_turn_sync(sub: Agent, prompt: str) -> tuple[str, str]:
    """One child turn: forward events to the parent's sink.

    Returns ``(text, stopped_by)`` — the verdict makes a usage-limit
    cut machine-visible to the runner. A turn that dies mid-flight
    (provider error, cancellation) still forwards its accounting: the
    child's rounds were debited to the pool, so the parent's
    ``children`` aggregation must see them too.
    """
    text = ""
    stopped_by = "stop"
    try:
        for event in sub.stream(prompt):
            _forward(event)
            if isinstance(event, TurnFinished):
                text = event.text
                stopped_by = event.diagnostics.stopped_by
    except BaseException:
        if sub.last_turn is not None:
            _forward(TurnFinished(text="", diagnostics=sub.last_turn))
        raise
    return text, stopped_by


async def _one_turn_async(sub: Agent, prompt: str) -> tuple[str, str]:
    """Async mirror of :func:`_one_turn_sync`."""
    text = ""
    stopped_by = "stop"
    try:
        async for event in sub.astream(prompt):
            _forward(event)
            if isinstance(event, TurnFinished):
                text = event.text
                stopped_by = event.diagnostics.stopped_by
    except BaseException:
        if sub.last_turn is not None:
            _forward(TurnFinished(text="", diagnostics=sub.last_turn))
        raise
    return text, stopped_by


# Appended to a plain-text child result the loop cut early — a usage
# limit (the child's own per-turn limit or the run's shared pool) or a
# stuck loop of identical tool calls — so the orchestrating model
# knows the answer is partial.
_PARTIAL_RESULT_NOTES = {
    "usage_limit": (
        "\n\n[subagent stopped early: usage limit reached — partial result]"
    ),
    "stuck": (
        "\n\n[subagent stopped early: stuck repeating identical tool "
        "calls — partial result]"
    ),
}


def _partial(text: str, stopped_by: str) -> str:
    return text + _PARTIAL_RESULT_NOTES.get(stopped_by, "")


def _guard_retry_budget(err: str) -> None:
    """Refuse a retry turn when the run's shared pool is exhausted.

    A retry against an empty pool would only burn another wrap-up
    round. A child's own per-turn limit does not suppress the retry —
    a new turn gets a fresh per-turn allowance by contract.
    """
    pool = _USAGE_POOL.get()
    if pool is not None and pool.breach() is not None:
        msg = (
            "usage limit reached before schema-valid output was "
            f"produced: {err}"
        )
        raise ValueError(msg)


def _run_sync(
    sub: Agent,
    task: str,
    output_model: type[BaseModel] | None,
    max_output_retries: int,
) -> str:
    text, stopped_by = _one_turn_sync(sub, _task_prompt(task, output_model))
    if output_model is None:
        return _partial(text, stopped_by)
    normalized, err = _validate_output(text, output_model)
    for _ in range(max_output_retries):
        if err is None:
            break
        _guard_retry_budget(err)
        text, _ = _one_turn_sync(sub, _retry_prompt(text, err, output_model))
        normalized, err = _validate_output(text, output_model)
    if err is not None or normalized is None:
        msg = f"subagent output failed schema validation: {err}"
        raise ValueError(msg)
    return normalized


async def _run_async(
    sub: Agent,
    task: str,
    output_model: type[BaseModel] | None,
    max_output_retries: int,
) -> str:
    text, stopped_by = await _one_turn_async(
        sub, _task_prompt(task, output_model),
    )
    if output_model is None:
        return _partial(text, stopped_by)
    normalized, err = _validate_output(text, output_model)
    for _ in range(max_output_retries):
        if err is None:
            break
        _guard_retry_budget(err)
        retry = _retry_prompt(text, err, output_model)
        text, _ = await _one_turn_async(sub, retry)
        normalized, err = _validate_output(text, output_model)
    if err is not None or normalized is None:
        msg = f"subagent output failed schema validation: {err}"
        raise ValueError(msg)
    return normalized


_TASK_PARAM_DESCRIPTION = (
    "Complete, self-contained task prompt. The subagent starts with a "
    "clean context and sees nothing of this conversation — include the "
    "objective, needed context, and the expected output."
)


def _make_subagent_tool(
    sub: Agent,
    name: str,
    description: str,
    *,
    output_model: type[BaseModel] | None = None,
    max_output_retries: int = 1,
) -> AgentTool:
    """Wrap *sub* as a single tool — the ``Agent.as_tool()`` primitive."""
    _guard_no_nesting(sub)
    _guard_clean_context(sub)

    def run(task: str) -> str:
        # Re-check at call time: the registration-time guard misses
        # subagents added to *sub* after as_tool() wrapped it.
        _guard_no_nesting(sub)
        return _run_sync(sub, task, output_model, max_output_retries)

    # Docstring is dynamic (per-subagent param description), so the
    # @tool decorator is applied functionally — it derives the schema
    # and the Pydantic input model from signature + Args section.
    run.__doc__ = f"Delegate a task to this subagent.\n\nArgs:\n    task: {_TASK_PARAM_DESCRIPTION}"
    # read_only: dispatching subagents concurrently is the point of
    # fan-out (Claude Code runs its Task calls the same way); whether
    # the children's own tools may overlap is governed per-tool inside
    # each child's loop.
    subagent_tool = tool(name=name, description=description, read_only=True)(run)

    async def acall(_original_name: str, tool_input: dict[str, Any]) -> str:
        _guard_no_nesting(sub)
        return await _run_async(
            sub, tool_input["task"], output_model, max_output_retries,
        )

    object.__setattr__(subagent_tool, "_anchor_async_caller", acall)
    object.__setattr__(subagent_tool, _SUBAGENT_MARKER, True)
    return subagent_tool


def _make_task_tool(
    subagents: dict[str, tuple[SubagentDefinition, Agent]],
) -> AgentTool:
    """Create the ``task`` meta-tool dispatching to registered subagents."""

    def _resolve(agent_name: str) -> tuple[SubagentDefinition, Agent] | str:
        entry = subagents.get(agent_name)
        if entry is None:
            available = ", ".join(subagents) or "none"
            return f"Unknown subagent: '{agent_name}'. Available: {available}"
        return entry

    @tool(
        name="task",
        description=(
            "Delegate a task to a named subagent from the available "
            "subagents list. The subagent starts with a clean context: "
            "write a complete, self-contained task prompt."
        ),
        # Parallel fan-out is the point of the task tool; each child's
        # own tools still serialize their writes inside the child loop.
        read_only=True,
    )
    def task(agent_name: str, task: str) -> str:
        """Delegate a task to a subagent.

        Args:
            agent_name: Name of the subagent to delegate to.
            task: Complete, self-contained task prompt.
        """
        resolved = _resolve(agent_name)
        if isinstance(resolved, str):
            return resolved
        definition, sub = resolved
        # Re-check at call time: subagents may have been added to the
        # registered child after with_subagents() built it.
        _guard_no_nesting(sub)
        return _run_sync(sub, task, definition.output_model, 1)

    async def acall(_original_name: str, tool_input: dict[str, Any]) -> str:
        resolved = _resolve(tool_input.get("agent_name", ""))
        if isinstance(resolved, str):
            return resolved
        definition, sub = resolved
        _guard_no_nesting(sub)
        return await _run_async(
            sub, tool_input.get("task", ""), definition.output_model, 1,
        )

    object.__setattr__(task, "_anchor_async_caller", acall)
    object.__setattr__(task, _SUBAGENT_MARKER, True)
    return task


def _subagent_listing(
    subagents: dict[str, tuple[SubagentDefinition, Agent]],
) -> str:
    """Static discovery block for the system prompt (cache-stable)."""
    if not subagents:
        return ""
    lines = [
        f"  - {name}: {definition.description}"
        for name, (definition, _) in subagents.items()
    ]
    return "Available subagents (delegate with the task tool):\n" + "\n".join(lines)
