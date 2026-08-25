"""Typed events emitted by the agent loop (``Agent.stream``/``Agent.astream``).

One discriminated union (``AgentEvent``, by the ``type`` literal) delivered
through a single iterator — ``chat``/``achat`` are text-only projections of
the same stream. Events forwarded from a subagent carry the parent tool
call's id in ``parent_tool_call_id``; top-level events leave it ``None``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Literal

from pydantic import BaseModel

from anchor.agent.models import RoundUsage, TurnDiagnostics


class _BaseEvent(BaseModel, frozen=True):
    parent_tool_call_id: str | None = None


class TurnStarted(_BaseEvent):
    """The turn's context is built; rounds are about to run."""

    type: Literal["turn_started"] = "turn_started"


class RoundStarted(_BaseEvent):
    """A model round is starting (``round`` is 0-based)."""

    type: Literal["round_started"] = "round_started"
    round: int
    max_rounds: int


class TextDelta(_BaseEvent):
    """Incremental assistant text — the projection target of ``chat``."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolStarted(_BaseEvent):
    """A tool call is about to execute.

    ``tool_input`` is the model's requested arguments; a pre-hook may
    still rewrite them before execution (observer callbacks receive the
    resolved input).
    """

    type: Literal["tool_started"] = "tool_started"
    tool_call_id: str
    name: str
    tool_input: dict[str, Any]


class ToolFinished(_BaseEvent):
    """A tool call completed. Tool failure is ``is_error=True``, not an
    exception — only turn-level failures (provider, MCP) raise."""

    type: Literal["tool_finished"] = "tool_finished"
    tool_call_id: str
    name: str
    result: str
    is_error: bool = False


class CompactionStarted(_BaseEvent):
    """Client-side compaction is summarizing older messages (LLM call)."""

    type: Literal["compaction_started"] = "compaction_started"


class CompactionFinished(_BaseEvent):
    """Client-side compaction replaced the head with a summary."""

    type: Literal["compaction_finished"] = "compaction_finished"
    tokens_before: int
    tokens_after: int


class RoundFinished(_BaseEvent):
    """A round completed, with its token accounting."""

    type: Literal["round_finished"] = "round_finished"
    round: int
    usage: RoundUsage


class TurnFinished(_BaseEvent):
    """Terminal event: the final text and per-round diagnostics.

    Also available after the turn as ``agent.last_turn`` /
    the return of the text projections.
    """

    type: Literal["turn_finished"] = "turn_finished"
    text: str
    diagnostics: TurnDiagnostics


AgentEvent = (
    TurnStarted
    | RoundStarted
    | TextDelta
    | ToolStarted
    | ToolFinished
    | CompactionStarted
    | CompactionFinished
    | RoundFinished
    | TurnFinished
)


# Per-tool-call event sink: ``(emit, parent_tool_call_id)``. The loop's
# tool phase sets it around each call; a subagent runner executing
# inside that call forwards the child's events through ``emit`` with
# ``parent_tool_call_id`` stamped, flattening nested runs into the
# parent stream. Each asyncio task gets its own context copy, so
# parallel tool calls never see each other's sink.
_EVENT_SINK: ContextVar[tuple[Callable[[AgentEvent], None], str] | None] = (
    ContextVar("anchor_agent_event_sink", default=None)
)


def _forward(event: AgentEvent) -> None:
    """Emit *event* to the active sink (if any) with the parent id set."""
    sink = _EVENT_SINK.get()
    if sink is not None:
        emit, parent_id = sink
        emit(event.model_copy(update={"parent_tool_call_id": parent_id}))
