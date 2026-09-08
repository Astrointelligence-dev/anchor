"""Pre/post tool-call hooks and observer callbacks for the agent loop.

Two distinct mechanisms:

- **Veto hooks** (``PreToolHook``/``PostToolHook``): a pre-hook can
  deny a call, rewrite its input, or answer ``"ask"`` to route the
  call to the agent's approval callback; a post-hook can replace the
  output (``decision`` is ignored in the post path). Exceptions in a
  pre-hook fail closed (the call is denied with the exception as
  reason).
- **Approval callback** (``ApprovalCallback``): the inline
  human-in-the-loop seam. Registered via ``Agent.with_approval``; the
  loop pauses the tool call until it returns an
  :class:`ApprovalDecision`. Triggered by
  ``AgentTool(requires_approval=True)`` or a pre-hook ``"ask"``; a
  trigger with no callback configured fails closed (deny). A denied
  call becomes an ``is_error`` tool result carrying the reason, so the
  model can adjust instead of blind-retrying.
- **Observer callbacks** (``AgentCallback``): fire-and-forget
  notifications dispatched via :func:`anchor._callbacks.fire_callbacks`;
  exceptions are swallowed and logged, mirroring pipeline/memory
  callbacks.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel


class HookResult(BaseModel, frozen=True):
    """Decision returned by a tool hook.

    ``decision``/``reason`` apply to pre-hooks only — the reason is fed
    back to the model on deny so it can adjust instead of
    blind-retrying; ``"ask"`` routes the call to the agent's approval
    callback. ``updated_input`` applies to pre-hooks,
    ``updated_output`` to post-hooks.
    """

    decision: Literal["allow", "deny", "ask"] = "allow"
    reason: str | None = None
    updated_input: dict[str, Any] | None = None
    updated_output: str | None = None


PreToolHook = Callable[[str, dict[str, Any]], "HookResult | None"]
"""``(tool_name, tool_input) -> HookResult | None`` (None = allow unchanged)."""

PostToolHook = Callable[[str, dict[str, Any], str], "HookResult | None"]
"""``(tool_name, tool_input, output) -> HookResult | None``."""


class ApprovalRequest(BaseModel, frozen=True):
    """A tool call awaiting human approval."""

    tool_call_id: str
    name: str
    tool_input: dict[str, Any]


class ApprovalDecision(BaseModel, frozen=True):
    """Answer to an :class:`ApprovalRequest`.

    ``reason`` is fed back to the model on deny; ``updated_input``
    replaces the call's input on approve.
    """

    approved: bool
    reason: str | None = None
    updated_input: dict[str, Any] | None = None


ApprovalCallback = Callable[
    [ApprovalRequest], "ApprovalDecision | Awaitable[ApprovalDecision]"
]
"""Sync or async ``(ApprovalRequest) -> ApprovalDecision``.

An async callback requires the async loop (``astream``/``achat``); it
may stay pending indefinitely — the turn waits.
"""


@runtime_checkable
class AgentCallback(Protocol):
    """Observer for agent-loop events.

    All methods are optional no-ops; implement only what you need.
    Exceptions raised by callbacks are swallowed and logged.
    """

    def on_round_start(self, round_index: int) -> None:
        ...

    def on_round_end(self, round_index: int) -> None:
        ...

    def on_tool_start(self, name: str, tool_input: dict[str, Any]) -> None:
        ...

    def on_tool_end(
        self, name: str, tool_input: dict[str, Any], result: str,
    ) -> None:
        ...

    def on_tool_error(
        self, name: str, tool_input: dict[str, Any], error: str,
    ) -> None:
        ...
