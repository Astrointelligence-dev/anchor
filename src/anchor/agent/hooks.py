"""Pre/post tool-call hooks and observer callbacks for the agent loop.

Two distinct mechanisms:

- **Veto hooks** (``PreToolHook``/``PostToolHook``): a pre-hook can
  deny a call or rewrite its input; a post-hook can replace the output
  (``decision`` is ignored in the post path). Exceptions in a pre-hook
  fail closed (the call is denied with the exception as reason).
- **Observer callbacks** (``AgentCallback``): fire-and-forget
  notifications dispatched via :func:`anchor._callbacks.fire_callbacks`;
  exceptions are swallowed and logged, mirroring pipeline/memory
  callbacks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel


class HookResult(BaseModel, frozen=True):
    """Decision returned by a tool hook.

    ``decision``/``reason`` apply to pre-hooks only — the reason is fed
    back to the model on deny so it can adjust instead of
    blind-retrying. ``updated_input`` applies to pre-hooks,
    ``updated_output`` to post-hooks.
    """

    decision: Literal["allow", "deny"] = "allow"
    reason: str | None = None
    updated_input: dict[str, Any] | None = None
    updated_output: str | None = None


PreToolHook = Callable[[str, dict[str, Any]], "HookResult | None"]
"""``(tool_name, tool_input) -> HookResult | None`` (None = allow unchanged)."""

PostToolHook = Callable[[str, dict[str, Any], str], "HookResult | None"]
"""``(tool_name, tool_input, output) -> HookResult | None``."""


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
