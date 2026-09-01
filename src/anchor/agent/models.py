"""Agent data models: tools, usage accounting, and the shared budget pool."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from anchor.models.scope import RetrievalScope


class AgentTool(BaseModel):
    """A tool that the Agent can use during conversation.

    Each tool has a name, description, JSON Schema for inputs,
    and a callable that executes the tool logic.

    Supports three tiers of creation:

    1. ``@tool`` decorator (auto-generates schema from type hints)
    2. ``@tool(input_model=MyModel)`` (explicit Pydantic model)
    3. Direct construction with a raw ``input_schema`` dict

    ``read_only`` declares the tool has no side effects: the async
    loop runs consecutive read-only calls concurrently and every
    undeclared (write) call alone, in call order — the safe default,
    matching MCP's ``readOnlyHint`` (absent = writes). ``max_result_tokens``
    overrides the agent-wide result cap for this tool; ``None`` inherits it.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]
    input_model: type[BaseModel] | None = None
    timeout: float | None = None
    defer_loading: bool = False
    input_examples: tuple[dict[str, Any], ...] = ()
    requires_approval: bool = False
    read_only: bool = False
    max_result_tokens: int | None = Field(default=None, gt=0)

    def to_tool_schema(self) -> ToolSchema:
        """Convert to provider-agnostic ToolSchema."""
        from anchor.llm.models import ToolSchema

        return ToolSchema(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            input_examples=self.input_examples,
        )

    def validate_input(self, tool_input: dict[str, Any]) -> tuple[bool, str]:
        """Validate tool input against the schema.

        When ``input_model`` is set, uses full Pydantic validation.
        Otherwise falls back to basic JSON Schema type checking.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` when valid, ``(False, error_message)`` otherwise.
        """
        if self.input_model is not None:
            return self._pydantic_validate(tool_input)
        return self._basic_validate(tool_input)

    def _pydantic_validate(self, tool_input: dict[str, Any]) -> tuple[bool, str]:
        """Validate using the attached Pydantic input model."""
        assert self.input_model is not None  # noqa: S101
        try:
            self.input_model.model_validate(tool_input)
        except ValidationError as exc:
            return False, str(exc)
        return True, ""

    def _basic_validate(self, tool_input: dict[str, Any]) -> tuple[bool, str]:
        """Fallback validation using basic JSON Schema type checking.

        Checks required fields and basic type matching (string, number,
        integer, boolean).  Extra fields are allowed (lenient mode).
        """
        properties: dict[str, Any] = self.input_schema.get("properties", {})
        required: list[str] = self.input_schema.get("required", [])

        for field_name in required:
            if field_name not in tool_input:
                return False, f"Missing required field: '{field_name}'"

        _type_map: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
        }
        for key, value in tool_input.items():
            if key not in properties:
                continue
            expected_type_name = properties[key].get("type")
            if expected_type_name is None:
                continue
            expected = _type_map.get(expected_type_name)
            if expected is None:
                continue
            if not isinstance(value, expected):
                return (
                    False,
                    f"Field '{key}' expected type '{expected_type_name}', "
                    f"got '{type(value).__name__}'",
                )

        return True, ""


class RoundUsage(BaseModel, frozen=True):
    """Per-round token accounting for one iteration of the tool loop.

    ``prompt_tokens``/``completion_tokens`` come from provider-reported
    usage; when the provider does not report usage on the stream they
    are estimated with the agent's tokenizer (messages + tool schemas
    for the prompt, streamed text + tool-call arguments for the
    completion). ``tool_schema_tokens`` and ``tool_result_tokens`` are
    tokenizer-counted visibility fields — subsets of the prompt, never
    added on top of it. ``cost_usd`` is the round's USD price
    (genai-prices); it stays 0.0 unless a ``cost_limit`` is active, or
    when the model has no price data.
    """

    round: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_schema_tokens: int = 0
    tool_result_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Tokens this round consumed, as billed: full input + output."""
        return (
            self.prompt_tokens
            + self.completion_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )


class UsageLimits(BaseModel, frozen=True):
    """Usage limits enforced by the agent loop, shared across subagents.

    The agent that starts a turn with limits creates a run-wide pool;
    every subagent spawned during that turn (``task``/``as_tool``)
    debits the same pool, so a child's spend counts against the
    orchestrator's budget. A child may carry its own narrower limits —
    both are enforced, so the effective budget only ever narrows.

    When a limit is crossed mid-turn the loop grants the model one
    wrap-up round (final-round notice; ``tool_choice="none"``, or the
    ``final_result`` tool when structured output is still pending) and
    stops with ``stopped_by="usage_limit"`` — no exception is raised.
    A subagent that observes the breach wraps up the same way and
    returns its partial result to the parent. Checks are post-hoc, so
    a run may overshoot by the rounds in flight (one per concurrently
    running agent) plus the bounded wrap-up calls. Rounds are capped
    separately by each agent's ``max_rounds``.

    ``cost_limit`` is USD, priced per round via ``genai-prices``
    (``pip install astro-anchor[pricing]``). A model the price table
    does not know logs one warning and debits zero cost — its tokens
    still count.
    """

    total_tokens_limit: int | None = Field(default=None, gt=0)
    tool_calls_limit: int | None = Field(default=None, ge=0)
    cost_limit: float | None = Field(default=None, gt=0)


class TurnDiagnostics(BaseModel, frozen=True):
    """Accounting and outcome for one full ``chat()``/``achat()`` turn.

    ``rounds`` covers only this agent's own model rounds; subagent
    turns observed while this turn ran arrive in ``children``, each
    with its own diagnostics — the ``run_total_*`` properties aggregate
    both.
    """

    rounds: tuple[RoundUsage, ...] = ()
    stopped_by: Literal[
        "stop", "max_rounds", "max_tokens", "usage_limit", "output_missing",
        "stuck",
    ] = "stop"
    children: tuple[ChildTurn, ...] = ()

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.rounds)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.rounds)

    @property
    def total_tool_result_tokens(self) -> int:
        return sum(r.tool_result_tokens for r in self.rounds)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.rounds)

    @property
    def total_tool_calls(self) -> int:
        return sum(r.tool_calls for r in self.rounds)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.rounds)

    @property
    def run_total_tokens(self) -> int:
        """Tokens spent by this turn plus every subagent turn under it."""
        return self.total_tokens + sum(
            c.diagnostics.run_total_tokens for c in self.children
        )

    @property
    def run_total_tool_calls(self) -> int:
        """Tool calls made by this turn plus every subagent turn under it."""
        return self.total_tool_calls + sum(
            c.diagnostics.run_total_tool_calls for c in self.children
        )

    @property
    def run_total_cost_usd(self) -> float:
        """USD spent by this turn plus every subagent turn under it."""
        return self.total_cost_usd + sum(
            c.diagnostics.run_total_cost_usd for c in self.children
        )


class ChildTurn(BaseModel, frozen=True):
    """One subagent turn observed while the parent's turn ran.

    ``diagnostics.stopped_by`` makes a child cut by the shared pool
    machine-visible (``"usage_limit"``), so an orchestrating app can
    tell partial child results from complete ones — including a
    structured-output child whose wrap-up happened to produce valid
    JSON, which the tool result alone cannot reveal. Output-model
    retries produce one entry per child turn, sharing the
    ``tool_call_id``; a child that died mid-turn (provider error,
    timeout) appears with the accounting it had accrued.
    """

    tool_call_id: str
    name: str
    diagnostics: TurnDiagnostics


TurnDiagnostics.model_rebuild()


class _UsagePool:
    """Mutable shared budget for one run: a root turn plus its subagents.

    Created by the agent that starts a turn with :class:`UsageLimits`
    and handed down through ``_USAGE_POOL`` (per-task context copies,
    like the event sink); every agent in the run debits it as rounds
    close. Debits are synchronous with no await between check and
    debit, so the single event loop makes them atomic — the same
    lock-free shape as Pydantic AI's ``RunUsage`` and OpenAI Agents'
    ``Usage``.
    """

    # ponytail: lock-free — correct while all agents share one event
    # loop; add a threading.Lock when sync tools move to a thread pool.

    __slots__ = ("cost_spent", "limits", "owner", "tokens_spent", "tool_calls_spent")

    def __init__(self, limits: UsageLimits, *, owner: object | None = None) -> None:
        self.limits = limits
        self.owner = owner  # the Agent whose limits seeded the pool
        self.tokens_spent = 0
        self.tool_calls_spent = 0
        self.cost_spent = 0.0

    def debit(self, usage: RoundUsage) -> None:
        self.tokens_spent += usage.total_tokens
        self.tool_calls_spent += usage.tool_calls
        self.cost_spent += usage.cost_usd

    def breach(
        self,
    ) -> tuple[
        Literal["total_tokens", "tool_calls", "cost"], int | float, int | float,
    ] | None:
        """First crossed limit as ``(kind, used, limit)``, or None."""
        limits = self.limits
        if (
            limits.total_tokens_limit is not None
            and self.tokens_spent > limits.total_tokens_limit
        ):
            return ("total_tokens", self.tokens_spent, limits.total_tokens_limit)
        if (
            limits.tool_calls_limit is not None
            and self.tool_calls_spent > limits.tool_calls_limit
        ):
            return ("tool_calls", self.tool_calls_spent, limits.tool_calls_limit)
        if limits.cost_limit is not None and self.cost_spent > limits.cost_limit:
            return ("cost", self.cost_spent, limits.cost_limit)
        return None


# Hand-off point for the run's shared pool. Set only inside frames that
# reset in the same frame (around each tool call, and around a prompted
# run()'s retry loop) — never across a generator yield, so it cannot
# leak into the consumer's context. Subagents executing inside a tool
# call read it at turn start and debit the same pool object directly.
# The retrieval scope active for the current tool call: published in the
# same set/reset window as _USAGE_POOL, inherited by subagent turns (the
# child's effective scope = published ∩ its own — it can only narrow).
# Read by scope-aware tools via anchor.agent.current_scope().
_ACTIVE_SCOPE: ContextVar[RetrievalScope | None] = ContextVar(
    "anchor_active_scope", default=None,
)

_USAGE_POOL: ContextVar[_UsagePool | None] = ContextVar(
    "anchor_usage_pool", default=None,
)
