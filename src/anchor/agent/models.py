"""Base model for agent tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class AgentTool(BaseModel):
    """A tool that the Agent can use during conversation.

    Each tool has a name, description, JSON Schema for inputs,
    and a callable that executes the tool logic.

    Supports three tiers of creation:

    1. ``@tool`` decorator (auto-generates schema from type hints)
    2. ``@tool(input_model=MyModel)`` (explicit Pydantic model)
    3. Direct construction with a raw ``input_schema`` dict
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
    added on top of it.
    """

    round: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_schema_tokens: int = 0
    tool_result_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    tool_calls: int = 0

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
    """Per-turn usage limits enforced by the agent loop.

    When a limit is crossed mid-turn the loop grants the model one
    wrap-up round (final-round notice; ``tool_choice="none"``, or the
    ``final_result`` tool when structured output is still pending) and
    stops with ``stopped_by="usage_limit"`` — no exception is raised.
    Checks are post-hoc, so a turn may overshoot by the round in
    flight plus the bounded wrap-up call. Rounds are capped separately
    by the agent's ``max_rounds``.
    """

    total_tokens_limit: int | None = Field(default=None, gt=0)
    tool_calls_limit: int | None = Field(default=None, ge=0)


class TurnDiagnostics(BaseModel, frozen=True):
    """Accounting and outcome for one full ``chat()``/``achat()`` turn."""

    rounds: tuple[RoundUsage, ...] = ()
    stopped_by: Literal[
        "stop", "max_rounds", "max_tokens", "usage_limit", "output_missing",
    ] = "stop"

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
