"""``@tool`` decorator for zero-boilerplate tool creation."""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

from pydantic import BaseModel

from anchor.agent.models import AgentTool
from anchor.agent.schema import (
    _get_first_doc_paragraph,
    build_input_model,
    clean_schema,
)


@overload
def tool(fn: Callable[..., str]) -> AgentTool: ...


@overload
def tool(
    fn: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    input_model: type[BaseModel] | None = None,
    requires_approval: bool = False,
    read_only: bool = False,
    max_result_tokens: int | None = None,
) -> Callable[[Callable[..., str]], AgentTool]: ...


def tool(
    fn: Callable[..., str] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    input_model: type[BaseModel] | None = None,
    requires_approval: bool = False,
    read_only: bool = False,
    max_result_tokens: int | None = None,
) -> AgentTool | Callable[[Callable[..., str]], AgentTool]:
    """Create an :class:`AgentTool` from a decorated function.

    Supports both bare and parameterised usage::

        @tool
        def greet(name: str) -> str:
            \"\"\"Greet someone.\"\"\"
            return f"Hello {name}"

        @tool(name="custom_greet", description="Custom greeting")
        def greet(name: str) -> str:
            return f"Hello {name}"

        @tool(input_model=GreetInput)
        def greet(name: str) -> str:
            return f"Hello {name}"

    Parameters
    ----------
    fn:
        The function to wrap (when used as bare ``@tool``).
    name:
        Override tool name (defaults to ``fn.__name__``).
    description:
        Override description (defaults to first docstring line).
    input_model:
        Explicit Pydantic model for input schema.  When omitted,
        a model is auto-generated from the function's type hints.
    read_only:
        Declare the tool side-effect free — the async loop may run it
        concurrently with other read-only calls. Default ``False``:
        an undeclared tool is a write and runs alone.
    max_result_tokens:
        Per-tool override of the agent's tool-result cap; ``None``
        inherits the agent-wide default.
    """
    if fn is not None:
        # Bare @tool usage
        return _build_agent_tool(
            fn, name=name, description=description, input_model=input_model,
            requires_approval=requires_approval, read_only=read_only,
            max_result_tokens=max_result_tokens,
        )

    # Parameterised @tool(...) usage — return a decorator
    def decorator(func: Callable[..., str]) -> AgentTool:
        return _build_agent_tool(
            func, name=name, description=description, input_model=input_model,
            requires_approval=requires_approval, read_only=read_only,
            max_result_tokens=max_result_tokens,
        )

    return decorator


def _build_agent_tool(
    fn: Callable[..., str],
    *,
    name: str | None,
    description: str | None,
    input_model: type[BaseModel] | None,
    requires_approval: bool = False,
    read_only: bool = False,
    max_result_tokens: int | None = None,
) -> AgentTool:
    """Internal helper that builds the AgentTool from a function."""
    tool_name = name or fn.__name__
    tool_description = description or _get_first_doc_paragraph(fn) or tool_name

    if input_model is not None:
        model = input_model
    else:
        model = build_input_model(fn, name=f"{tool_name}_input")

    raw_schema = model.model_json_schema()
    input_schema = clean_schema(raw_schema)

    return AgentTool(
        name=tool_name,
        description=tool_description,
        input_schema=input_schema,
        fn=fn,
        input_model=model,
        requires_approval=requires_approval,
        read_only=read_only,
        max_result_tokens=max_result_tokens,
    )
