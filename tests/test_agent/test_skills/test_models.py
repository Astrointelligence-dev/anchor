"""Tests for the Skill data model."""

from __future__ import annotations

import pytest

from anchor.agent.skills.models import Skill
from anchor.agent.tools import AgentTool


def _noop() -> str:
    return "ok"


def _make_tool(name: str = "t") -> AgentTool:
    return AgentTool(
        name=name,
        description="test",
        input_schema={"type": "object", "properties": {}},
        fn=_noop,
    )


class TestSkillCreation:
    def test_defaults(self) -> None:
        skill = Skill(name="test", description="A test skill")
        assert skill.name == "test"
        assert skill.description == "A test skill"
        assert skill.instructions == ""
        assert skill.tools == ()
        assert skill.activation == "on_demand"
        assert skill.tags == ()

    def test_full_creation(self) -> None:
        tool = _make_tool()
        skill = Skill(
            name="memory",
            description="Memory management",
            instructions="Use save_fact to store info.",
            tools=(tool,),
            activation="on_demand",
            tags=("core", "memory"),
        )
        assert skill.name == "memory"
        assert skill.activation == "on_demand"
        assert len(skill.tools) == 1
        assert skill.tags == ("core", "memory")

    def test_frozen(self) -> None:
        skill = Skill(name="x", description="d")
        with pytest.raises((AttributeError, Exception)):
            skill.name = "y"  # type: ignore[misc]

    def test_multiple_tools(self) -> None:
        tools = (_make_tool("a"), _make_tool("b"), _make_tool("c"))
        skill = Skill(name="multi", description="d", tools=tools)
        assert len(skill.tools) == 3
        assert [t.name for t in skill.tools] == ["a", "b", "c"]
