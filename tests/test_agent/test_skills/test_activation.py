"""Tests for the activate_skill meta-tool."""

from __future__ import annotations

from anchor.agent.skills.activate import _make_activate_skill_tool
from anchor.agent.skills.models import Skill
from anchor.agent.skills.registry import SkillRegistry
from anchor.agent.tools import AgentTool


def _noop() -> str:
    return "ok"


def _make_tool(name: str = "t") -> AgentTool:
    return AgentTool(
        name=name, description="test",
        input_schema={"type": "object", "properties": {}}, fn=_noop,
    )


class TestActivateSkillTool:
    def test_tool_schema(self) -> None:
        reg = SkillRegistry()
        tool = _make_activate_skill_tool(reg)
        assert tool.name == "activate_skill"
        schema = tool.to_tool_schema()
        assert "skill_name" in schema.input_schema["properties"]

    def test_activates_skill(self) -> None:
        reg = SkillRegistry()
        skill = Skill(
            name="rag", description="Search docs",
            instructions="Use search_docs for queries.",
            tools=(_make_tool("search_docs"),), activation="on_demand",
        )
        reg.register(skill)
        tool = _make_activate_skill_tool(reg)

        result = tool.fn(skill_name="rag")
        assert "activated" in result
        assert "search_docs" in result
        assert "Use search_docs" in result
        assert reg.is_active("rag") is True

    def test_unknown_skill_returns_error(self) -> None:
        reg = SkillRegistry()
        reg.register(Skill(
            name="rag", description="d",
            tools=(_make_tool("sd"),), activation="on_demand",
        ))
        tool = _make_activate_skill_tool(reg)

        result = tool.fn(skill_name="nonexistent")
        assert "Unknown skill" in result
        assert "rag" in result  # lists available skills

    def test_no_instructions_still_works(self) -> None:
        reg = SkillRegistry()
        skill = Skill(
            name="calc", description="Math",
            tools=(_make_tool("calculate"),), activation="on_demand",
        )
        reg.register(skill)
        tool = _make_activate_skill_tool(reg)

        result = tool.fn(skill_name="calc")
        assert "activated" in result
        assert "calculate" in result

    def test_activation_makes_tools_available(self) -> None:
        reg = SkillRegistry()
        skill = Skill(
            name="rag", description="d",
            tools=(_make_tool("search_docs"),), activation="on_demand",
        )
        reg.register(skill)

        # Before activation: no tools from on-demand skill
        assert reg.active_tools() == []

        tool = _make_activate_skill_tool(reg)
        tool.fn(skill_name="rag")

        # After activation: skill's tools available
        active = reg.active_tools()
        assert len(active) == 1
        assert active[0].name == "search_docs"

    def test_no_available_skills_message(self) -> None:
        reg = SkillRegistry()
        tool = _make_activate_skill_tool(reg)
        result = tool.fn(skill_name="nope")
        assert "Unknown skill" in result
        assert "none" in result
