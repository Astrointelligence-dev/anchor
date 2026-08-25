"""Tests for the SkillRegistry."""

from __future__ import annotations

import pytest

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


def _always_skill(name: str = "s1", tool_name: str = "t1") -> Skill:
    return Skill(
        name=name, description=f"{name} desc",
        tools=(_make_tool(tool_name),), activation="always",
    )


def _on_demand_skill(name: str = "s2", tool_name: str = "t2") -> Skill:
    return Skill(
        name=name, description=f"{name} desc",
        instructions=f"Use {tool_name} to do things.",
        tools=(_make_tool(tool_name),), activation="on_demand",
    )


class TestRegister:
    def test_register_and_get(self) -> None:
        reg = SkillRegistry()
        skill = _always_skill()
        reg.register(skill)
        assert reg.get("s1") is skill

    def test_get_unknown_returns_none(self) -> None:
        reg = SkillRegistry()
        assert reg.get("missing") is None

    def test_duplicate_raises(self) -> None:
        reg = SkillRegistry()
        reg.register(_always_skill("dup"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_always_skill("dup", tool_name="other"))


class TestActivation:
    def test_always_is_active_immediately(self) -> None:
        reg = SkillRegistry()
        reg.register(_always_skill())
        assert reg.is_active("s1") is True

    def test_on_demand_not_active_initially(self) -> None:
        reg = SkillRegistry()
        reg.register(_on_demand_skill())
        assert reg.is_active("s2") is False

    def test_activate_on_demand(self) -> None:
        reg = SkillRegistry()
        reg.register(_on_demand_skill())
        result = reg.activate("s2")
        assert result.name == "s2"
        assert reg.is_active("s2") is True

    def test_activate_unknown_raises(self) -> None:
        reg = SkillRegistry()
        with pytest.raises(KeyError, match="Unknown skill"):
            reg.activate("nope")

    def test_deactivate(self) -> None:
        reg = SkillRegistry()
        reg.register(_on_demand_skill())
        reg.activate("s2")
        assert reg.is_active("s2") is True
        reg.deactivate("s2")
        assert reg.is_active("s2") is False

    def test_deactivate_unknown_is_safe(self) -> None:
        reg = SkillRegistry()
        reg.deactivate("nonexistent")  # should not raise

    def test_is_active_unknown_returns_false(self) -> None:
        reg = SkillRegistry()
        assert reg.is_active("ghost") is False

    def test_reset_clears_activations(self) -> None:
        reg = SkillRegistry()
        reg.register(_on_demand_skill("a"))
        reg.register(_on_demand_skill("b", tool_name="t3"))
        reg.activate("a")
        reg.activate("b")
        reg.reset()
        assert reg.is_active("a") is False
        assert reg.is_active("b") is False


class TestActiveTools:
    def test_always_tools_returned(self) -> None:
        reg = SkillRegistry()
        reg.register(_always_skill("s1", "tool_a"))
        tools = reg.active_tools()
        assert len(tools) == 1
        assert tools[0].name == "tool_a"

    def test_inactive_on_demand_excluded(self) -> None:
        reg = SkillRegistry()
        reg.register(_always_skill("s1", "tool_a"))
        reg.register(_on_demand_skill("s2", "tool_b"))
        tools = reg.active_tools()
        assert [t.name for t in tools] == ["tool_a"]

    def test_activated_on_demand_included(self) -> None:
        reg = SkillRegistry()
        reg.register(_always_skill("s1", "tool_a"))
        reg.register(_on_demand_skill("s2", "tool_b"))
        reg.activate("s2")
        tools = reg.active_tools()
        assert sorted(t.name for t in tools) == ["tool_a", "tool_b"]

    def test_duplicate_tool_names_raise_at_registration(self) -> None:
        reg = SkillRegistry()
        reg.register(_always_skill("s1", "same_name"))
        with pytest.raises(ValueError, match="Tool name collision"):
            reg.register(_always_skill("s2", "same_name"))

    def test_empty_registry(self) -> None:
        reg = SkillRegistry()
        assert reg.active_tools() == []


class TestOnDemandSkills:
    def test_returns_only_on_demand(self) -> None:
        reg = SkillRegistry()
        reg.register(_always_skill("a", "t1"))
        reg.register(_on_demand_skill("b", "t2"))
        reg.register(_on_demand_skill("c", "t3"))
        on_demand = reg.on_demand_skills()
        assert [s.name for s in on_demand] == ["b", "c"]


class TestDiscoveryPrompt:
    def test_empty_when_no_on_demand(self) -> None:
        reg = SkillRegistry()
        reg.register(_always_skill())
        assert reg.skill_discovery_prompt() == ""

    def test_lists_on_demand_skills(self) -> None:
        reg = SkillRegistry()
        reg.register(_on_demand_skill("rag", "search_docs"))
        prompt = reg.skill_discovery_prompt()
        assert "rag" in prompt
        assert "activate_skill" in prompt

    def test_prompt_is_stable_across_activation(self) -> None:
        """Discovery text must not change when a skill activates (cache-friendly)."""
        reg = SkillRegistry()
        reg.register(_on_demand_skill("rag", "search_docs"))
        before = reg.skill_discovery_prompt()
        reg.activate("rag")
        assert reg.skill_discovery_prompt() == before

    def test_empty_registry_empty_prompt(self) -> None:
        reg = SkillRegistry()
        assert reg.skill_discovery_prompt() == ""
