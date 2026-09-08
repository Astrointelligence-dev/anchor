"""Integration tests for SKILL.md loader with SkillRegistry and Agent."""

from __future__ import annotations

from pathlib import Path

import pytest

from anchor.agent.agent import Agent
from anchor.agent.skills.models import Skill
from anchor.agent.skills.registry import SkillRegistry
from anchor.agent.tools import AgentTool

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "skills"


def _noop() -> str:
    return "ok"


def _make_tool(name: str = "t") -> AgentTool:
    return AgentTool(
        name=name, description="test",
        input_schema={"type": "object", "properties": {}}, fn=_noop,
    )


class TestRegistryLoadFromPath:
    def test_load_and_register(self) -> None:
        reg = SkillRegistry()
        skill = reg.load_from_path(FIXTURES / "brainstorm")
        assert skill.name == "brainstorm"
        assert reg.get("brainstorm") is skill

    def test_duplicate_name_raises(self) -> None:
        reg = SkillRegistry()
        reg.register(Skill(name="brainstorm", description="native"))
        with pytest.raises(ValueError, match="already registered"):
            reg.load_from_path(FIXTURES / "brainstorm")


class TestRegistryLoadFromDirectory:
    def test_loads_all_valid(self) -> None:
        reg = SkillRegistry()
        skills = reg.load_from_directory(FIXTURES)
        assert len(skills) >= 2
        assert reg.get("brainstorm") is not None
        assert reg.get("minimal-helper") is not None

    def test_skips_duplicate_continues(self) -> None:
        reg = SkillRegistry()
        reg.register(Skill(name="brainstorm", description="native"))
        skills = reg.load_from_directory(FIXTURES)
        # brainstorm skipped, minimal-helper loaded
        names = {s.name for s in skills}
        assert "minimal-helper" in names
        assert "brainstorm" not in names


class TestActivationFlow:
    def test_skillmd_on_demand_activation(self) -> None:
        reg = SkillRegistry()
        reg.load_from_path(FIXTURES / "brainstorm")
        assert reg.is_active("brainstorm") is False
        reg.activate("brainstorm")
        assert reg.is_active("brainstorm") is True
        tools = reg.active_tools()
        assert any(t.name == "save_brainstorm_result" for t in tools)

    def test_mixed_native_and_skillmd(self) -> None:
        reg = SkillRegistry()
        native = Skill(
            name="native-skill", description="native",
            tools=(_make_tool("native_tool"),), activation="always",
        )
        reg.register(native)
        reg.load_from_path(FIXTURES / "brainstorm")
        reg.activate("brainstorm")
        tools = reg.active_tools()
        names = {t.name for t in tools}
        assert "native_tool" in names
        assert "save_brainstorm_result" in names


class _FakeLLMProvider:
    """Minimal stand-in so Agent.__init__ doesn't need any SDK installed."""

    @property
    def model_id(self) -> str:
        return "mock/test"

    @property
    def provider_name(self) -> str:
        return "mock"

    def stream(self, messages, **kwargs):  # type: ignore[override]
        return iter([])

    async def astream(self, messages, **kwargs):  # type: ignore[override]
        return
        yield  # make it an async generator

    def invoke(self, messages, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    async def ainvoke(self, messages, **kwargs):  # type: ignore[override]
        raise NotImplementedError


class TestAgentSkillLoading:
    def test_with_skills_directory(self) -> None:
        agent = Agent(llm=_FakeLLMProvider())
        agent.with_skills_directory(FIXTURES)
        reg = agent._skill_registry
        assert reg.get("brainstorm") is not None
        assert reg.get("minimal-helper") is not None

    def test_with_skill_from_path(self) -> None:
        agent = Agent(llm=_FakeLLMProvider())
        agent.with_skill_from_path(FIXTURES / "brainstorm")
        reg = agent._skill_registry
        assert reg.get("brainstorm") is not None

    def test_chaining(self) -> None:
        agent = (
            Agent(llm=_FakeLLMProvider())
            .with_skill_from_path(FIXTURES / "brainstorm")
            .with_skills_directory(FIXTURES)
        )
        # Should not raise -- chaining works
        assert agent._skill_registry.get("brainstorm") is not None
        assert agent._skill_registry.get("minimal-helper") is not None

    def test_activate_tool_created_for_on_demand(self) -> None:
        agent = Agent(llm=_FakeLLMProvider())
        agent.with_skill_from_path(FIXTURES / "brainstorm")
        assert agent._activate_tool is not None
