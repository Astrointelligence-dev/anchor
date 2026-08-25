"""Phase 1 regression tests: agentskills.io spec compliance + level-3 disclosure.

Pins the behaviors documented in docs/research/2026-08-25-sota-gap-analysis.md §1:
real YAML frontmatter, spec fields, name==directory validation, lazy
references/, opt-in scripts/, always-skill instruction injection, and a
cache-stable discovery listing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from anchor.agent.agent import Agent
from anchor.agent.skills.evals import SkillEvalCase, run_skill_eval
from anchor.agent.skills.loader import load_skill
from anchor.agent.skills.models import Skill
from anchor.agent.skills.registry import SkillRegistry
from anchor.agent.skills.resources import (
    _make_read_skill_file_tool,
    _make_run_skill_script_tool,
)
from anchor.llm.models import LLMResponse, Message, StopReason, StreamChunk, ToolSchema

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "skills"


class _EchoLLM:
    """Minimal fake provider: one text chunk, captures messages."""

    def __init__(self, text: str = "ok") -> None:
        self._text = text
        self.last_messages: list[Message] | None = None

    def stream(
        self, messages: list[Message], *, tools: list[ToolSchema] | None = None, **kwargs
    ) -> Iterator[StreamChunk]:
        self.last_messages = list(messages)
        yield StreamChunk(content=self._text)
        yield StreamChunk(stop_reason=StopReason.STOP)

    async def astream(
        self, messages: list[Message], *, tools: list[ToolSchema] | None = None, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        self.last_messages = list(messages)
        yield StreamChunk(content=self._text)
        yield StreamChunk(stop_reason=StopReason.STOP)

    def complete(self, messages: list[Message], **kwargs) -> LLMResponse:
        raise NotImplementedError

    async def acomplete(self, messages: list[Message], **kwargs) -> LLMResponse:
        raise NotImplementedError


class TestSpecFrontmatter:
    def test_full_spec_fixture_loads(self) -> None:
        skill = load_skill(FIXTURES / "spec-demo")
        assert skill.name == "spec-demo"
        # multi-line >- folded description parsed by real YAML
        assert "multi-line YAML descriptions" in skill.description
        assert "\n" not in skill.description
        assert skill.license == "Apache-2.0"
        assert skill.compatibility == "Requires Python 3.11+"
        assert skill.allowed_tools == ("read_skill_file", "run_skill_script")
        assert skill.metadata["version"] == "1.0.0"
        assert skill.tags == ("testing", "spec")
        assert skill.path == (FIXTURES / "spec-demo").resolve()

    def test_activation_from_metadata(self) -> None:
        skill = load_skill(FIXTURES / "spec-demo")
        assert skill.activation == "on_demand"

    def test_name_must_match_directory(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "wrong-dir"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: other-name\ndescription: mismatch test\n---\nBody"
        )
        with pytest.raises(ValueError, match="directory name"):
            load_skill(skill_dir)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "bad-yaml"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: bad-yaml\ndescription: [unclosed\n---\nBody"
        )
        with pytest.raises(ValueError, match="YAML"):
            load_skill(skill_dir)

    def test_unknown_keys_warn_but_load(self, tmp_path: Path, caplog) -> None:
        skill_dir = tmp_path / "extra-keys"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: extra-keys\ndescription: has extras\nversion: 2\n---\nBody"
        )
        with caplog.at_level("WARNING"):
            skill = load_skill(skill_dir)
        assert skill.name == "extra-keys"
        assert "unknown frontmatter keys" in caplog.text

    def test_bundled_file_listings(self) -> None:
        skill = load_skill(FIXTURES / "spec-demo")
        assert skill.reference_files() == ["references/guide.md"]
        assert skill.script_files() == ["scripts/hello.py"]


class TestLevel3Disclosure:
    def _registry(self) -> SkillRegistry:
        reg = SkillRegistry()
        reg.load_from_path(FIXTURES / "spec-demo")
        return reg

    def test_read_reference_file(self) -> None:
        tool = _make_read_skill_file_tool(self._registry())
        content = tool.fn(skill_name="spec-demo", file_path="references/guide.md")
        assert "Step 1: greet the user." in content

    def test_read_rejects_path_traversal(self) -> None:
        tool = _make_read_skill_file_tool(self._registry())
        result = tool.fn(skill_name="spec-demo", file_path="../invalid/SKILL.md")
        assert "escapes the skill directory" in result

    def test_read_unknown_file_lists_available(self) -> None:
        tool = _make_read_skill_file_tool(self._registry())
        result = tool.fn(skill_name="spec-demo", file_path="references/nope.md")
        assert "references/guide.md" in result

    def test_run_script(self) -> None:
        tool = _make_run_skill_script_tool(self._registry())
        result = tool.fn(
            skill_name="spec-demo", script_path="scripts/hello.py", args="anchor"
        )
        assert "exit code: 0" in result
        assert "hello anchor" in result

    def test_scripts_gated_by_opt_in(self) -> None:
        agent = Agent(llm=_EchoLLM()).with_skill_from_path(FIXTURES / "spec-demo")
        names = [t.name for t in agent._all_active_tools()]
        assert "read_skill_file" in names
        assert "run_skill_script" not in names

        allowed = Agent(llm=_EchoLLM(), allow_skill_scripts=True).with_skill_from_path(
            FIXTURES / "spec-demo"
        )
        names = [t.name for t in allowed._all_active_tools()]
        assert "run_skill_script" in names

    def test_activation_response_lists_references(self) -> None:
        reg = self._registry()
        from anchor.agent.skills.activate import _make_activate_skill_tool

        activate = _make_activate_skill_tool(reg)
        response = activate.fn(skill_name="spec-demo")
        assert "references/guide.md" in response
        assert "scripts/hello.py" in response


class TestSystemPromptAssembly:
    def test_always_instructions_reach_the_model(self) -> None:
        llm = _EchoLLM()
        skill = Skill(
            name="style",
            description="House style rules.",
            instructions="Always answer in haiku.",
            activation="always",
        )
        agent = Agent(llm=llm).with_system_prompt("Base prompt.").with_skill(skill)
        list(agent.chat("hi"))
        assert llm.last_messages is not None
        system = llm.last_messages[0].content
        assert "Base prompt." in system
        assert "Always answer in haiku." in system

    def test_discovery_joined_with_newlines_not_space(self) -> None:
        llm = _EchoLLM()
        skill = Skill(name="finder", description="Finds things.", activation="on_demand")
        agent = Agent(llm=llm).with_system_prompt("Base prompt.").with_skill(skill)
        list(agent.chat("hi"))
        system = llm.last_messages[0].content
        assert "Base prompt.\n\n" in system
        assert "finder: Finds things." in system

    def test_discovery_listing_capped(self) -> None:
        reg = SkillRegistry()
        for i in range(50):
            reg.register(
                Skill(
                    name=f"skill-{i:02d}",
                    description="d" * 100,
                    activation="on_demand",
                )
            )
        prompt = reg.skill_discovery_prompt(max_chars=500)
        assert len(prompt) < 700  # cap + summary line
        assert "more skills not listed" in prompt


class TestSkillEvalHarness:
    def test_with_without_baseline(self) -> None:
        skill = Skill(
            name="haiku",
            description="Answer in haiku",
            instructions="Answer in haiku.",
            activation="always",
        )

        def make_agent() -> Agent:
            return Agent(llm=_EchoLLM("five seven five syllables"))

        cases = [SkillEvalCase(prompt="write about the sea", expected="syllables")]
        report = run_skill_eval(
            make_agent,
            skill,
            cases,
            judge_fn=lambda case, output: case.expected in output,
        )
        assert report.pass_rate_with_skill == 1.0
        assert report.pass_rate_without_skill == 1.0
        assert report.lift == 0.0
        assert len(report.results) == 1
