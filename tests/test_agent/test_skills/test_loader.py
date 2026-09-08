"""Tests for the SKILL.md loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from anchor.agent.skills.loader import load_skill, load_skills_directory

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures" / "skills"


class TestLoadSkillParsing:
    def test_parses_valid_frontmatter(self) -> None:
        skill = load_skill(FIXTURES / "brainstorm")
        assert skill.name == "brainstorm"
        assert skill.description == "Guide the agent through a structured brainstorming process"
        assert skill.activation == "on_demand"
        assert skill.tags == ("creative",)

    def test_instructions_from_markdown_body(self) -> None:
        skill = load_skill(FIXTURES / "brainstorm")
        assert "# Brainstorming Skill" in skill.instructions
        assert "Propose approaches" in skill.instructions

    def test_instructions_only_skill(self) -> None:
        skill = load_skill(FIXTURES / "minimal-helper")
        assert skill.name == "minimal-helper"
        assert skill.tools == ()
        assert skill.activation == "on_demand"

    def test_default_activation_is_on_demand(self) -> None:
        skill = load_skill(FIXTURES / "minimal-helper")
        assert skill.activation == "on_demand"


class TestLoadSkillValidation:
    def test_missing_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="name"):
            load_skill(FIXTURES / "invalid")

    def test_nonexistent_path_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_skill(FIXTURES / "does-not-exist")

    def test_invalid_name_format_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "bad_name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: BadName!!\ndescription: test\n---\nBody"
        )
        with pytest.raises(ValueError, match="name"):
            load_skill(skill_dir)

    def test_name_too_long_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "long"
        skill_dir.mkdir()
        long_name = "a" * 65
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {long_name}\ndescription: test\n---\nBody"
        )
        with pytest.raises(ValueError, match="name"):
            load_skill(skill_dir)

    def test_missing_description_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: no-desc\n---\nBody"
        )
        with pytest.raises(ValueError, match="description"):
            load_skill(skill_dir)

    def test_description_too_long_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        long_desc = "a" * 1025
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: test-skill\ndescription: {long_desc}\n---\nBody"
        )
        with pytest.raises(ValueError, match="description"):
            load_skill(skill_dir)


class TestActivationOverride:
    def test_activation_always_override(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "always-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: always-skill\ndescription: test\nactivation: always\n---\nBody"
        )
        skill = load_skill(skill_dir)
        assert skill.activation == "always"


class TestToolDiscovery:
    def test_discovers_tools_from_tools_py(self) -> None:
        skill = load_skill(FIXTURES / "brainstorm")
        assert len(skill.tools) == 1
        assert skill.tools[0].name == "save_brainstorm_result"

    def test_instructions_only_has_no_tools(self) -> None:
        skill = load_skill(FIXTURES / "minimal-helper")
        assert skill.tools == ()

    def test_tool_is_callable(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        skill = load_skill(FIXTURES / "brainstorm")
        tool = skill.tools[0]
        result = tool.fn(title="Test", summary="Sum", approaches="A vs B")
        assert "saved" in result.lower() or "brainstorm" in result.lower()
        assert (tmp_path / "brainstorm_output.json").exists()

    def test_tools_py_import_error_raises(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "bad-tools"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: bad-tools\ndescription: test\n---\nBody"
        )
        (skill_dir / "tools.py").write_text("import nonexistent_module_xyz")
        with pytest.raises(ValueError, match="Failed to import"):
            load_skill(skill_dir)


class TestLoadSkillsDirectory:
    def test_loads_all_valid_skills(self) -> None:
        skills = load_skills_directory(FIXTURES)
        names = {s.name for s in skills}
        assert "brainstorm" in names
        assert "minimal-helper" in names

    def test_skips_invalid_skills(self) -> None:
        skills = load_skills_directory(FIXTURES)
        names = {s.name for s in skills}
        # "invalid" directory has bad frontmatter -- should be skipped, not present
        assert "brainstorm" in names
        assert "minimal-helper" in names
        # invalid skill directory should not appear in loaded skills
        for name in names:
            assert name  # no empty names
        assert len(skills) >= 2  # at least brainstorm + minimal

    def test_nonexistent_dir_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_skills_directory("/nonexistent/path")

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        skills = load_skills_directory(tmp_path)
        assert skills == []
