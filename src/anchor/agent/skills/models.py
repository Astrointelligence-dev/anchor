"""Skill data model for progressive tool disclosure.

Follows the Agent Skills open standard (https://agentskills.io/specification):
a skill is a directory with a ``SKILL.md`` (YAML frontmatter + markdown body)
plus optional ``references/``, ``scripts/``, and ``assets/`` subdirectories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from anchor.agent.models import AgentTool


class Skill(BaseModel):
    """A named unit of procedural knowledge, optionally bundling tools.

    Progressive disclosure levels:

    1. ``name`` + ``description`` are always visible in the discovery prompt.
    2. ``instructions`` (the SKILL.md body) load when the skill activates.
    3. Files under ``references/`` are read lazily via the ``read_skill_file``
       meta-tool; files under ``scripts/`` are *executed* (never loaded) via
       ``run_skill_script`` when script execution is enabled.

    Spec fields (agentskills.io): ``name``, ``description``, ``license``,
    ``compatibility``, ``metadata``, ``allowed_tools``.
    Anchor extensions: ``activation``, ``tags``, ``tools`` (Python-authored
    tools), ``path`` (set by the loader for level-3 disclosure).

    Parameters
    ----------
    name:
        Unique identifier (lowercase, digits, hyphens; matches the skill
        directory name for file-loaded skills).
    description:
        What the skill does and when to use it. Shown in the discovery prompt.
    instructions:
        Detailed usage guide injected when the skill is activated.
    tools:
        The :class:`AgentTool` instances this skill provides.
    activation:
        ``"always"`` means instructions are injected into the system prompt
        and tools are loaded from round 1. ``"on_demand"`` (default) means
        the agent must call ``activate_skill`` first.
    tags:
        Optional tags for filtering or grouping skills.
    license:
        SPDX identifier or license description (spec field).
    compatibility:
        Environment requirements, max 500 chars (spec field).
    metadata:
        Free string-to-string map (spec field). Anchor reads the optional
        ``activation`` key from here for file-loaded skills.
    allowed_tools:
        Tool names this skill pre-approves (spec field, experimental).
    path:
        The skill directory on disk, set by the loader. Enables lazy
        ``references/`` reading and ``scripts/`` execution.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str
    instructions: str = ""
    tools: tuple[AgentTool, ...] = ()
    activation: Literal["always", "on_demand"] = "on_demand"
    tags: tuple[str, ...] = ()
    license: str = ""
    compatibility: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    path: Path | None = None

    def reference_files(self) -> list[str]:
        """List files under ``references/``, relative to the skill directory.

        Returns an empty list when the skill has no on-disk path or no
        references directory.
        """
        return self._list_dir("references")

    def script_files(self) -> list[str]:
        """List files under ``scripts/``, relative to the skill directory."""
        return self._list_dir("scripts")

    def _list_dir(self, subdir: str) -> list[str]:
        if self.path is None:
            return []
        base = self.path / subdir
        if not base.is_dir():
            return []
        return sorted(
            str(p.relative_to(self.path))
            for p in base.rglob("*")
            if p.is_file()
        )
