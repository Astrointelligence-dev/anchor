"""Skill registry for progressive tool disclosure."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from anchor.agent.skills.models import Skill

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from anchor.agent.models import AgentTool


class SkillRegistry:
    """Registry that tracks registered skills and their activation state.

    Always-loaded skills are considered active from the moment they are
    registered.  On-demand skills must be explicitly activated via
    :meth:`activate` (typically by the ``activate_skill`` meta-tool).
    """

    __slots__ = ("_activated", "_skills")

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._activated: set[str] = set()

    # -- Mutation --

    def register(self, skill: Skill) -> None:
        """Register a skill.

        Raises :class:`ValueError` on a duplicate skill name, or when the
        skill provides a tool whose name collides with a tool from any
        already-registered skill — collisions surface here, at registration
        time, never mid-conversation.
        """
        if skill.name in self._skills:
            msg = f"Skill already registered: '{skill.name}'"
            raise ValueError(msg)

        existing_tools: dict[str, str] = {
            tool.name: existing.name
            for existing in self._skills.values()
            for tool in existing.tools
        }
        for tool in skill.tools:
            if tool.name in existing_tools:
                msg = (
                    f"Tool name collision: skill '{skill.name}' provides "
                    f"'{tool.name}', already provided by skill "
                    f"'{existing_tools[tool.name]}'"
                )
                raise ValueError(msg)

        self._skills[skill.name] = skill

    def activate(self, name: str) -> Skill:
        """Mark an on-demand skill as active.  Returns the skill.

        Raises :class:`KeyError` if the skill is not registered.
        """
        skill = self._skills.get(name)
        if skill is None:
            msg = f"Unknown skill: '{name}'"
            raise KeyError(msg)
        self._activated.add(name)
        return skill

    def deactivate(self, name: str) -> None:
        """Remove a skill from the active set."""
        self._activated.discard(name)

    def reset(self) -> None:
        """Clear all activation state (keeps registrations)."""
        self._activated.clear()

    # -- SKILL.md loading --

    def load_from_path(self, path: str | Path) -> Skill:
        """Load a SKILL.md skill from *path* and register it.

        Returns the loaded :class:`Skill`.
        """
        from anchor.agent.skills.loader import load_skill

        skill = load_skill(Path(path))
        self.register(skill)
        return skill

    def load_from_directory(self, path: str | Path) -> list[Skill]:
        """Load all SKILL.md skills from *path* and register them.

        Skips skills that fail to load or have duplicate names.
        """
        from anchor.agent.skills.loader import load_skills_directory

        loaded = load_skills_directory(Path(path))
        registered: list[Skill] = []
        for skill in loaded:
            try:
                self.register(skill)
                registered.append(skill)
            except ValueError as exc:
                logger.warning("Skipping skill '%s': %s", skill.name, exc)
        return registered

    # -- Queries --

    def get(self, name: str) -> Skill | None:
        """Look up a skill by name, or ``None`` if not found."""
        return self._skills.get(name)

    def all_skills(self) -> list[Skill]:
        """Return all registered skills in registration order."""
        return list(self._skills.values())

    def is_active(self, name: str) -> bool:
        """Return ``True`` if the skill's tools should be available now.

        Always-loaded skills are always active.  On-demand skills are
        active only after an explicit :meth:`activate` call.
        """
        skill = self._skills.get(name)
        if skill is None:
            return False
        if skill.activation == "always":
            return True
        return name in self._activated

    def active_tools(self) -> list[AgentTool]:
        """Return all tools from currently-active skills.

        Collisions are prevented at :meth:`register` time, so this never
        raises mid-conversation.
        """
        tools: list[AgentTool] = []
        for name, skill in self._skills.items():
            if self.is_active(name):
                tools.extend(skill.tools)
        return tools

    def on_demand_skills(self) -> list[Skill]:
        """Return skills that require activation."""
        return [s for s in self._skills.values() if s.activation == "on_demand"]

    def always_skills(self) -> list[Skill]:
        """Return skills active from round 1."""
        return [s for s in self._skills.values() if s.activation == "always"]

    def always_instructions(self) -> str:
        """Concatenated instructions of all ``always`` skills.

        Injected into the system prompt at build time — an always-on
        skill's usage guide must reach the model without an activation
        round-trip.  Returns an empty string when no always skill has
        instructions.
        """
        blocks: list[str] = []
        for skill in self.always_skills():
            if skill.instructions.strip():
                blocks.append(f"## Skill: {skill.name}\n{skill.instructions.strip()}")
        return "\n\n".join(blocks)

    def skill_discovery_prompt(self, max_chars: int | None = None) -> str:
        """Build the Level-1 discovery text for the system prompt.

        The text is *static* for a given set of registered skills — it never
        changes with activation state, so the system prompt stays stable
        across rounds and turns (prompt-cache friendly).

        Parameters
        ----------
        max_chars:
            Optional budget for the listing. Skills beyond the budget are
            summarized as a count; earlier-registered skills win.

        Returns an empty string when there are no on-demand skills.
        """
        on_demand = self.on_demand_skills()
        if not on_demand:
            return ""

        header = "Available skills (use activate_skill to enable):"
        lines = [header]
        dropped = 0
        used = len(header)
        for skill in on_demand:
            line = f"  - {skill.name}: {skill.description}"
            if max_chars is not None and used + len(line) + 1 > max_chars:
                dropped += 1
                continue
            lines.append(line)
            used += len(line) + 1
        if dropped:
            lines.append(f"  (+{dropped} more skills not listed)")
        return "\n".join(lines)
