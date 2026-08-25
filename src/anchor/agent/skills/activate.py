"""Meta-tool for on-demand skill activation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anchor.agent.models import AgentTool
from anchor.agent.tool_decorator import tool

if TYPE_CHECKING:
    from anchor.agent.skills.registry import SkillRegistry


def _make_activate_skill_tool(registry: SkillRegistry) -> AgentTool:
    """Create the ``activate_skill`` meta-tool bound to *registry*.

    When called, this tool activates an on-demand skill and returns
    its instructions plus the names of newly available tools so the
    agent knows what it can call in subsequent rounds.
    """

    @tool(
        name="activate_skill",
        description=(
            "Activate an on-demand skill to make its tools available. "
            "Call this with the skill name from the available skills list."
        ),
    )
    def activate_skill(skill_name: str) -> str:
        """Activate an on-demand skill.

        Args:
            skill_name: Name of the skill to activate.
        """
        try:
            skill = registry.activate(skill_name)
        except KeyError:
            available = [s.name for s in registry.on_demand_skills()]
            return (
                f"Unknown skill: '{skill_name}'. "
                f"Available skills: {', '.join(available) or 'none'}"
            )

        tool_names = [t.name for t in skill.tools]
        parts = [f"Skill '{skill.name}' activated."]
        if skill.instructions:
            parts.append(f"\n{skill.instructions}")
        if tool_names:
            parts.append(f"\nNew tools available: {', '.join(tool_names)}")
        references = skill.reference_files()
        if references:
            parts.append(
                "\nBundled reference files (load on demand with "
                f"read_skill_file): {', '.join(references)}"
            )
        scripts = skill.script_files()
        if scripts:
            parts.append(f"\nBundled scripts: {', '.join(scripts)}")
        return "\n".join(parts)

    return activate_skill
