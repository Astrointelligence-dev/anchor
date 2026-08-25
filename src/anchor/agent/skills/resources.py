"""Level-3 progressive disclosure meta-tools.

``read_skill_file`` reads bundled files (``references/``, ``assets/``) lazily
so they cost zero context until actually needed. ``run_skill_script``
*executes* ``scripts/`` files instead of loading them — only their output
enters context — and is opt-in because it runs code from the skill directory.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

from anchor.agent.models import AgentTool
from anchor.agent.tool_decorator import tool

if TYPE_CHECKING:
    from anchor.agent.skills.registry import SkillRegistry

_MAX_FILE_CHARS = 50_000
_MAX_OUTPUT_CHARS = 10_000
_SCRIPT_TIMEOUT_SECONDS = 60.0


def _resolve_in_skill(registry: SkillRegistry, skill_name: str, file_path: str):
    """Resolve *file_path* inside the named skill's directory.

    Returns ``(resolved_path, error_message)`` — exactly one is non-None.
    Rejects paths that escape the skill directory (traversal, symlinks).
    """
    skill = registry.get(skill_name)
    if skill is None:
        names = [s.name for s in registry.all_skills() if s.path is not None]
        return None, (
            f"Unknown skill: '{skill_name}'. "
            f"Skills with files: {', '.join(names) or 'none'}"
        )
    if skill.path is None:
        return None, f"Skill '{skill_name}' has no on-disk directory."

    base = skill.path.resolve()
    resolved = (base / file_path).resolve()
    if base != resolved and base not in resolved.parents:
        return None, f"Path '{file_path}' escapes the skill directory."
    if not resolved.is_file():
        available = skill.reference_files() + skill.script_files()
        return None, (
            f"File not found: '{file_path}'. "
            f"Available files: {', '.join(available) or 'none'}"
        )
    return resolved, None


def _make_read_skill_file_tool(registry: SkillRegistry) -> AgentTool:
    """Create the ``read_skill_file`` meta-tool bound to *registry*."""

    @tool(
        name="read_skill_file",
        description=(
            "Read a bundled file from a skill's directory (e.g. "
            "references/guide.md). Use this to load skill reference "
            "material on demand instead of guessing."
        ),
    )
    def read_skill_file(skill_name: str, file_path: str) -> str:
        """Read a file bundled with a skill.

        Args:
            skill_name: Name of the skill that bundles the file.
            file_path: Path relative to the skill directory, e.g. 'references/api.md'.
        """
        resolved, error = _resolve_in_skill(registry, skill_name, file_path)
        if error:
            return error
        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Failed to read '{file_path}': {exc}"
        if len(content) > _MAX_FILE_CHARS:
            return (
                content[:_MAX_FILE_CHARS]
                + f"\n\n[Truncated: file is {len(content)} chars, "
                f"showing first {_MAX_FILE_CHARS}]"
            )
        return content

    return read_skill_file


def _make_run_skill_script_tool(registry: SkillRegistry) -> AgentTool:
    """Create the ``run_skill_script`` meta-tool bound to *registry*.

    Only wired into the agent when script execution was explicitly enabled
    (``allow_scripts=True``) — scripts are arbitrary code.
    """

    @tool(
        name="run_skill_script",
        description=(
            "Execute a script bundled with a skill (scripts/ directory) and "
            "return its output. Python scripts run with the current "
            "interpreter; other files must be executable."
        ),
    )
    def run_skill_script(skill_name: str, script_path: str, args: str = "") -> str:
        """Run a skill's bundled script and return stdout/stderr.

        Args:
            skill_name: Name of the skill that bundles the script.
            script_path: Path relative to the skill directory, e.g. 'scripts/convert.py'.
            args: Optional space-separated arguments passed to the script.
        """
        resolved, error = _resolve_in_skill(registry, skill_name, script_path)
        if error:
            return error

        skill = registry.get(skill_name)
        if skill is None or skill.path is None:  # unreachable after _resolve_in_skill
            return f"Skill '{skill_name}' has no on-disk directory."

        argv = args.split() if args else []
        if resolved.suffix == ".py":
            cmd = [sys.executable, str(resolved), *argv]
        else:
            cmd = [str(resolved), *argv]

        try:
            proc = subprocess.run(  # noqa: S603 -- opt-in by construction
                cmd,
                cwd=skill.path,
                capture_output=True,
                text=True,
                timeout=_SCRIPT_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"Script '{script_path}' timed out after {_SCRIPT_TIMEOUT_SECONDS}s."
        except OSError as exc:
            return f"Failed to execute '{script_path}': {exc}"

        parts = [f"exit code: {proc.returncode}"]
        if proc.stdout:
            parts.append(f"stdout:\n{proc.stdout}")
        if proc.stderr:
            parts.append(f"stderr:\n{proc.stderr}")
        output = "\n".join(parts)
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n[Output truncated]"
        return output

    return run_skill_script
