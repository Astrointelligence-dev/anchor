"""SKILL.md loader for the Agent Skills open standard.

Parses SKILL.md files (YAML frontmatter + markdown body, per
https://agentskills.io/specification) into native Skill instances, with
optional tool discovery from tools.py in the same directory.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import yaml

from anchor.agent.skills.models import Skill

if TYPE_CHECKING:
    from anchor.agent.models import AgentTool

logger = logging.getLogger(__name__)

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1024
_MAX_COMPATIBILITY_LENGTH = 500

# Spec fields (agentskills.io) + anchor extensions accepted at top level.
_KNOWN_KEYS = frozenset(
    {
        "name",
        "description",
        "license",
        "compatibility",
        "metadata",
        "allowed-tools",
        # anchor extensions (also accepted under metadata)
        "activation",
        "tags",
    }
)

_VALID_ACTIVATIONS = ("always", "on_demand")


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split SKILL.md content into a YAML frontmatter dict and markdown body.

    Frontmatter is delimited by ``---`` lines at the start of the file and
    parsed with a real YAML parser (multi-line values, quoting, and nested
    maps all work).  Returns ``(frontmatter_dict, body_text)``.

    Raises ``ValueError`` when the frontmatter is not valid YAML or not a
    mapping.
    """
    stripped = text.strip()
    if not stripped.startswith("---"):
        return {}, stripped

    lines = stripped.splitlines()
    end_line = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_line = i
            break

    if end_line is None:
        return {}, stripped

    raw_fm = "\n".join(lines[1:end_line])
    body = "\n".join(lines[end_line + 1 :]).strip()

    try:
        parsed = yaml.safe_load(raw_fm)
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML frontmatter in SKILL.md: {exc}"
        raise ValueError(msg) from exc

    if parsed is None:
        return {}, body
    if not isinstance(parsed, dict):
        msg = f"SKILL.md frontmatter must be a YAML mapping, got {type(parsed).__name__}"
        raise ValueError(msg)
    return parsed, body


def _parse_tags(raw: Any) -> tuple[str, ...]:
    """Normalize tags from a YAML list or comma-separated string."""
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(t).strip() for t in raw if str(t).strip())
    raw = str(raw).strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    return tuple(t.strip() for t in raw.split(",") if t.strip())


def _parse_allowed_tools(raw: Any) -> tuple[str, ...]:
    """Normalize allowed-tools: space-separated string (spec) or YAML list."""
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(t).strip() for t in raw if str(t).strip())
    return tuple(t for t in str(raw).split() if t)


def _parse_metadata(raw: Any) -> dict[str, str]:
    """Normalize the spec ``metadata`` map to string -> string."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        msg = f"SKILL.md 'metadata' must be a mapping, got {type(raw).__name__}"
        raise ValueError(msg)
    return {str(k): str(v) for k, v in raw.items()}


def _validate_name(name: str, skill_dir: Path) -> None:
    """Validate skill name per spec: format, length, matches directory name."""
    if not name:
        msg = "SKILL.md frontmatter missing required field: 'name'"
        raise ValueError(msg)
    if len(name) > _MAX_NAME_LENGTH:
        msg = f"Skill name exceeds {_MAX_NAME_LENGTH} characters: '{name}'"
        raise ValueError(msg)
    if not _NAME_PATTERN.match(name):
        msg = (
            f"Invalid skill name '{name}': must be lowercase letters, "
            "digits, and hyphens only (no leading/trailing/consecutive hyphens)"
        )
        raise ValueError(msg)
    if name != skill_dir.name:
        msg = (
            f"Skill name '{name}' must match its directory name "
            f"'{skill_dir.name}' (agentskills.io spec)"
        )
        raise ValueError(msg)


def _validate_description(description: str) -> None:
    """Validate description is present and within length limit."""
    if not description:
        msg = "SKILL.md frontmatter missing required field: 'description'"
        raise ValueError(msg)
    if len(description) > _MAX_DESCRIPTION_LENGTH:
        msg = f"Skill description exceeds {_MAX_DESCRIPTION_LENGTH} characters"
        raise ValueError(msg)


def _resolve_activation(fm: dict[str, Any], metadata: dict[str, str], name: str) -> str:
    """Resolve activation mode: ``metadata.activation`` (canonical anchor
    extension) wins over a legacy top-level ``activation`` key; defaults to
    ``on_demand``.
    """
    raw = metadata.get("activation") or fm.get("activation") or "on_demand"
    raw = str(raw)
    if raw not in _VALID_ACTIVATIONS:
        logger.warning(
            "Invalid activation '%s' in SKILL.md for '%s': must be one of %s, "
            "defaulting to 'on_demand'",
            raw,
            name,
            _VALID_ACTIVATIONS,
        )
        raw = "on_demand"
    return raw


def _discover_tools(skill_dir: Path, skill_name: str) -> tuple[AgentTool, ...]:
    """Import tools.py from skill directory and collect AgentTool instances.

    This is an anchor extension for trusted, Python-authored skills — it
    executes arbitrary code at load time. Spec-portable skills should ship
    ``scripts/`` (executed on demand via ``run_skill_script``) instead.
    """
    from anchor.agent.models import AgentTool as AgentToolCls

    tools_path = skill_dir / "tools.py"
    if not tools_path.exists():
        return ()

    # Use path hash in module name to prevent collisions when two directories
    # define skills with the same name (e.g. during load-then-reject flows).
    path_hash = hex(hash(str(tools_path)))[-8:]
    module_name = f"anchor.skills.{skill_name}.{path_hash}.tools"
    logger.info("Loading tools from %s as %s", tools_path, module_name)

    # Remove any previously-cached version so reloads pick up changes.
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, tools_path)
    if spec is None or spec.loader is None:
        msg = f"Cannot load module spec from {tools_path}"
        raise ValueError(msg)

    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except (ImportError, SyntaxError, AttributeError, TypeError) as exc:
        # Clean up partial registration on import-related failures.
        sys.modules.pop(module_name, None)
        msg = f"Failed to import tools for skill '{skill_name}': {exc}"
        raise ValueError(msg) from exc
    except Exception as exc:
        # Unexpected error — still clean up, but preserve the original type.
        sys.modules.pop(module_name, None)
        msg = f"Failed to import tools for skill '{skill_name}': {exc}"
        raise ValueError(msg) from exc

    tools: list[AgentToolCls] = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, AgentToolCls):
            tools.append(attr)
    return tuple(tools)


def load_skill(path: str | Path) -> Skill:
    """Load a single SKILL.md directory into a Skill instance.

    Parameters
    ----------
    path:
        Path to a directory containing a ``SKILL.md`` file.

    Raises
    ------
    FileNotFoundError
        If *path* or ``SKILL.md`` does not exist.
    ValueError
        If frontmatter is invalid or ``tools.py`` import fails.
    """
    skill_dir = Path(path).resolve()
    if not skill_dir.is_dir():
        msg = f"Skill directory not found: {skill_dir}"
        raise FileNotFoundError(msg)

    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        msg = f"SKILL.md not found in {skill_dir}"
        raise FileNotFoundError(msg)

    text = skill_file.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)

    unknown = set(fm) - _KNOWN_KEYS
    if unknown:
        logger.warning(
            "SKILL.md in %s has unknown frontmatter keys (ignored): %s",
            skill_dir.name,
            ", ".join(sorted(unknown)),
        )

    name = str(fm.get("name", "") or "")
    _validate_name(name, skill_dir)

    description = str(fm.get("description", "") or "").strip()
    _validate_description(description)

    compatibility = str(fm.get("compatibility", "") or "").strip()
    if len(compatibility) > _MAX_COMPATIBILITY_LENGTH:
        msg = f"Skill compatibility exceeds {_MAX_COMPATIBILITY_LENGTH} characters"
        raise ValueError(msg)

    metadata = _parse_metadata(fm.get("metadata"))
    activation = cast(
        Literal["always", "on_demand"], _resolve_activation(fm, metadata, name)
    )

    tags = _parse_tags(fm.get("tags"))
    tools = _discover_tools(skill_dir, name)

    return Skill(
        name=name,
        description=description,
        instructions=body,
        tools=tools,
        activation=activation,
        tags=tags,
        license=str(fm.get("license", "") or "").strip(),
        compatibility=compatibility,
        metadata=metadata,
        allowed_tools=_parse_allowed_tools(fm.get("allowed-tools")),
        path=skill_dir,
    )


def load_skills_directory(path: str | Path) -> list[Skill]:
    """Scan a directory for ``*/SKILL.md`` patterns and load all skills.

    Parameters
    ----------
    path:
        Path to a directory containing skill subdirectories.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    skills_dir = Path(path).resolve()
    if not skills_dir.is_dir():
        msg = f"Skills directory not found: {skills_dir}"
        raise FileNotFoundError(msg)

    skill_files = sorted(skills_dir.glob("*/SKILL.md"))
    if not skill_files:
        logger.warning("No SKILL.md files found in %s", skills_dir)
        return []

    skills: list[Skill] = []
    for skill_file in skill_files:
        skill_dir = skill_file.parent
        try:
            skill = load_skill(skill_dir)
            skills.append(skill)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Skipping skill in %s: %s", skill_dir.name, exc)
    return skills
