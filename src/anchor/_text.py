"""Shared small text utilities."""

from __future__ import annotations


def strip_markdown_fences(text: str) -> str:
    """Strip a surrounding markdown code fence (```lang ... ```), if any.

    Drops the opening fence line and, only when actually present, the
    closing fence line. Non-fenced text is returned stripped.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines[1:])
    return stripped.strip()
