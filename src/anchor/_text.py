"""Shared small text utilities."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anchor.llm.base import LLMProvider


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


def ask_json(
    llm: LLMProvider, prompt: str, *, expect: type = list, log: logging.Logger, what: str
) -> Any | None:
    """One model call whose answer is JSON: the parsed value, or ``None`` after a warning.

    The fail-soft contract shared by the memory extractor and consolidator
    and the LLM graph extractor: a provider error or an answer that is not
    the *expect* JSON type costs this call its result, never the caller's run.
    *what* is the phrase the warning opens with ("LLM memory extraction failed").
    """
    from anchor.llm.models import Message, Role

    try:
        response = llm.invoke([Message(role=Role.USER, content=prompt)])
        data = json.loads(strip_markdown_fences(response.content or ""))
        if not isinstance(data, expect):
            msg = f"response is not a JSON {expect.__name__}"
            raise TypeError(msg)
    except Exception as exc:
        log.warning("%s: %s", what, exc)
        return None
    return data
