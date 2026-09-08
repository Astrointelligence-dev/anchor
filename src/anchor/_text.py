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
    llm: LLMProvider,
    prompt: str,
    *,
    expect: type = list,
    log: logging.Logger,
    what: str,
    fail_soft: bool = True,
) -> Any | None:
    """One model call whose answer is JSON: the parsed value, or ``None`` after a warning.

    The contract shared by the memory extractor and consolidator and the LLM
    graph extractor: an answer that is not the *expect* JSON type costs this
    call its result, never the caller's run. A provider error does the same
    when *fail_soft* is true; with ``fail_soft=False`` it propagates, for
    callers that must retry rather than lose the input. *what* is the phrase
    the warning opens with ("LLM memory extraction failed").
    """
    from anchor.llm.models import Message, Role

    try:
        response = llm.invoke([Message(role=Role.USER, content=prompt)])
    except Exception as exc:
        if not fail_soft:
            raise
        log.warning("%s: %s", what, exc)
        return None
    text = strip_markdown_fences(response.content or "")
    try:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = json.loads(_outermost(text, expect))  # prose around the value
        if not isinstance(data, expect):
            msg = f"response is not a JSON {expect.__name__}"
            raise TypeError(msg)
    except Exception as exc:
        log.warning("%s: %s", what, exc)
        return None
    return data


def _outermost(text: str, expect: type) -> str:
    """The slice from the first opening bracket to the last closing one for *expect*."""
    opening, closing = ("[", "]") if expect is list else ("{", "}")
    start, end = text.find(opening), text.rfind(closing)
    return text[start : end + 1] if 0 <= start < end else text
