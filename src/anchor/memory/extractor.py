"""Memory extraction from conversation turns.

``CallbackExtractor`` delegates to a user-provided function — the library
calls no model on its own. ``LLMExtractor`` is the opt-in alternative: the
extraction phase of the mem0 paper, one call to an injected provider.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from anchor._text import ask_json
from anchor.models.memory import MemoryEntry, MemoryType

if TYPE_CHECKING:
    from anchor.llm.base import LLMProvider
    from anchor.models.memory import ConversationTurn

logger = logging.getLogger(__name__)

_MEMORY_TYPES = {t.value for t in MemoryType}

_EXTRACTION_PROMPT = (
    "You extract durable facts about the user from a conversation excerpt: preferences, "
    "biographical details, decisions, constraints, relationships — what is worth knowing "
    "in a future conversation. Skip pleasantries, unanswered questions and anything only "
    "true for this moment. Write each fact as one self-contained sentence in the third "
    'person ("User lives in Rio"), resolving pronouns and relative dates from the excerpt. '
    "Do not repeat a fact.\n"
    "Return ONLY a JSON array, no prose:\n"
    '[{{"content": "User lives in Rio", "tags": ["location"], "memory_type": "semantic"}}]\n'
    "memory_type is one of: semantic (facts), episodic (events), procedural (how-tos). "
    "Return [] when nothing is worth remembering.\n\n"
    "Conversation:\n{conversation}\n"
)


class CallbackExtractor:
    """Delegates memory extraction to a user-provided function.

    The user function receives a list of conversation turns and returns
    a list of plain dictionaries. Each dictionary must contain at least
    a ``"content"`` key; optional keys include ``"tags"``,
    ``"memory_type"``, ``"metadata"``, ``"relevance_score"``,
    ``"user_id"``, ``"session_id"``, and any other ``MemoryEntry`` field.

    Example user function::

        def my_extractor(turns):
            return [
                {"content": "User prefers dark mode", "tags": ["preference"]},
                {"content": "User's name is Alice", "memory_type": "semantic"},
            ]
    """

    __slots__ = ("_default_type", "_extract_fn")

    def __init__(
        self,
        extract_fn: Callable[[list[ConversationTurn]], list[dict[str, Any]]],
        default_type: MemoryType = MemoryType.SEMANTIC,
    ) -> None:
        self._extract_fn = extract_fn
        self._default_type = default_type

    def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]:
        """Extract memory entries from conversation turns.

        Parameters:
            turns: Recent conversation turns to extract memories from.

        Returns:
            A list of ``MemoryEntry`` objects built from the user
            function's output dictionaries.

        Raises:
            ValueError: If a returned dictionary is missing the required
                ``"content"`` key.
        """
        raw_results = self._extract_fn(turns)
        entries: list[MemoryEntry] = []

        for raw_original in raw_results:
            raw = dict(raw_original)  # defensive copy
            if "content" not in raw:
                msg = "extraction result must contain a 'content' key"
                raise ValueError(msg)

            # Resolve memory_type: use provided string/enum or fall back to default
            memory_type_raw = raw.pop("memory_type", None)
            if memory_type_raw is not None:
                memory_type = MemoryType(memory_type_raw)
            else:
                memory_type = self._default_type

            # Build the source_turns list from turn timestamps if not provided
            source_turns: list[str] = raw.pop("source_turns", [])
            if not source_turns:
                source_turns = [t.timestamp.isoformat() for t in turns]

            entry = MemoryEntry(
                content=raw.pop("content"),
                memory_type=memory_type,
                source_turns=source_turns,
                **raw,
            )
            entries.append(entry)

        return entries


class LLMExtractor(CallbackExtractor):
    """Facts from recent turns in one model call — the mem0 extraction phase.

    Opt-in: the provider is injected (the agent's own), never a new client.
    Only turns whose role is in *roles* are read — tool turns are noise for
    facts about the user — and no call is made when none remain. Each fact
    becomes a ``MemoryEntry`` the way ``CallbackExtractor`` builds them. A
    malformed or failed response yields no entries and logs a warning, the
    fail-soft contract of ``LLMGraphExtractor``.
    """

    __slots__ = ("_llm", "_roles")

    def __init__(
        self,
        llm: LLMProvider,
        *,
        roles: Iterable[str] = ("user", "assistant"),
        default_type: MemoryType = MemoryType.SEMANTIC,
    ) -> None:
        super().__init__(self._ask, default_type)
        self._llm = llm
        self._roles = tuple(roles)

    def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]:
        kept = [t for t in turns if t.role in self._roles]
        return super().extract(kept) if kept else []

    def _ask(self, turns: list[ConversationTurn]) -> list[dict[str, Any]]:
        conversation = "\n".join(f"{t.role}: {t.content}" for t in turns)
        prompt = _EXTRACTION_PROMPT.format(conversation=conversation)
        data = ask_json(self._llm, prompt, log=logger, what="LLM memory extraction failed")
        if data is None:
            return []
        facts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in data:
            content = raw.get("content") if isinstance(raw, dict) else None
            if not isinstance(content, str):
                continue
            content = content.strip()
            if not content or content in seen:
                continue
            seen.add(content)
            fact: dict[str, Any] = {"content": content}
            tags = raw.get("tags")
            if isinstance(tags, list):
                fact["tags"] = [t for t in tags if isinstance(t, str)]
            memory_type = raw.get("memory_type")
            if isinstance(memory_type, str) and memory_type in _MEMORY_TYPES:
                fact["memory_type"] = memory_type
            facts.append(fact)
        return facts

    def __repr__(self) -> str:
        return f"LLMExtractor(llm={self._llm!r}, roles={self._roles!r})"
