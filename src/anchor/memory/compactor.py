"""LLM-driven tier compaction for progressive summarization.

``TierCompactor`` handles summarization and key-fact extraction by
calling an ``LLMProvider``. Each tier transition uses a tier-specific
prompt template. Errors fall back to raw content concatenation.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from anchor._text import strip_markdown_fences
from anchor.llm.models import Message, Role
from anchor.models.memory import FactType, KeyFact
from anchor.tokens.counter import get_default_counter

if TYPE_CHECKING:
    from anchor.llm.base import LLMProvider
    from anchor.protocols.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

_SUMMARIZE_PROMPTS: dict[int, str] = {
    1: (
        "Summarize the following conversation preserving all reasoning, "
        "decisions made, and important context. Be thorough but concise. "
        "Target approximately {target_tokens} tokens.\n\n{content}"
    ),
    2: (
        "Compress the following summary to key points only. Remove "
        "conversational noise and redundancy. Retain decisions, facts, "
        "and conclusions. Target approximately {target_tokens} tokens.\n\n{content}"
    ),
    3: (
        "Reduce the following to a single headline-level statement that "
        "captures the essential thread of the conversation. Target "
        "approximately {target_tokens} tokens.\n\n{content}"
    ),
}

_MERGE_PROMPT = (
    "You have an existing summary and new content to incorporate. "
    "Produce a unified, non-redundant summary that covers both. "
    "Target approximately {target_tokens} tokens.\n\n"
    "EXISTING SUMMARY:\n{existing_summary}\n\n"
    "NEW CONTENT:\n{new_content}"
)

_FACT_EXTRACTION_PROMPT = (
    'Extract key facts from the following content. Return a JSON array '
    'where each element has "type" (one of: decision, entity, number, '
    'date, preference, constraint) and "content" (the fact itself, concise).\n\n'
    "Only extract facts that would be important to remember if the original "
    "text were lost. Return [] if no key facts are found.\n\n{content}"
)

_FACT_RETRY_PROMPT = (
    "Return ONLY valid JSON. No markdown fences, no explanation. "
    "Just a JSON array of objects with 'type' and 'content' keys.\n\n{content}"
)

_VALID_FACT_TYPES = {ft.value for ft in FactType}


class TierCompactor:
    """Handles LLM calls for summarization and fact extraction."""

    __slots__ = ("_llm", "_tokenizer")

    def __init__(
        self,
        llm: LLMProvider,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._llm = llm
        self._tokenizer: Tokenizer = tokenizer or get_default_counter()

    def summarize(
        self,
        content: str,
        target_tier: int,
        target_tokens: int,
        existing_summary: str | None = None,
    ) -> str:
        """Synchronously summarize content for a target tier.

        Falls back to raw content (truncated) on LLM failure.
        """
        prompt = self._build_summarize_prompt(
            content, target_tier, target_tokens, existing_summary
        )
        try:
            response = self._llm.invoke(
                [Message(role=Role.USER, content=prompt)]
            )
            return response.content or content
        except Exception:
            logger.exception(
                "Summarization failed for tier %d; using raw content", target_tier
            )
            return self._tokenizer.truncate_to_tokens(content, target_tokens)

    async def asummarize(
        self,
        content: str,
        target_tier: int,
        target_tokens: int,
        existing_summary: str | None = None,
    ) -> str:
        """Asynchronously summarize content for a target tier."""
        prompt = self._build_summarize_prompt(
            content, target_tier, target_tokens, existing_summary
        )
        try:
            response = await self._llm.ainvoke(
                [Message(role=Role.USER, content=prompt)]
            )
            return response.content or content
        except Exception:
            logger.exception(
                "Async summarization failed for tier %d; using raw content",
                target_tier,
            )
            return self._tokenizer.truncate_to_tokens(content, target_tokens)

    def extract_facts(self, content: str, source_tier: int) -> list[KeyFact]:
        """Synchronously extract key facts from content."""
        return self._parse_facts(
            self._call_fact_extraction(content, sync=True), source_tier
        )

    async def aextract_facts(self, content: str, source_tier: int) -> list[KeyFact]:
        """Asynchronously extract key facts from content."""
        return await self._parse_facts_async(
            await self._call_fact_extraction_async(content), source_tier
        )

    # -- Private helpers --

    def _build_summarize_prompt(
        self,
        content: str,
        target_tier: int,
        target_tokens: int,
        existing_summary: str | None,
    ) -> str:
        if existing_summary is not None:
            return _MERGE_PROMPT.format(
                target_tokens=target_tokens,
                existing_summary=existing_summary,
                new_content=content,
            )
        template = _SUMMARIZE_PROMPTS.get(target_tier, _SUMMARIZE_PROMPTS[1])
        return template.format(target_tokens=target_tokens, content=content)

    def _call_fact_extraction(self, content: str, *, sync: bool = True) -> str:
        prompt = _FACT_EXTRACTION_PROMPT.format(content=content)
        try:
            response = self._llm.invoke([Message(role=Role.USER, content=prompt)])
            return response.content or "[]"
        except Exception:
            logger.warning("Fact extraction failed; skipping facts")
            return "[]"

    async def _call_fact_extraction_async(self, content: str) -> str:
        prompt = _FACT_EXTRACTION_PROMPT.format(content=content)
        try:
            response = await self._llm.ainvoke(
                [Message(role=Role.USER, content=prompt)]
            )
            return response.content or "[]"
        except Exception:
            logger.warning("Async fact extraction failed; skipping facts")
            return "[]"

    def _parse_facts(self, raw_json: str, source_tier: int) -> list[KeyFact]:
        """Parse JSON response into KeyFact objects, retrying once on failure."""
        for attempt in range(2):
            try:
                # Strip markdown fences if present
                data = json.loads(strip_markdown_fences(raw_json))
                if not isinstance(data, list):
                    return []

                facts: list[KeyFact] = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    fact_type_str = item.get("type", "")
                    if fact_type_str not in _VALID_FACT_TYPES:
                        continue
                    fact_content = item.get("content", "")
                    if not fact_content:
                        continue
                    token_count = self._tokenizer.count_tokens(fact_content)
                    facts.append(
                        KeyFact(
                            fact_type=FactType(fact_type_str),
                            content=fact_content,
                            source_tier=source_tier,
                            token_count=token_count,
                        )
                    )
                return facts
            except (json.JSONDecodeError, KeyError, TypeError):
                if attempt == 0:
                    logger.warning("JSON parse failed; retrying with stricter prompt")
                    try:
                        retry_prompt = _FACT_RETRY_PROMPT.format(content=raw_json)
                        response = self._llm.invoke(
                            [Message(role=Role.USER, content=retry_prompt)]
                        )
                        raw_json = response.content or "[]"
                    except Exception:
                        return []
                else:
                    logger.warning("Fact extraction JSON parse failed after retry; skipping")
                    return []
        return []

    async def _parse_facts_async(
        self, raw_json: str, source_tier: int
    ) -> list[KeyFact]:
        """Async variant of _parse_facts — uses ainvoke for retry."""
        for attempt in range(2):
            try:
                data = json.loads(strip_markdown_fences(raw_json))
                if not isinstance(data, list):
                    return []

                facts: list[KeyFact] = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    fact_type_str = item.get("type", "")
                    if fact_type_str not in _VALID_FACT_TYPES:
                        continue
                    fact_content = item.get("content", "")
                    if not fact_content:
                        continue
                    token_count = self._tokenizer.count_tokens(fact_content)
                    facts.append(
                        KeyFact(
                            fact_type=FactType(fact_type_str),
                            content=fact_content,
                            source_tier=source_tier,
                            token_count=token_count,
                        )
                    )
                return facts
            except (json.JSONDecodeError, KeyError, TypeError):
                if attempt == 0:
                    logger.warning("JSON parse failed; retrying with stricter prompt")
                    try:
                        retry_prompt = _FACT_RETRY_PROMPT.format(content=raw_json)
                        response = await self._llm.ainvoke(
                            [Message(role=Role.USER, content=retry_prompt)]
                        )
                        raw_json = response.content or "[]"
                    except Exception:
                        return []
                else:
                    logger.warning("Fact extraction JSON parse failed after retry; skipping")
                    return []
        return []
