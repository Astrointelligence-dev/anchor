"""Anthropic Messages API formatter."""

from __future__ import annotations

from typing import Any

from anchor.models.context import ContextWindow

from .utils import classify_window_items, ensure_alternating_roles, get_message_role


class AnthropicFormatter:
    """Formats context for the Anthropic Messages API.

    Produces a dict with:
    - "system": list of content blocks (Anthropic API accepts both string and list)
    - "messages": list of message dicts

    Parameters
    ----------
    enable_caching:
        When ``True``, adds ``"cache_control": {"type": "ephemeral"}`` to the
        last system content block and the last retrieval/context message.  This
        enables Anthropic prompt caching to reduce latency and cost for
        repeated prefixes.

    Security Note:
        Content from memory and retrieval items is inserted verbatim into
        the formatted output without sanitization.  If these items originate
        from untrusted sources (e.g. user-supplied documents, web scrapes),
        they may contain prompt injection payloads.  Callers should implement
        content validation or filtering *before* items enter the pipeline.
    """

    def __init__(self, *, enable_caching: bool = False) -> None:
        self._enable_caching = enable_caching

    @property
    def format_type(self) -> str:
        return "anthropic"

    def format(self, window: ContextWindow) -> dict[str, Any]:
        """Format the context window for the Anthropic Messages API.

        Returns a dict with a ``"system"`` key (list of content blocks) and
        a ``"messages"`` list.  Retrieval/tool context is prepended as a
        ``user`` message before the conversation history.
        """
        classified = classify_window_items(window)

        # Build system content blocks — one block per system item for
        # finer-grained Anthropic prompt caching.  When caching is enabled,
        # ``cache_control`` is placed on the *last* system block so the
        # entire static prefix is cached up to that breakpoint.
        system_blocks: list[dict[str, Any]] = []
        for part in classified.system_parts:
            block: dict[str, Any] = {"type": "text", "text": part}
            system_blocks.append(block)
        if self._enable_caching and system_blocks:
            system_blocks[-1]["cache_control"] = {"type": "ephemeral"}

        # Build messages
        messages: list[dict[str, Any]] = []

        for item in classified.memory_items:
            role = get_message_role(item)
            messages.append({"role": role, "content": item.content})

        if classified.context_parts:
            context_block = "Here is relevant context:\n\n" + "\n\n---\n\n".join(
                classified.context_parts
            )
            messages.insert(0, {"role": "user", "content": context_block})

        # Enforce strict user/assistant alternation required by the
        # Anthropic Messages API.  The context block above is always
        # role="user"; if the first memory item is also role="user",
        # the two consecutive user messages would be rejected.
        messages = ensure_alternating_roles(messages)

        # Cache breakpoint on the context message, in valid wire format:
        # cache_control lives on a content block, never on the message
        # dict. Applied after merging (merging concatenates str content).
        if self._enable_caching and classified.context_parts and messages:
            first = messages[0]
            messages[0] = {
                "role": first["role"],
                "content": [
                    {
                        "type": "text",
                        "text": first["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }

        return {
            "system": system_blocks,
            "messages": messages,
        }
