"""OpenRouterProvider — OpenAI-compatible adapter for OpenRouter.

Thin subclass of OpenAIProvider that points at OpenRouter's API endpoint
and reads the OPENROUTER_API_KEY from the environment.

Self-registers via register_provider() at module import time.
"""

from __future__ import annotations

import os
from typing import Any

from anchor.llm.providers.openai import OpenAIProvider
from anchor.llm.registry import register_provider


class OpenRouterProvider(OpenAIProvider):
    """Adapter for OpenRouter's API (OpenAI-compatible)."""

    provider_name = "openrouter"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            base_url=base_url or "https://openrouter.ai/api/v1",
            # OpenRouter reports the billed cost (usage.cost) and cached/audio
            # token details only when asked; the caller's extra_body wins.
            extra_body={"usage": {"include": True}, **(extra_body or {})},
            **kwargs,
        )

    def _resolve_api_key(self) -> str | None:
        return os.environ.get("OPENROUTER_API_KEY")


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register_provider("openrouter", OpenRouterProvider)
