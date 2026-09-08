"""OpenRouterProvider — OpenAI-compatible adapter for OpenRouter.

Thin subclass of OpenAIProvider that points at OpenRouter's API endpoint
and reads the OPENROUTER_API_KEY from the environment.

Self-registers via register_provider() at module import time.
"""

from __future__ import annotations

import os

from anchor.llm.providers.openai import OpenAIProvider
from anchor.llm.registry import register_provider


class OpenRouterProvider(OpenAIProvider):
    """Adapter for OpenRouter's API (OpenAI-compatible)."""

    provider_name = "openrouter"

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(
            model=model,
            base_url=base_url or "https://openrouter.ai/api/v1",
            **kwargs,
        )

    def _resolve_api_key(self) -> str | None:
        return os.environ.get("OPENROUTER_API_KEY")


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register_provider("openrouter", OpenRouterProvider)
