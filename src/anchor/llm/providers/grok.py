"""GrokProvider — OpenAI-compatible adapter for xAI's Grok models.

Thin subclass of OpenAIProvider that points at xAI's API endpoint and
reads the XAI_API_KEY (or GROK_API_KEY as fallback) from the environment.

Self-registers via register_provider() at module import time.
"""

from __future__ import annotations

import os

from anchor.llm.providers.openai import OpenAIProvider
from anchor.llm.registry import register_provider


class GrokProvider(OpenAIProvider):
    """Adapter for xAI's Grok API (OpenAI-compatible)."""

    provider_name = "grok"

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(
            model=model,
            base_url=base_url or "https://api.x.ai/v1",
            **kwargs,
        )

    def _resolve_api_key(self) -> str | None:
        return os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register_provider("grok", GrokProvider)
