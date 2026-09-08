"""OllamaProvider — OpenAI-compatible adapter for local Ollama models.

Thin subclass of OpenAIProvider that points at the local Ollama endpoint
(http://localhost:11434/v1) and uses "ollama" as the default API key since
Ollama does not require authentication.

Reads OLLAMA_API_KEY from the environment if set, otherwise falls back to the
string "ollama".

Self-registers via register_provider() at module import time.
"""

from __future__ import annotations

import os

from anchor.llm.providers.openai import OpenAIProvider
from anchor.llm.registry import register_provider


class OllamaProvider(OpenAIProvider):
    """Adapter for local Ollama models (OpenAI-compatible API)."""

    provider_name = "ollama"

    def __init__(self, model: str, **kwargs):
        kwargs.setdefault("base_url", "http://localhost:11434/v1")
        super().__init__(model=model, **kwargs)

    def _resolve_api_key(self) -> str | None:
        return os.environ.get("OLLAMA_API_KEY", "ollama")


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

register_provider("ollama", OllamaProvider)
