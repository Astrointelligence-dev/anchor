"""Embedding provider protocol definitions.

Any object exposing these methods can produce vectors for anchor's dense
retrieval — no inheritance required. The interface carries the two
distinctions every 2026 embedding API exposes: query/document asymmetry
(instruction-tuned models embed queries and documents differently) and an
output-dimension parameter (Matryoshka truncation), fixed per instance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding backends (API or local models).

    Implementations must embed queries and documents through separate
    methods — instruction-tuned models (E5, Qwen3-Embedding, Voyage)
    apply different prompts to each side, worth 1-5% retrieval quality.
    Batching happens inside ``embed_documents`` so backends can use their
    native batch endpoints.
    """

    @property
    def dimensions(self) -> int:
        """Output dimensionality of every vector this provider returns."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents for indexing."""
        ...

    async def aembed_query(self, text: str) -> list[float]:
        """Async counterpart of :meth:`embed_query`."""
        ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async counterpart of :meth:`embed_documents`."""
        ...
