"""Adapter that lifts a bare embedding callable into the provider protocol."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class CallableEmbeddingProvider:
    """Wrap a user-supplied embedding callable as an :class:`EmbeddingProvider`.

    Bridges the legacy ``embed_fn`` shapes to the provider protocol so all
    retrieval paths speak one interface. The callable may be sync or async
    and may take a single string; batching falls back to a per-text loop.

    Parameters
    ----------
    embed_fn:
        ``(str) -> list[float]`` — sync single-text embedder.
    aembed_fn:
        Optional ``(str) -> Awaitable[list[float]]`` async embedder. When
        omitted, async calls run ``embed_fn`` in a thread.
    batch_fn:
        Optional ``(list[str]) -> list[list[float]]`` native batch embedder.
        When omitted, batches loop over ``embed_fn``.
    """

    __slots__ = ("_aembed_fn", "_batch_fn", "_dimensions", "_embed_fn")

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]] | None = None,
        *,
        aembed_fn: Callable[[str], Awaitable[list[float]]] | None = None,
        batch_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        if embed_fn is None and aembed_fn is None:
            msg = "Provide at least one of embed_fn or aembed_fn"
            raise ValueError(msg)
        self._embed_fn = embed_fn
        self._aembed_fn = aembed_fn
        self._batch_fn = batch_fn
        self._dimensions: int | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(dimensions={self._dimensions})"

    @property
    def dimensions(self) -> int:
        """Dimensionality, learned from the first embedded text (0 if unknown)."""
        return self._dimensions or 0

    def _remember_dim(self, vector: list[float]) -> list[float]:
        if self._dimensions is None and vector:
            self._dimensions = len(vector)
        return vector

    def _require_sync(self) -> Callable[[str], list[float]]:
        if self._embed_fn is None:
            msg = "This provider was built with only an async embed_fn; use aembed_*"
            raise TypeError(msg)
        return self._embed_fn

    def embed_query(self, text: str) -> list[float]:
        return self._remember_dim(self._require_sync()(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._batch_fn is not None:
            vectors = self._batch_fn(texts)
            for v in vectors:
                self._remember_dim(v)
            return vectors
        fn = self._require_sync()
        return [self._remember_dim(fn(t)) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        if self._aembed_fn is not None:
            return self._remember_dim(await self._aembed_fn(text))
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._aembed_fn is not None:
            vectors: list[list[float]] = []
            for text in texts:
                vectors.append(self._remember_dim(await self._aembed_fn(text)))
            return vectors
        return await asyncio.to_thread(self.embed_documents, texts)


def as_embedding_provider(candidate: Any) -> Any:
    """Coerce *candidate* to an embedding provider.

    Accepts an object already satisfying the protocol (returned as-is) or a
    bare sync callable (wrapped). Returns ``None`` for ``None``.
    """
    if candidate is None:
        return None
    if callable(candidate) and not hasattr(candidate, "embed_query"):
        return CallableEmbeddingProvider(candidate)
    return candidate
