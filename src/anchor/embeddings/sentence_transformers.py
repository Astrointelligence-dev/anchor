"""Local embedding provider via sentence-transformers.

Requires the ``local-embeddings`` extra:
``pip install astro-anchor[local-embeddings]``.

Default model is BGE-M3 — the open-source multilingual workhorse (100+
languages, 8192-token context), a strong choice for Portuguese corpora.
"""

from __future__ import annotations

import asyncio
from typing import Any


class SentenceTransformerEmbeddingProvider:
    """Embeddings from a local sentence-transformers model.

    The model loads lazily on first use. Query/document asymmetry is
    supported through optional prefixes (E5-style ``query:``/``passage:``)
    or the model's built-in prompts.

    Parameters
    ----------
    model:
        Hugging Face model id.
    query_prefix / document_prefix:
        Optional strings prepended to queries / documents before encoding
        (e.g. ``"query: "`` and ``"passage: "`` for E5-family models).
        BGE-M3 needs none.
    normalize:
        L2-normalize output vectors (default True — makes dot == cosine).
    """

    __slots__ = (
        "_dimensions",
        "_document_prefix",
        "_model",
        "_model_name",
        "_normalize",
        "_query_prefix",
    )

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        *,
        query_prefix: str = "",
        document_prefix: str = "",
        normalize: bool = True,
    ) -> None:
        self._model_name = model
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._normalize = normalize
        self._model: Any = None
        self._dimensions = 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model_name!r})"

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import (  # type: ignore[import-not-found]
                    SentenceTransformer,
                )
            except ImportError as e:
                msg = (
                    "sentence-transformers is required for "
                    "SentenceTransformerEmbeddingProvider. Install it with: "
                    "pip install astro-anchor[local-embeddings]"
                )
                raise ImportError(msg) from e
            self._model = SentenceTransformer(self._model_name)
            self._dimensions = int(self._model.get_sentence_embedding_dimension() or 0)
        return self._model

    @property
    def dimensions(self) -> int:
        if self._dimensions == 0:
            self._get_model()
        return self._dimensions

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        prefixed = [prefix + t for t in texts] if prefix else texts
        vectors = model.encode(
            prefixed, normalize_embeddings=self._normalize, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self._query_prefix)[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, self._document_prefix)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)
