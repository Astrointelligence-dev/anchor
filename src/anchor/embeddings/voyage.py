"""Voyage AI embedding provider (voyage-3.5 / voyage-4 family).

Requires the ``voyage`` extra: ``pip install astro-anchor[voyage]``.
"""

from __future__ import annotations

_MODEL_DIMENSIONS = {
    "voyage-4-large": 1024,
    "voyage-4": 1024,
    "voyage-4-lite": 1024,
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-code-4": 1024,
}


class VoyageEmbeddingProvider:
    """Embeddings via the Voyage AI API.

    Uses Voyage's native query/document asymmetry (``input_type``) and
    Matryoshka truncation (``output_dimension``).

    Parameters
    ----------
    model:
        Voyage embedding model name.
    dimensions:
        Optional output dimensionality (256/512/1024/2048 depending on model).
    api_key:
        Optional API key; falls back to ``VOYAGE_API_KEY``.
    """

    __slots__ = ("_async_client", "_client", "_dimensions", "_model", "_output_dim")

    def __init__(
        self,
        model: str = "voyage-3.5",
        *,
        dimensions: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._output_dim = dimensions
        self._dimensions = dimensions or _MODEL_DIMENSIONS.get(model, 0)
        try:
            import voyageai  # type: ignore[import-not-found]
        except ImportError as e:
            msg = (
                "voyageai is required for VoyageEmbeddingProvider. "
                "Install it with: pip install astro-anchor[voyage]"
            )
            raise ImportError(msg) from e
        self._client = voyageai.Client(api_key=api_key)
        self._async_client = voyageai.AsyncClient(api_key=api_key)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r}, dimensions={self._dimensions})"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        kwargs = {"output_dimension": self._output_dim} if self._output_dim else {}
        result = self._client.embed(
            texts, model=self._model, input_type=input_type, **kwargs
        )
        return list(result.embeddings)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    async def _aembed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        kwargs = {"output_dimension": self._output_dim} if self._output_dim else {}
        result = await self._async_client.embed(
            texts, model=self._model, input_type=input_type, **kwargs
        )
        return list(result.embeddings)

    async def aembed_query(self, text: str) -> list[float]:
        return (await self._aembed([text], "query"))[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._aembed(texts, "document")
