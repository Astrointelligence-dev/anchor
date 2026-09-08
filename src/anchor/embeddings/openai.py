"""OpenAI embedding provider (text-embedding-3 family).

Requires the ``openai`` extra: ``pip install astro-anchor[openai]``.
"""

from __future__ import annotations

from typing import Any

_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider:
    """Embeddings via the OpenAI API.

    Batches natively (one API call per ``embed_documents``). Supports
    Matryoshka truncation through the ``dimensions`` parameter on the
    text-embedding-3 models.

    Parameters
    ----------
    model:
        OpenAI embedding model name.
    dimensions:
        Optional output dimensionality (Matryoshka truncation). Defaults
        to the model's native size.
    api_key:
        Optional API key; falls back to ``OPENAI_API_KEY``.
    """

    __slots__ = ("_async_client", "_client", "_dimensions", "_kwargs", "_model")

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        dimensions: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions or _MODEL_DIMENSIONS.get(model, 0)
        self._kwargs: dict[str, Any] = {}
        if dimensions is not None:
            self._kwargs["dimensions"] = dimensions
        try:
            import openai
        except ImportError as e:
            msg = (
                "openai is required for OpenAIEmbeddingProvider. "
                "Install it with: pip install astro-anchor[openai]"
            )
            raise ImportError(msg) from e
        self._client = openai.OpenAI(api_key=api_key)
        self._async_client = openai.AsyncOpenAI(api_key=api_key)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r}, dimensions={self._dimensions})"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._model, input=texts, **self._kwargs
        )
        return [item.embedding for item in response.data]

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._async_client.embeddings.create(
            model=self._model, input=texts, **self._kwargs
        )
        return [item.embedding for item in response.data]
