"""Cross-modal retrieval with shared embedding spaces.

Provides a cross-modal encoder that maps different modalities (text, image,
audio, etc.) into a shared vector space, and a retriever that performs
similarity search across modalities.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from anchor._math import cosine_similarity as _cosine_sim
from anchor.models.context import ContextItem
from anchor.models.query import QueryBundle

logger = logging.getLogger(__name__)


class CrossModalEncoder:
    """Encodes content from multiple modalities into a shared vector space.

    Each modality has its own encoder callback.  The resulting embeddings
    should live in the same vector space so that cross-modal similarity
    search is meaningful.

    Parameters:
        encoders: Mapping from modality name to encoder callable.
            Each callable takes content of the appropriate type and
            returns a list of floats (the embedding).
    """

    __slots__ = ("_encoders",)

    def __init__(self, encoders: dict[str, Callable[[Any], list[float]]]) -> None:
        self._encoders = encoders

    def encode(self, content: Any, modality: str) -> list[float]:
        """Encode content using the specified modality encoder.

        Parameters:
            content: The content to encode (type depends on the modality).
            modality: The modality name (e.g. "text", "image", "audio").

        Returns:
            An embedding vector as a list of floats.

        Raises:
            ValueError: If the modality is not registered.
        """
        if modality not in self._encoders:
            msg = f"Unknown modality {modality!r}. Available modalities: {sorted(self._encoders)}"
            raise ValueError(msg)
        return self._encoders[modality](content)

    @property
    def modalities(self) -> list[str]:
        """Return available modality names.

        Returns:
            A sorted list of registered modality names.
        """
        return sorted(self._encoders)

    def __repr__(self) -> str:
        return f"CrossModalEncoder(modalities={sorted(self._encoders)})"


class SharedSpaceRetriever:
    """Retriever that searches across modalities in a shared embedding space.

    Items are indexed with their modality-specific encoder and queries
    are encoded using the ``query_modality`` encoder.  Similarity is
    computed in the shared space, enabling cross-modal retrieval.

    Implements the ``Retriever`` protocol.

    Parameters:
        encoder: A cross-modal encoder for embedding content.
        query_modality: The modality to use for encoding queries.
        similarity_fn: Optional similarity function.  Defaults to cosine similarity.
    """

    __slots__ = ("_default_modality", "_encoder", "_items", "_similarity_fn")

    def __init__(
        self,
        encoder: CrossModalEncoder,
        query_modality: str = "text",
        similarity_fn: Callable[[list[float], list[float]], float] | None = None,
    ) -> None:
        self._encoder = encoder
        self._default_modality = query_modality
        self._similarity_fn = similarity_fn if similarity_fn is not None else _cosine_sim
        self._items: list[tuple[ContextItem, list[float]]] = []

    def index(self, items: list[ContextItem], modality: str | None = None) -> None:
        """Index items into the shared embedding space.

        Parameters:
            items: Context items to index.
            modality: The modality to use for encoding all items.
                If ``None``, each item's ``metadata["modality"]`` is used,
                defaulting to ``"text"`` when the key is absent.
        """
        for item in items:
            item_modality = modality or item.metadata.get("modality", "text")
            embedding = self._encoder.encode(item.content, item_modality)
            self._items.append((item, embedding))

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        """Retrieve items most similar to the query across modalities.

        Parameters:
            query: The query bundle containing the user's query text.
            top_k: Maximum number of items to return.

        Returns:
            A list of ``ContextItem`` objects sorted by similarity descending.
        """
        if not self._items:
            return []

        query_embedding = self._encoder.encode(query.query_str, self._default_modality)

        scored: list[tuple[float, ContextItem]] = []
        for item, embedding in self._items:
            sim = self._similarity_fn(query_embedding, embedding)
            scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def __repr__(self) -> str:
        return (
            f"SharedSpaceRetriever(encoder={self._encoder!r}, "
            f"query_modality={self._default_modality!r}, "
            f"indexed_items={len(self._items)})"
        )
