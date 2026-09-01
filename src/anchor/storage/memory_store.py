"""In-memory storage implementations for development and testing.

These are the default backends -- no external dependencies needed.
Production users provide their own implementations (Redis, Postgres, etc.)
that satisfy the storage protocols.
"""

from __future__ import annotations

import heapq
import logging
import threading
from typing import Any

from anchor._math import cosine_similarity
from anchor.models.context import ContextItem
from anchor.models.scope import (
    DEFAULT_VAULT,
    ROOT_NAMESPACE,
    RetrievalScope,
    normalize_namespace,
)
from anchor.storage._where import matches_where

logger = logging.getLogger(__name__)


class InMemoryContextStore:
    """Dict-backed context store. Implements ContextStore protocol.

    Bound to one vault at construction (front #3): items are stamped
    with the store's vault on write, and only that vault is visible —
    the mount is the isolation boundary, not the caller's queries.
    """

    __slots__ = ("_items", "_lock", "_vault")

    def __init__(self, *, vault: str = DEFAULT_VAULT) -> None:
        self._items: dict[str, ContextItem] = {}
        self._lock = threading.Lock()
        self._vault = vault

    @property
    def vault(self) -> str:
        return self._vault

    def add(self, item: ContextItem) -> None:
        if item.vault != self._vault:
            item = item.model_copy(update={"vault": self._vault})
        with self._lock:
            self._items[item.id] = item

    def get(self, item_id: str) -> ContextItem | None:
        with self._lock:
            item = self._items.get(item_id)
        if item is None or item.vault != self._vault:
            return None
        return item

    def get_all(self) -> list[ContextItem]:
        with self._lock:
            return list(self._items.values())

    def delete(self, item_id: str) -> bool:
        with self._lock:
            return self._items.pop(item_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(items={len(self._items)})"


class InMemoryVectorStore:
    """Brute-force cosine similarity vector store.

    For development/testing only. Production use should provide
    FAISS, Chroma, Qdrant, etc. via the VectorStore protocol.
    """

    __slots__ = (
        "_embeddings",
        "_large_store_warned",
        "_lock",
        "_metadata",
        "_namespaces",
        "_vault",
    )

    _LARGE_STORE_THRESHOLD: int = 5000

    def __init__(self, *, vault: str = DEFAULT_VAULT) -> None:
        self._embeddings: dict[str, list[float]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._namespaces: dict[str, str] = {}
        self._large_store_warned: bool = False
        self._lock = threading.Lock()
        self._vault = vault

    @property
    def vault(self) -> str:
        return self._vault

    def add_embedding(
        self,
        item_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        *,
        namespace: str = ROOT_NAMESPACE,
    ) -> None:
        with self._lock:
            self._embeddings[item_id] = embedding
            self._namespaces[item_id] = normalize_namespace(namespace)
            if metadata:
                self._metadata[item_id] = metadata

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[tuple[str, float]]:
        with self._lock:
            if not self._embeddings:
                return []
            n = len(self._embeddings)
            if n > self._LARGE_STORE_THRESHOLD and not self._large_store_warned:
                logger.warning(
                    "InMemoryVectorStore has %d embeddings. Consider using a dedicated "
                    "vector database (FAISS, Chroma) for better performance.",
                    n,
                )
                self._large_store_warned = True
            results: list[tuple[str, float]] = []
            for item_id, emb in self._embeddings.items():
                if scope is not None and not scope.matches(
                    self._namespaces.get(item_id, ROOT_NAMESPACE)
                ):
                    continue
                if where is not None and not matches_where(
                    self._metadata.get(item_id, {}), where
                ):
                    continue
                score = cosine_similarity(query_embedding, emb)
                results.append((item_id, score))
            return heapq.nlargest(top_k, results, key=lambda x: x[1])

    def delete(self, item_id: str) -> bool:
        with self._lock:
            removed = self._embeddings.pop(item_id, None) is not None
            self._metadata.pop(item_id, None)
            self._namespaces.pop(item_id, None)
            return removed

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors without numpy.

        Delegates to :func:`anchor._math.cosine_similarity`.
        Kept for backwards compatibility with code that calls this static method.
        """
        return cosine_similarity(a, b)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(embeddings={len(self._embeddings)})"


class InMemoryDocumentStore:
    """Dict-backed document store. Implements DocumentStore protocol."""

    __slots__ = ("_documents", "_lock", "_metadata")

    def __init__(self) -> None:
        self._documents: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            self._documents[doc_id] = content
            if metadata:
                self._metadata[doc_id] = metadata

    def get_document(self, doc_id: str) -> str | None:
        with self._lock:
            return self._documents.get(doc_id)

    def list_documents(self) -> list[str]:
        with self._lock:
            return list(self._documents.keys())

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            removed = self._documents.pop(doc_id, None) is not None
            self._metadata.pop(doc_id, None)
            return removed

    def __repr__(self) -> str:
        return f"{type(self).__name__}(documents={len(self._documents)})"
