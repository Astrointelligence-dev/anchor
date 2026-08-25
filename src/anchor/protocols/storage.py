"""Storage protocol definitions.

All storage backends implement these protocols using structural subtyping (PEP 544).
Users can provide any object that matches the interface -- no inheritance required.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from anchor.models.context import ContextItem
from anchor.models.memory import MemoryEntry


@runtime_checkable
class ContextStore(Protocol):
    """Protocol for storing and retrieving context items."""

    def add(self, item: ContextItem) -> None:
        """Persist a single context item to the store.

        Parameters:
            item: The context item to store. If an item with the same
                ``id`` already exists, the implementation should overwrite it.

        Side Effects:
            The item is durably persisted (or held in memory, depending on
            the backend) and will be visible to subsequent ``get`` /
            ``get_all`` calls.
        """
        ...

    def get(self, item_id: str) -> ContextItem | None:
        """Retrieve a single context item by its unique identifier.

        Parameters:
            item_id: The unique string identifier of the item.

        Returns:
            The matching ``ContextItem``, or ``None`` if no item with the
            given id exists.
        """
        ...

    def get_all(self) -> list[ContextItem]:
        """Return every context item currently held in the store.

        Returns:
            A list of all stored ``ContextItem`` objects.  The order is
            implementation-defined.  Returns an empty list when the store
            is empty.
        """
        ...

    def delete(self, item_id: str) -> bool:
        """Remove a context item by id.

        Parameters:
            item_id: The unique identifier of the item to delete.

        Returns:
            ``True`` if an item was found and deleted, ``False`` if no
            item with the given id existed.

        Side Effects:
            The item is permanently removed from the store.
        """
        ...

    def clear(self) -> None:
        """Remove all items from the store.

        Side Effects:
            The store is left empty.  This operation is irreversible.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector similarity search backends.

    Implementations might wrap FAISS, Chroma, Qdrant, Pinecone, etc.
    """

    def add_embedding(
        self, item_id: str, embedding: list[float], metadata: dict[str, Any] | None = None
    ) -> None:
        """Store an embedding vector associated with an item id.

        Parameters:
            item_id: A unique identifier linking the embedding back to its
                source context item or document.
            embedding: The dense vector representation of the item.
            metadata: Optional key-value metadata attached to the vector
                entry (e.g., source filename, chunk index).

        Side Effects:
            The embedding is persisted in the vector index.  If an entry
            for ``item_id`` already exists, the implementation should
            overwrite it.
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Find the most similar embeddings to a query vector.

        Parameters:
            query_embedding: The dense vector to compare against stored
                embeddings.
            top_k: Maximum number of results to return.
            where: Optional metadata filter — only entries whose metadata
                contains every ``key: value`` pair are considered
                (pre-filtering, applied before top_k).

        Returns:
            A list of ``(item_id, score)`` tuples ordered by descending
            similarity.  The score semantics (cosine, dot-product, etc.)
            are implementation-defined.
        """
        ...

    def delete(self, item_id: str) -> bool:
        """Remove the embedding associated with an item id.

        Parameters:
            item_id: The identifier of the embedding to remove.

        Returns:
            ``True`` if the embedding was found and removed, ``False``
            otherwise.

        Side Effects:
            The embedding is permanently removed from the vector index.
        """
        ...


@runtime_checkable
class DocumentStore(Protocol):
    """Protocol for document storage (raw text before chunking/indexing)."""

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Store a raw document (pre-chunking / pre-indexing).

        Parameters:
            doc_id: A unique identifier for the document.
            content: The full text content of the document.
            metadata: Optional key-value metadata (e.g., source URL,
                author, ingestion timestamp).

        Side Effects:
            The document is persisted and becomes visible to subsequent
            ``get_document`` and ``list_documents`` calls.
        """
        ...

    def get_document(self, doc_id: str) -> str | None:
        """Retrieve a document's text content by its unique identifier.

        Parameters:
            doc_id: The identifier assigned when the document was added.

        Returns:
            The full text content of the document, or ``None`` if no
            document with the given id exists.
        """
        ...

    def list_documents(self) -> list[str]:
        """Return the ids of all stored documents.

        Returns:
            A list of document id strings.  The order is
            implementation-defined.  Returns an empty list when the store
            contains no documents.
        """
        ...

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document by its id.

        Parameters:
            doc_id: The identifier of the document to delete.

        Returns:
            ``True`` if a document was found and deleted, ``False`` if no
            document with the given id existed.

        Side Effects:
            The document is permanently removed from the store.
        """
        ...


@runtime_checkable
class MemoryEntryStore(Protocol):
    """Protocol for persistent memory entry storage.

    Implementations hold ``MemoryEntry`` objects and provide basic
    CRUD + search operations.  The store is used by ``MemoryManager``,
    ``ScoredMemoryRetriever``, and the eviction-promotion pipeline steps.
    """

    def add(self, entry: MemoryEntry) -> None:
        """Persist a single memory entry to the store.

        Parameters:
            entry: The memory entry to store.  If an entry with the same
                ``id`` already exists, the implementation should overwrite it.

        Side Effects:
            The entry is durably persisted (or held in memory, depending on
            the backend) and will be visible to subsequent ``search`` /
            ``list_all`` calls.
        """
        ...

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search for memory entries relevant to a query string.

        Parameters:
            query: Free-text search query.  The matching semantics
                (keyword, embedding, hybrid) are implementation-defined.
            top_k: Maximum number of results to return.

        Returns:
            A list of up to ``top_k`` matching ``MemoryEntry`` objects,
            ordered by relevance (most relevant first).  Returns an empty
            list when no entries match.
        """
        ...

    def list_all(self) -> list[MemoryEntry]:
        """Return every non-expired memory entry in the store.

        Returns:
            A list of all stored ``MemoryEntry`` objects.  The order is
            implementation-defined.  Returns an empty list when the store
            is empty.
        """
        ...

    def delete(self, entry_id: str) -> bool:
        """Remove a memory entry by its unique identifier.

        Parameters:
            entry_id: The ``id`` of the entry to delete.

        Returns:
            ``True`` if an entry was found and deleted, ``False`` if no
            entry with the given id existed.

        Side Effects:
            The entry is permanently removed from the store.
        """
        ...

    def clear(self) -> None:
        """Remove all entries from the store.

        Side Effects:
            The store is left empty.  This operation is irreversible.
        """
        ...


@runtime_checkable
class GarbageCollectableStore(Protocol):
    """A memory entry store that supports garbage collection.

    Extends the base ``MemoryEntryStore`` contract with
    ``list_all_unfiltered`` so that expired entries can be discovered
    and deleted by the ``MemoryGarbageCollector``.
    """

    def list_all_unfiltered(self) -> list[MemoryEntry]:
        """Return *all* entries including expired ones.

        Unlike ``MemoryEntryStore.list_all`` which may filter out expired
        entries, this method returns every entry so that the garbage
        collector can identify and prune them.

        Returns:
            A list of all ``MemoryEntry`` objects in the store,
            regardless of expiration status.
        """
        ...

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by id.

        Parameters:
            entry_id: The ``id`` of the entry to delete.

        Returns:
            ``True`` if found and deleted, ``False`` otherwise.
        """
        ...


# ---------------------------------------------------------------------------
# Async protocol variants
# ---------------------------------------------------------------------------


@runtime_checkable
class AsyncContextStore(Protocol):
    """Async variant of :class:`ContextStore`.

    All methods mirror their sync counterparts.  See :class:`ContextStore`
    for full parameter documentation.
    """

    async def add(self, item: ContextItem) -> None: ...
    async def get(self, item_id: str) -> ContextItem | None: ...
    async def get_all(self) -> list[ContextItem]: ...
    async def delete(self, item_id: str) -> bool: ...
    async def clear(self) -> None: ...


@runtime_checkable
class AsyncVectorStore(Protocol):
    """Async variant of :class:`VectorStore`.

    All methods mirror their sync counterparts.  See :class:`VectorStore`
    for full parameter documentation.
    """

    async def add_embedding(
        self, item_id: str, embedding: list[float], metadata: dict[str, Any] | None = None
    ) -> None: ...
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]: ...
    async def delete(self, item_id: str) -> bool: ...


@runtime_checkable
class AsyncDocumentStore(Protocol):
    """Async variant of :class:`DocumentStore`.

    All methods mirror their sync counterparts.  See :class:`DocumentStore`
    for full parameter documentation.
    """

    async def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None: ...
    async def get_document(self, doc_id: str) -> str | None: ...
    async def list_documents(self) -> list[str]: ...
    async def delete_document(self, doc_id: str) -> bool: ...


@runtime_checkable
class AsyncMemoryEntryStore(Protocol):
    """Async variant of :class:`MemoryEntryStore`.

    All methods mirror their sync counterparts.  See :class:`MemoryEntryStore`
    for full parameter documentation.
    """

    async def add(self, entry: MemoryEntry) -> None: ...
    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]: ...
    async def list_all(self) -> list[MemoryEntry]: ...
    async def delete(self, entry_id: str) -> bool: ...
    async def clear(self) -> None: ...


@runtime_checkable
class AsyncGarbageCollectableStore(Protocol):
    """Async variant of :class:`GarbageCollectableStore`.

    All methods mirror their sync counterparts.  See
    :class:`GarbageCollectableStore` for full parameter documentation.
    """

    async def list_all_unfiltered(self) -> list[MemoryEntry]: ...
    async def delete(self, entry_id: str) -> bool: ...
