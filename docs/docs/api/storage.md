# Storage API Reference

Complete API reference for all storage protocols and implementations in
anchor.

For a usage guide, see [Storage Guide](../guides/storage.md).

---

## Protocols

### VectorStore

Protocol for vector similarity search backends (FAISS, Chroma, Qdrant, etc.).

```python
class VectorStore(Protocol):
    def add_embedding(
        self, item_id: str, embedding: list[float], metadata: dict[str, Any] | None = None
    ) -> None: ...

    def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]: ...

    def delete(self, item_id: str) -> bool: ...
```

| Method | Description |
|---|---|
| `add_embedding(item_id, embedding, metadata, *, namespace="/")` | Store an embedding vector under a namespace. Overwrites if `item_id` exists in this vault. |
| `search(query_embedding, top_k, where=None, *, scope=None)` | Return `(item_id, score)` tuples sorted by descending similarity; `where` (operators `$eq $ne $in $nin $gt $gte $lt $lte`) and `scope` (`RetrievalScope`) are pre-filters. |
| `delete(item_id)` | Remove an embedding. Returns `True` if found. |

---

### ContextStore

Protocol for storing and retrieving `ContextItem` objects.

```python
class ContextStore(Protocol):
    def add(self, item: ContextItem) -> None: ...
    def get(self, item_id: str) -> ContextItem | None: ...
    def get_all(self) -> list[ContextItem]: ...
    def delete(self, item_id: str) -> bool: ...
    def clear(self) -> None: ...
```

| Method | Description |
|---|---|
| `add(item)` | Persist a context item. Overwrites if same `id` exists. |
| `get(item_id)` | Retrieve by ID. Returns `None` if not found. |
| `get_all()` | Return all items. Order is implementation-defined. |
| `delete(item_id)` | Remove by ID. Returns `True` if found. |
| `clear()` | Remove all items. |

---

### DocumentStore

Protocol for raw document text storage (pre-chunking).

```python
class DocumentStore(Protocol):
    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None: ...
    def get_document(self, doc_id: str) -> str | None: ...
    def list_documents(self) -> list[str]: ...
    def delete_document(self, doc_id: str) -> bool: ...
```

| Method | Description |
|---|---|
| `add_document(doc_id, content, metadata)` | Store a document. |
| `get_document(doc_id)` | Retrieve text by ID. Returns `None` if not found. |
| `list_documents()` | Return all document IDs. |
| `delete_document(doc_id)` | Remove by ID. Returns `True` if found. |

---

### MemoryEntryStore

Protocol for persistent `MemoryEntry` storage.

```python
class MemoryEntryStore(Protocol):
    def add(self, entry: MemoryEntry) -> None: ...
    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]: ...
    def list_all(self) -> list[MemoryEntry]: ...
    def delete(self, entry_id: str) -> bool: ...
    def clear(self) -> None: ...
```

| Method | Description |
|---|---|
| `add(entry)` | Persist a memory entry. Overwrites if same `id` exists. |
| `search(query, top_k)` | Search by query string. Matching semantics are implementation-defined. |
| `list_all()` | Return all non-expired entries. |
| `delete(entry_id)` | Remove by ID. Returns `True` if found. |
| `clear()` | Remove all entries. |

---

### GarbageCollectableStore

Extended protocol for stores that support garbage collection of expired
entries.

```python
class GarbageCollectableStore(Protocol):
    def list_all_unfiltered(self) -> list[MemoryEntry]: ...
    def delete(self, entry_id: str) -> bool: ...
```

| Method | Description |
|---|---|
| `list_all_unfiltered()` | Return all entries including expired ones. |
| `delete(entry_id)` | Remove by ID. Returns `True` if found. |

---

## In-Memory Implementations

### InMemoryVectorStore

Brute-force cosine similarity vector store for development and testing.

```python
class InMemoryVectorStore:
    def __init__(self) -> None: ...
```

Implements `VectorStore`. Thread-safe via `threading.Lock`.

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `add_embedding` | `(item_id: str, embedding: list[float], metadata: dict[str, Any] \| None = None, *, namespace: str = "/") -> None` | Store an embedding vector with optional metadata, under a namespace. |
| `search` | `(query_embedding: list[float], top_k: int = 10, where: dict \| None = None, *, scope: RetrievalScope \| None = None) -> list[tuple[str, float]]` | Brute-force cosine similarity search over the rows the `where`/`scope` pre-filters keep. Logs a warning above 5,000 embeddings. |
| `delete` | `(item_id: str) -> bool` | Remove an embedding. Returns `True` if found. |

!!! warning
    Linear scan complexity O(N) per query. Use a dedicated vector database
    for production workloads.

---

### InMemoryContextStore

Dict-backed context store. Thread-safe.

```python
class InMemoryContextStore:
    def __init__(self) -> None: ...
```

Implements `ContextStore`.

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `add` | `(item: ContextItem) -> None` | Store a context item. |
| `get` | `(item_id: str) -> ContextItem \| None` | Retrieve by ID. |
| `get_all` | `() -> list[ContextItem]` | Return all items. |
| `delete` | `(item_id: str) -> bool` | Remove by ID. |
| `clear` | `() -> None` | Remove all items. |

---

### InMemoryDocumentStore

Dict-backed document store. Thread-safe.

```python
class InMemoryDocumentStore:
    def __init__(self) -> None: ...
```

Implements `DocumentStore`.

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `add_document` | `(doc_id: str, content: str, metadata: dict[str, Any] \| None = None) -> None` | Store a document. |
| `get_document` | `(doc_id: str) -> str \| None` | Retrieve by ID. |
| `list_documents` | `() -> list[str]` | Return all document IDs. |
| `delete_document` | `(doc_id: str) -> bool` | Remove by ID. |

---

### InMemoryEntryStore

In-memory `MemoryEntry` store with filtering and search.

```python
class InMemoryEntryStore:
    def __init__(self) -> None: ...
```

Implements `MemoryEntryStore` and `GarbageCollectableStore`. Thread-safe.

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `add` | `(entry: MemoryEntry) -> None` | Store a memory entry. |
| `delete` | `(entry_id: str) -> bool` | Remove by ID. |
| `clear` | `() -> None` | Remove all entries. |
| `search` | `(query: str, top_k: int = 5) -> list[MemoryEntry]` | Substring search, excluding expired entries. |
| `list_all` | `() -> list[MemoryEntry]` | Return all non-expired entries. |
| `list_all_unfiltered` | `() -> list[MemoryEntry]` | Return all entries including expired. |
| `get` | `(entry_id: str) -> MemoryEntry \| None` | Retrieve a single entry by ID. |
| `search_filtered` | `(query, top_k, *, user_id, session_id, memory_type, tags, created_after, created_before) -> list[MemoryEntry]` | Filtered search with multiple criteria. |
| `delete_by_user` | `(user_id: str) -> int` | Delete all entries for a user. Returns count deleted. |

---

## Persistent Implementations

### JsonFileMemoryStore

JSON-file-backed persistent `MemoryEntry` store. Uses atomic writes (temp
file + rename) to prevent corruption.

```python
class JsonFileMemoryStore:
    def __init__(
        self,
        file_path: str | Path,
        *,
        auto_save: bool = True,
    ) -> None: ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file_path` | `str \| Path` | -- | Path to the JSON file. Created on first save. Loaded on init if exists. |
| `auto_save` | `bool` | `True` | Persist after every mutation (`add`, `delete`, `clear`). |

Implements `MemoryEntryStore` and `GarbageCollectableStore`. Thread-safe.

**Methods:**

| Method | Signature | Description |
|---|---|---|
| `add` | `(entry: MemoryEntry) -> None` | Store and optionally persist to disk. |
| `delete` | `(entry_id: str) -> bool` | Remove by ID and optionally persist. |
| `clear` | `() -> None` | Remove all entries and persist. |
| `save` | `() -> None` | Explicitly flush to disk (atomic write). |
| `load` | `() -> None` | Reload entries from the JSON file. |
| `search` | `(query: str, top_k: int = 5) -> list[MemoryEntry]` | Substring search, excluding expired. |
| `list_all` | `() -> list[MemoryEntry]` | Return non-expired entries. |
| `list_all_unfiltered` | `() -> list[MemoryEntry]` | Return all entries including expired. |
| `get` | `(entry_id: str) -> MemoryEntry \| None` | Retrieve a single entry. |
| `search_filtered` | `(query, top_k, *, user_id, session_id, memory_type, tags, created_after, created_before) -> list[MemoryEntry]` | Filtered search. |
| `delete_by_user` | `(user_id: str) -> int` | Delete all entries for a user. |
| `export_user_entries` | `(user_id: str) -> list[MemoryEntry]` | Export all entries for a user (GDPR data portability). |

!!! note
    Not suitable for concurrent multi-process access. For multi-worker
    deployments, implement `MemoryEntryStore` with a database backend.
