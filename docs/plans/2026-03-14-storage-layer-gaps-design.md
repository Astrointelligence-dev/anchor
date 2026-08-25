# Spec 1: Storage Layer Gaps — GraphStore, ConversationStore, CacheBackends

**Date:** 2026-03-14
**Status:** Draft
**Scope:** Three new storage protocols + InMemory/SQLite/PostgreSQL backends + Redis/SQLite cache backends
**Companion Spec:** Spec 2 (Agent Checkpointing + Unified StorageManager) — depends on this spec

## Problem

Anchor has 5 core storage protocols (ContextStore, VectorStore, DocumentStore, MemoryEntryStore, GarbageCollectableStore) with implementations across InMemory, SQLite, PostgreSQL, and Redis. However, three major components lack persistent storage:

1. **SimpleGraphMemory** — in-memory only. No GraphStore protocol. Entity-relationship data is lost on restart.
2. **ConversationMemory** (SlidingWindow, SummaryBuffer, ProgressiveSummarization) — in-memory only. Conversation turns and summary tiers are lost on restart.
3. **CacheBackend** — only InMemoryCacheBackend exists. No Redis or SQLite cache despite both connection managers already existing in the project.

The 2026 state of the art (Mem0, LangGraph, Letta, MemOS) treats persistent memory as table stakes. Agents need to survive restarts, share state across sessions, and maintain long-term knowledge graphs.

## Design Decisions

- **Approach A (Mirror Existing Patterns)** — each new protocol follows the same structure as existing Anchor storage: PEP 544 runtime-checkable protocol, InMemory default, SQLite + PostgreSQL persistent backends, sync + async variants.
- **GraphStore backends:** SQLite + PostgreSQL only (matches existing convention). Protocol designed so Neo4j/KuzuDB wrappers satisfy it trivially.
- **ConversationStore:** Persists turns (append-only) + summary tiers. Key facts route to existing MemoryEntryStore (no duplication).
- **CacheBackend protocol:** Unchanged. New Redis + SQLite backends implement existing 4-method interface.
- **Serialization:** JSON throughout (matches LangGraph's JsonPlusSerializer pattern, safe and portable).

## 1. GraphStore Protocol

### Protocol Definition

```python
@runtime_checkable
class GraphStore(Protocol):
    """Protocol for persistent graph storage (entities + relationships).

    Implementations might wrap SQLite adjacency tables, PostgreSQL with
    recursive CTEs, Neo4j, KuzuDB, or any graph-capable backend.
    """

    def add_node(
        self, node_id: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add or update a node. Merges metadata if node exists."""
        ...

    def add_edge(
        self, source: str, relation: str, target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a directed edge. Auto-creates source/target nodes if missing."""
        ...

    def get_neighbors(
        self, node_id: str, max_depth: int = 1,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]:
        """BFS traversal returning neighbor node IDs.

        Traverses edges in both directions up to max_depth hops.
        Optional relation_filter limits traversal to specific edge types.
        The starting node is NOT included in results.
        """
        ...

    def get_edges(self, node_id: str) -> list[tuple[str, str, str]]:
        """Return all edges involving a node as (source, relation, target)."""
        ...

    def get_node_metadata(self, node_id: str) -> dict[str, Any] | None:
        """Return metadata for a node, or None if not found.

        Unlike SimpleGraphMemory.get_entity_metadata (which raises KeyError),
        the protocol returns None for missing nodes. The backwards-compatible
        alias on InMemoryGraphStore preserves the KeyError behavior.
        """
        ...

    def link_memory(self, node_id: str, memory_id: str) -> None:
        """Associate a MemoryEntry ID with a graph node.

        Raises KeyError if the node does not exist. Unlike add_edge
        (which auto-creates nodes), link_memory requires explicit node
        creation first — this prevents orphaned memory links to
        non-existent entities.
        """
        ...

    def get_memory_ids(
        self, node_id: str, max_depth: int = 1
    ) -> list[str]:
        """Get memory IDs for a node and its neighborhood."""
        ...

    def remove_node(self, node_id: str) -> None:
        """Remove a node, its edges, and its memory linkages."""
        ...

    def remove_edge(self, source: str, relation: str, target: str) -> bool:
        """Remove a specific directed edge. Returns True if found."""
        ...

    def list_nodes(self) -> list[str]:
        """Return all node IDs."""
        ...

    def list_edges(self) -> list[tuple[str, str, str]]:
        """Return all edges as (source, relation, target) tuples."""
        ...

    def clear(self) -> None:
        """Remove all nodes, edges, and memory linkages."""
        ...
```

Async variant `AsyncGraphStore` mirrors all methods with `async def`.

### Why These Methods

- **`edge metadata`** — production graph memory (Neo4j constitutional graph pattern) stores weight, confidence, timestamps on edges. Without this, agents cannot reason about edge quality.
- **`relation_filter` on `get_neighbors`** — agents filter by relationship type during multi-hop traversal (e.g., "only follow 'causes' edges"). This is the dominant pattern in Neo4j agent-memory and KuzuDB production deployments.
- **`remove_edge`** — targeted edge removal without destroying nodes. Graph memory needs surgical updates as relationships change.
- **`link_memory` / `get_memory_ids`** — preserves Anchor's existing entity-to-memory linking pattern from SimpleGraphMemory. Enables the fan-out retrieval pattern (query graph, find entities, fetch linked memories).

### InMemory Implementation

`SimpleGraphMemory` becomes the InMemory `GraphStore` implementation. Changes:
- Add `metadata` parameter to `add_edge` (store as dict alongside edge tuples)
- Add `relation_filter` parameter to `get_related_entities` (filter during BFS)
- Add `remove_edge` method
- Rename `add_entity` to `add_node`, `get_related_entities` to `get_neighbors` (protocol alignment)
- Keep backwards-compatible aliases for the old method names

### SQLite Backend

**Schema:**

```sql
CREATE TABLE graph_nodes (
    node_id TEXT PRIMARY KEY,
    metadata_json TEXT DEFAULT '{}'
);

CREATE TABLE graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    relation TEXT NOT NULL,
    target TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    UNIQUE(source, relation, target)
);

CREATE TABLE graph_memory_links (
    node_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    PRIMARY KEY (node_id, memory_id)
);

CREATE INDEX idx_edges_source ON graph_edges(source);
CREATE INDEX idx_edges_target ON graph_edges(target);
CREATE INDEX idx_edges_relation ON graph_edges(relation);
CREATE INDEX idx_memory_links_node ON graph_memory_links(node_id);
```

**Traversal:** `get_neighbors` with `max_depth > 1` uses iterative Python-side BFS loading neighbors level by level. SQLite lacks `WITH RECURSIVE` performance for filtered multi-hop, so iterative is more predictable.

**File:** `src/anchor/storage/sqlite/_graph_store.py` — classes `SqliteGraphStore`, `AsyncSqliteGraphStore`.

### PostgreSQL Backend

**Schema:** Same structure as SQLite but with JSONB for metadata.

**Traversal:** `get_neighbors` uses `WITH RECURSIVE` CTE for efficient multi-hop:

```sql
WITH RECURSIVE reachable(node_id, depth, path) AS (
    -- Base case: direct neighbors
    SELECT target, 1, ARRAY[$1, target]
    FROM graph_edges WHERE source = $1
    UNION
    SELECT source, 1, ARRAY[$1, source]
    FROM graph_edges WHERE target = $1
    UNION ALL
    -- Recursive step: expand with cycle prevention via path array
    SELECT CASE WHEN e.source = r.node_id THEN e.target ELSE e.source END,
           r.depth + 1,
           r.path || CASE WHEN e.source = r.node_id THEN e.target ELSE e.source END
    FROM reachable r
    JOIN graph_edges e ON (e.source = r.node_id OR e.target = r.node_id)
    WHERE r.depth < $2
      AND NOT (CASE WHEN e.source = r.node_id THEN e.target ELSE e.source END) = ANY(r.path)
      -- Optional: AND e.relation = ANY($3)
)
SELECT DISTINCT node_id FROM reachable WHERE node_id != $1;
```

**Cycle prevention:** The `path` array tracks visited nodes per traversal branch. The `NOT ... = ANY(r.path)` guard prevents revisiting nodes within a single traversal path, avoiding infinite recursion in cyclic graphs. The depth bound provides an additional safety limit.

**File:** `src/anchor/storage/postgres/_graph_store.py` — class `PostgresGraphStore` (async only, matches existing Postgres pattern).

### SimpleGraphMemory Refactoring

`SimpleGraphMemory` is refactored to accept an optional `GraphStore` backend:

```python
class SimpleGraphMemory:
    def __init__(self, store: GraphStore | None = None) -> None:
        # If no store provided, uses InMemoryGraphStore (self-contained, current behavior)
        # If store provided, delegates all operations to it
```

This preserves full backwards compatibility — existing code that does `SimpleGraphMemory()` works identically. But users can now do `SimpleGraphMemory(store=SqliteGraphStore(conn))` for persistence.

## 2. ConversationStore Protocol

### Protocol Definition

```python
@runtime_checkable
class ConversationStore(Protocol):
    """Protocol for persistent conversation history storage.

    Stores conversation turns (append-only) and summary tiers per session.
    Key facts are stored separately via MemoryEntryStore.
    """

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Append a single turn to a session's history.

        Append-only semantics (write-ahead log pattern). Turns are
        ordered by insertion order within a session.
        """
        ...

    def load_turns(
        self, session_id: str, limit: int | None = None
    ) -> list[ConversationTurn]:
        """Load turns for a session, most recent last.

        Parameters:
            session_id: The session to load.
            limit: If set, return only the most recent N turns.
                   None means load all turns.
        """
        ...

    def save_summary_tiers(
        self, session_id: str, tiers: dict[int, SummaryTier | None]
    ) -> None:
        """Persist summary tiers for a session.

        Full-replace semantics: overwrites any existing tiers for
        the session. Called on compaction events, not every turn.
        """
        ...

    def load_summary_tiers(
        self, session_id: str
    ) -> dict[int, SummaryTier | None]:
        """Load summary tiers for a session.

        Returns a dict mapping tier level (1, 2, 3) to SummaryTier
        or None if that tier has no content.
        """
        ...

    def truncate_turns(self, session_id: str, keep_last: int) -> None:
        """Remove all but the most recent N turns for a session.

        Called when SlidingWindowMemory evicts turns, to keep
        persistent storage bounded.
        """
        ...

    def delete_session(self, session_id: str) -> bool:
        """Delete all data (turns + tiers) for a session."""
        ...

    def list_sessions(self) -> list[str]:
        """Return all session IDs with stored data."""
        ...

    def clear(self) -> None:
        """Remove all sessions and their data."""
        ...
```

Async variant `AsyncConversationStore` mirrors all methods with `async def`.

### Why These Methods

- **`append_turn` (not `save_turns`)** — the 2026 pattern (LangGraph checkpointers, Redis RedisSaver, DynamoDB) is append-only. Cheaper I/O, no data loss on crash, matches write-ahead log semantics.
- **`limit` on `load_turns`** — load only the most recent N turns for the sliding window. Production agents (AWS Session Management, Databricks stateful agents) do not load full history when they only need the last 20 messages.
- **`truncate_turns`** — syncs with eviction. Keeps persistent storage bounded. Without this, append-only storage grows unbounded.
- **Separate `save/load_summary_tiers`** — tiers change infrequently (only on compaction events). Separating from turns avoids rewriting tiers on every message.

### InMemory Implementation

**File:** `src/anchor/storage/memory_store.py` — class `InMemoryConversationStore`.

```python
class InMemoryConversationStore:
    """Dict-backed conversation store. Implements ConversationStore protocol."""

    def __init__(self) -> None:
        self._turns: dict[str, list[ConversationTurn]] = {}
        self._tiers: dict[str, dict[int, SummaryTier | None]] = {}
        self._lock = threading.Lock()
```

### SQLite Backend

**Schema:**

```sql
CREATE TABLE conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE summary_tiers (
    session_id TEXT NOT NULL,
    tier_level INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    source_turn_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, tier_level)
);

CREATE INDEX idx_turns_session ON conversation_turns(session_id, turn_index);
```

**Turn indexing:** `turn_index` is auto-assigned by the store on `append_turn`, using `COALESCE((SELECT MAX(turn_index) FROM conversation_turns WHERE session_id = ?), -1) + 1`. This is a store-managed sequence, not a field on `ConversationTurn`.

**Truncation:** `DELETE FROM conversation_turns WHERE session_id = ? AND turn_index < (SELECT MAX(turn_index) - ? FROM conversation_turns WHERE session_id = ?)`.

**File:** `src/anchor/storage/sqlite/_conversation_store.py` — classes `SqliteConversationStore`, `AsyncSqliteConversationStore`.

### PostgreSQL Backend

Same schema with TIMESTAMPTZ, JSONB. `truncate_turns` uses the same subquery pattern.

**File:** `src/anchor/storage/postgres/_conversation_store.py` — class `PostgresConversationStore` (async only).

### MemoryManager Integration

`MemoryManager` gains:

```python
class MemoryManager:
    def __init__(
        self,
        ...,
        conversation_store: ConversationStore | None = None,
        session_id: str | None = None,
        auto_persist: bool = False,
    ) -> None:
```

- **`conversation_store`** — optional persistent backend.
- **`session_id`** — required when `conversation_store` is set. `__init__` raises `ValueError` if `conversation_store` is provided without `session_id`. Scopes all persistence to this session.
- **`auto_persist`** — when True, each `add_*_message()` call appends to the store. Summary tiers persist on compaction events via a callback.
- **`save()` / `load()`** — explicit save/load for manual persistence mode.
- **`load()` restores full state** — loads turns into the SlidingWindowMemory, loads tiers into ProgressiveSummarizationMemory (if applicable).

### ProgressiveSummarizationMemory Integration

When `auto_persist=True` and a `ConversationStore` is configured, `ProgressiveSummarizationMemory` calls `save_summary_tiers()` after each compaction event (in the `_handle_eviction` / `_handle_eviction_async` callbacks). Key facts extracted during compaction are routed to `MemoryEntryStore` (existing behavior, unchanged).

## 3. Cache Backends

### Protocol

The existing `CacheBackend` protocol is unchanged:

```python
@runtime_checkable
class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: float | None = None) -> None: ...
    def invalidate(self, key: str) -> None: ...
    def clear(self) -> None: ...
```

### Redis CacheBackend

**File:** `src/anchor/cache/redis_backend.py`

```python
class RedisCacheBackend:
    """Redis-backed cache. Implements CacheBackend protocol.

    Uses JSON serialization for portability and debuggability.
    Leverages native Redis SETEX for TTL management.
    """

    def __init__(
        self,
        connection_manager: RedisConnectionManager,
        default_ttl: float | None = 300.0,
        key_prefix: str = "cache:",
    ) -> None:
```

**Implementation details:**
- Reuses existing `RedisConnectionManager` from `src/anchor/storage/redis/_connection.py`
- JSON serialization via `json.dumps` / `json.loads` (matches LangGraph's JsonPlusSerializer approach, safe and portable)
- TTL via native Redis `SETEX` / `PSETEX`
- Key prefix scoping: full key = `{connection_manager.prefix}{key_prefix}{key}` (note: `RedisConnectionManager` exposes the property as `prefix`, not `key_prefix`)
- `clear()` uses `SCAN` + `DEL` with prefix matching (safe for shared Redis instances)

### SQLite CacheBackend

**File:** `src/anchor/cache/sqlite_backend.py`

```python
class SqliteCacheBackend:
    """SQLite-backed cache for local persistent caching.

    Survives process restarts. Lazy expiration on read
    (matches InMemoryCacheBackend pattern).
    """

    def __init__(
        self,
        connection_manager: SqliteConnectionManager,
        default_ttl: float | None = 300.0,
    ) -> None:
```

**Schema:**

```sql
CREATE TABLE IF NOT EXISTS cache_entries (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL
);

CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at);
```

**Implementation details:**
- Reuses existing `SqliteConnectionManager` from `src/anchor/storage/sqlite/_connection.py`
- JSON serialization via `json.dumps` / `json.loads`
- Lazy expiration on `get()`: checks `expires_at`, deletes if expired (same pattern as `InMemoryCacheBackend`)
- Periodic cleanup: `DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?` (can be called explicitly or on a schedule)
- No max_size enforcement (SQLite handles large tables well; users manage via TTL)

## 4. Serialization

All new backends use the existing serialization module (`src/anchor/storage/_serialization.py`) extended with:

- `conversation_turn_to_row(turn: ConversationTurn) -> dict` — serializes a turn for SQL storage
- `row_to_conversation_turn(row) -> ConversationTurn` — deserializes from SQL row
- `summary_tier_to_row(tier: SummaryTier) -> dict` — serializes a tier for SQL storage
- `row_to_summary_tier(row) -> SummaryTier` — deserializes from SQL row
- Graph nodes/edges use simple dict/tuple serialization (metadata as JSON)

## 5. Schema Management

New tables are added to the existing schema modules:

- **SQLite:** `src/anchor/storage/sqlite/_schema.py` — add `GRAPH_SCHEMA`, `CONVERSATION_SCHEMA`, `CACHE_SCHEMA` constants alongside existing `CONTEXT_SCHEMA`, `VECTOR_SCHEMA`, etc.
- **PostgreSQL:** `src/anchor/storage/postgres/_schema.py` — same pattern.

Schema creation happens in each store's `__init__` (or `initialize()` for async Postgres), matching the existing pattern.

## 6. Public API Exports

New classes exported from `src/anchor/__init__.py`:

```python
# Protocols
from anchor.protocols.storage import GraphStore, AsyncGraphStore
from anchor.protocols.storage import ConversationStore, AsyncConversationStore

# InMemory implementations
from anchor.storage.memory_store import InMemoryGraphStore, InMemoryConversationStore

# Cache backends
from anchor.cache.redis_backend import RedisCacheBackend
from anchor.cache.sqlite_backend import SqliteCacheBackend
```

SQLite/PostgreSQL backends exported from their respective `__init__.py` files (gated behind optional dependency imports, matching existing pattern).

## 7. Testing Strategy

Each new component gets:

1. **Protocol conformance tests** — verify all implementations satisfy the protocol via `isinstance()` checks
2. **Unit tests per backend** — CRUD operations, edge cases (duplicate nodes, missing sessions, expired cache entries)
3. **Integration tests** — GraphStore + MemoryEntryStore fan-out retrieval, ConversationStore + MemoryManager round-trip, CacheBackend in pipeline
4. **Backwards compatibility tests** — `SimpleGraphMemory()` (no store) behaves identically to current behavior

Test files follow existing convention:
- `tests/test_storage/test_graph_store.py`
- `tests/test_storage/test_conversation_store.py`
- `tests/test_cache/test_redis_backend.py`
- `tests/test_cache/test_sqlite_backend.py`
- `tests/test_integration/test_persistent_memory.py`

## 8. File Inventory

### New Files

| File | Contents |
|------|----------|
| `src/anchor/storage/memory_store.py` | Add `InMemoryGraphStore`, `InMemoryConversationStore` (extend existing file) |
| `src/anchor/storage/sqlite/_graph_store.py` | `SqliteGraphStore`, `AsyncSqliteGraphStore` |
| `src/anchor/storage/sqlite/_conversation_store.py` | `SqliteConversationStore`, `AsyncSqliteConversationStore` |
| `src/anchor/storage/postgres/_graph_store.py` | `PostgresGraphStore` |
| `src/anchor/storage/postgres/_conversation_store.py` | `PostgresConversationStore` |
| `src/anchor/cache/redis_backend.py` | `RedisCacheBackend` |
| `src/anchor/cache/sqlite_backend.py` | `SqliteCacheBackend` |

### Modified Files

| File | Changes |
|------|---------|
| `src/anchor/protocols/storage.py` | Add `GraphStore`, `AsyncGraphStore`, `ConversationStore`, `AsyncConversationStore` |
| `src/anchor/memory/graph_memory.py` | Refactor to accept optional `GraphStore`, add new methods |
| `src/anchor/memory/manager.py` | Add `conversation_store`, `session_id`, `auto_persist`, `save()`, `load()`; update `__slots__` tuple |
| `src/anchor/storage/_serialization.py` | Add turn/tier serialization helpers |
| `src/anchor/storage/sqlite/_schema.py` | Add graph + conversation + cache schemas |
| `src/anchor/storage/postgres/_schema.py` | Add graph + conversation schemas |
| `src/anchor/storage/sqlite/__init__.py` | Export new store classes |
| `src/anchor/storage/postgres/__init__.py` | Export new store classes |
| `src/anchor/cache/__init__.py` | Export new cache backends |
| `src/anchor/__init__.py` | Export new public API |

## 9. Non-Goals (Spec 2)

The following are explicitly out of scope for this spec and will be addressed in Spec 2:

- **Agent state checkpointing** — full agent state save/restore (LangGraph checkpointer equivalent)
- **Unified StorageManager** — coordination layer for atomic cross-store operations (fan-out queries, consistent writes)
- **Semantic caching** — vector-similarity-based cache lookup (requires embedding functions)
- **Neo4j / KuzuDB backends** — protocol supports them, but implementations are user-provided or future work
- **Redis VectorStore / DocumentStore** — not in current scope
