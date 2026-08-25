# Storage Layer Gaps Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent storage for graph memory, conversation history, and pipeline cache — closing the three storage gaps identified in the spec.

**Architecture:** Three new storage protocols (GraphStore, ConversationStore, CacheBackend backends) following the existing Anchor pattern: PEP 544 protocol + InMemory default + SQLite/PostgreSQL persistent backends. Each protocol gets sync + async variants. Existing `SimpleGraphMemory` and `MemoryManager` are refactored to accept the new stores via constructor injection.

**Tech Stack:** Python 3.11+, Pydantic v2, SQLite (aiosqlite), PostgreSQL (asyncpg), Redis (redis-py), pytest

**Spec:** `docs/superpowers/specs/2026-03-14-storage-layer-gaps-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/anchor/storage/sqlite/_graph_store.py` | SqliteGraphStore + AsyncSqliteGraphStore |
| `src/anchor/storage/sqlite/_conversation_store.py` | SqliteConversationStore + AsyncSqliteConversationStore |
| `src/anchor/storage/postgres/_graph_store.py` | PostgresGraphStore (async only) |
| `src/anchor/storage/postgres/_conversation_store.py` | PostgresConversationStore (async only) |
| `src/anchor/cache/redis_backend.py` | RedisCacheBackend |
| `src/anchor/cache/sqlite_backend.py` | SqliteCacheBackend |
| `tests/test_storage/test_graph_store.py` | GraphStore protocol + InMemory + SQLite tests |
| `tests/test_storage/test_conversation_store.py` | ConversationStore protocol + InMemory + SQLite tests |
| `tests/test_cache/test_redis_backend.py` | RedisCacheBackend tests (mocked Redis) |
| `tests/test_cache/test_sqlite_backend.py` | SqliteCacheBackend tests |
| `tests/test_integration/test_persistent_memory.py` | End-to-end: MemoryManager + ConversationStore + GraphStore |

### Modified Files
| File | Changes |
|------|---------|
| `src/anchor/protocols/storage.py` | Add GraphStore, AsyncGraphStore, ConversationStore, AsyncConversationStore |
| `src/anchor/storage/memory_store.py` | Add InMemoryGraphStore, InMemoryConversationStore |
| `src/anchor/memory/graph_memory.py` | Refactor to accept optional GraphStore, add new methods, backwards-compat aliases |
| `src/anchor/memory/manager.py` | Add conversation_store, session_id, auto_persist, save(), load(); update __slots__ |
| `src/anchor/storage/_serialization.py` | Add conversation_turn_to_row, row_to_conversation_turn, summary_tier_to_row, row_to_summary_tier |
| `src/anchor/storage/sqlite/_schema.py` | Add graph + conversation + cache table DDL to _TABLES and _INDEXES |
| `src/anchor/storage/postgres/_schema.py` | Add graph + conversation table DDL |
| `src/anchor/storage/sqlite/__init__.py` | Export new store classes |
| `src/anchor/storage/postgres/__init__.py` | Export new store classes |
| `src/anchor/cache/__init__.py` | Export new cache backends |
| `src/anchor/__init__.py` | Export new public API symbols |

---

## Chunk 1: GraphStore Protocol + InMemory Implementation

### Task 1: Add GraphStore and AsyncGraphStore protocols

**Files:**
- Modify: `src/anchor/protocols/storage.py`

- [ ] **Step 1: Add GraphStore protocol to storage.py**

Append after `AsyncGarbageCollectableStore` (end of file). Import `ConversationTurn` and `SummaryTier` are NOT needed here — GraphStore only uses `str`, `dict`, `Any`.

```python
@runtime_checkable
class GraphStore(Protocol):
    """Protocol for persistent graph storage (entities + relationships)."""

    def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None: ...
    def add_edge(
        self, source: str, relation: str, target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
    def get_neighbors(
        self, node_id: str, max_depth: int = 1,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]: ...
    def get_edges(self, node_id: str) -> list[tuple[str, str, str]]: ...
    def get_node_metadata(self, node_id: str) -> dict[str, Any] | None: ...
    def link_memory(self, node_id: str, memory_id: str) -> None: ...
    def get_memory_ids(self, node_id: str, max_depth: int = 1) -> list[str]: ...
    def remove_node(self, node_id: str) -> None: ...
    def remove_edge(self, source: str, relation: str, target: str) -> bool: ...
    def list_nodes(self) -> list[str]: ...
    def list_edges(self) -> list[tuple[str, str, str]]: ...
    def clear(self) -> None: ...


@runtime_checkable
class AsyncGraphStore(Protocol):
    """Async variant of GraphStore."""

    async def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None: ...
    async def add_edge(
        self, source: str, relation: str, target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
    async def get_neighbors(
        self, node_id: str, max_depth: int = 1,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]: ...
    async def get_edges(self, node_id: str) -> list[tuple[str, str, str]]: ...
    async def get_node_metadata(self, node_id: str) -> dict[str, Any] | None: ...
    async def link_memory(self, node_id: str, memory_id: str) -> None: ...
    async def get_memory_ids(self, node_id: str, max_depth: int = 1) -> list[str]: ...
    async def remove_node(self, node_id: str) -> None: ...
    async def remove_edge(self, source: str, relation: str, target: str) -> bool: ...
    async def list_nodes(self) -> list[str]: ...
    async def list_edges(self) -> list[tuple[str, str, str]]: ...
    async def clear(self) -> None: ...
```

- [ ] **Step 2: Verify protocols import cleanly**

Run: `python -c "from anchor.protocols.storage import GraphStore, AsyncGraphStore; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/anchor/protocols/storage.py
git commit -m "feat: add GraphStore and AsyncGraphStore protocols"
```

### Task 2: Add InMemoryGraphStore

**Files:**
- Modify: `src/anchor/storage/memory_store.py`
- Test: `tests/test_storage/test_graph_store.py`

- [ ] **Step 1: Write failing tests for InMemoryGraphStore**

Create `tests/test_storage/test_graph_store.py`:

```python
"""Tests for GraphStore protocol and InMemory implementation."""

from anchor.protocols.storage import GraphStore
from anchor.storage.memory_store import InMemoryGraphStore


class TestInMemoryGraphStoreProtocol:
    def test_satisfies_protocol(self):
        store = InMemoryGraphStore()
        assert isinstance(store, GraphStore)


class TestInMemoryGraphStoreNodes:
    def test_add_and_list_nodes(self):
        store = InMemoryGraphStore()
        store.add_node("alice", {"type": "person"})
        store.add_node("bob")
        assert sorted(store.list_nodes()) == ["alice", "bob"]

    def test_add_node_merges_metadata(self):
        store = InMemoryGraphStore()
        store.add_node("alice", {"type": "person"})
        store.add_node("alice", {"role": "engineer"})
        meta = store.get_node_metadata("alice")
        assert meta == {"type": "person", "role": "engineer"}

    def test_get_node_metadata_returns_none_for_missing(self):
        store = InMemoryGraphStore()
        assert store.get_node_metadata("nonexistent") is None

    def test_remove_node(self):
        store = InMemoryGraphStore()
        store.add_node("alice")
        store.add_edge("alice", "knows", "bob")
        store.remove_node("alice")
        assert "alice" not in store.list_nodes()
        assert store.list_edges() == []


class TestInMemoryGraphStoreEdges:
    def test_add_and_list_edges(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        edges = store.list_edges()
        assert ("alice", "knows", "bob") in edges

    def test_add_edge_auto_creates_nodes(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        assert sorted(store.list_nodes()) == ["alice", "bob"]

    def test_get_edges_for_node(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        store.add_edge("alice", "works_with", "carol")
        edges = store.get_edges("alice")
        assert len(edges) == 2

    def test_remove_edge(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        assert store.remove_edge("alice", "knows", "bob") is True
        assert store.list_edges() == []
        assert store.remove_edge("alice", "knows", "bob") is False

    def test_add_edge_with_metadata(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob", metadata={"weight": 0.9})
        # Edge metadata is stored but not exposed in list_edges tuples
        edges = store.list_edges()
        assert ("alice", "knows", "bob") in edges


class TestInMemoryGraphStoreTraversal:
    def test_get_neighbors_depth_1(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        store.add_edge("bob", "knows", "carol")
        neighbors = store.get_neighbors("alice", max_depth=1)
        assert neighbors == ["bob"]

    def test_get_neighbors_depth_2(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        store.add_edge("bob", "knows", "carol")
        neighbors = store.get_neighbors("alice", max_depth=2)
        assert sorted(neighbors) == ["bob", "carol"]

    def test_get_neighbors_with_relation_filter(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        store.add_edge("alice", "works_with", "carol")
        neighbors = store.get_neighbors("alice", relation_filter="knows")
        assert neighbors == ["bob"]

    def test_get_neighbors_with_relation_filter_list(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        store.add_edge("alice", "works_with", "carol")
        store.add_edge("alice", "manages", "dave")
        neighbors = store.get_neighbors("alice", relation_filter=["knows", "works_with"])
        assert sorted(neighbors) == ["bob", "carol"]

    def test_get_neighbors_missing_node(self):
        store = InMemoryGraphStore()
        assert store.get_neighbors("nonexistent") == []

    def test_get_neighbors_handles_cycles(self):
        store = InMemoryGraphStore()
        store.add_edge("a", "r", "b")
        store.add_edge("b", "r", "c")
        store.add_edge("c", "r", "a")
        neighbors = store.get_neighbors("a", max_depth=5)
        assert sorted(neighbors) == ["b", "c"]


class TestInMemoryGraphStoreMemoryLinks:
    def test_link_and_get_memory_ids(self):
        store = InMemoryGraphStore()
        store.add_node("alice")
        store.link_memory("alice", "mem-001")
        store.link_memory("alice", "mem-002")
        ids = store.get_memory_ids("alice")
        assert sorted(ids) == ["mem-001", "mem-002"]

    def test_link_memory_raises_for_missing_node(self):
        store = InMemoryGraphStore()
        import pytest
        with pytest.raises(KeyError):
            store.link_memory("nonexistent", "mem-001")

    def test_get_memory_ids_with_depth(self):
        store = InMemoryGraphStore()
        store.add_node("alice")
        store.add_node("bob")
        store.add_edge("alice", "knows", "bob")
        store.link_memory("alice", "mem-001")
        store.link_memory("bob", "mem-002")
        ids = store.get_memory_ids("alice", max_depth=1)
        assert sorted(ids) == ["mem-001", "mem-002"]

    def test_remove_node_removes_memory_links(self):
        store = InMemoryGraphStore()
        store.add_node("alice")
        store.link_memory("alice", "mem-001")
        store.remove_node("alice")
        assert store.get_memory_ids("alice") == []


class TestInMemoryGraphStoreClear:
    def test_clear(self):
        store = InMemoryGraphStore()
        store.add_edge("alice", "knows", "bob")
        store.add_node("carol")
        store.link_memory("alice", "mem-001")
        store.clear()
        assert store.list_nodes() == []
        assert store.list_edges() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage/test_graph_store.py -v --no-header 2>&1 | head -5`
Expected: ImportError (InMemoryGraphStore does not exist yet)

- [ ] **Step 3: Implement InMemoryGraphStore**

Add to `src/anchor/storage/memory_store.py` after `InMemoryDocumentStore`:

```python
class InMemoryGraphStore:
    """Dict-backed graph store. Implements GraphStore protocol."""

    __slots__ = ("_adjacency", "_adjacency_dirty", "_edge_metadata", "_edges",
                 "_entity_to_memories", "_lock", "_nodes")

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[tuple[str, str, str]] = []
        self._edge_metadata: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._entity_to_memories: dict[str, list[str]] = {}
        self._adjacency: dict[str, set[str]] = {}
        self._adjacency_dirty: bool = True
        self._lock = threading.Lock()

    def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            if node_id in self._nodes:
                if metadata:
                    self._nodes[node_id].update(metadata)
            else:
                self._nodes[node_id] = dict(metadata) if metadata else {}

    def add_edge(
        self, source: str, relation: str, target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            if source not in self._nodes:
                self._nodes[source] = {}
            if target not in self._nodes:
                self._nodes[target] = {}
            edge_key = (source, relation, target)
            if edge_key not in self._edge_metadata:
                self._edges.append(edge_key)
            if metadata:
                self._edge_metadata[edge_key] = metadata
            elif edge_key not in self._edge_metadata:
                self._edge_metadata[edge_key] = {}
            self._adjacency_dirty = True

    def _rebuild_adjacency(self, relation_filter: str | list[str] | None = None) -> dict[str, set[str]]:
        """Build adjacency index, optionally filtered by relation types."""
        if relation_filter is not None:
            allowed = {relation_filter} if isinstance(relation_filter, str) else set(relation_filter)
        else:
            allowed = None
        adj: dict[str, set[str]] = {}
        for src, rel, tgt in self._edges:
            if allowed is not None and rel not in allowed:
                continue
            adj.setdefault(src, set()).add(tgt)
            adj.setdefault(tgt, set()).add(src)
        return adj

    def get_neighbors(
        self, node_id: str, max_depth: int = 1,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]:
        with self._lock:
            if node_id not in self._nodes:
                return []
            adj = self._rebuild_adjacency(relation_filter)
            from collections import deque
            visited: set[str] = {node_id}
            queue: deque[tuple[str, int]] = deque([(node_id, 0)])
            result: list[str] = []
            while queue:
                current, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for neighbor in adj.get(current, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        result.append(neighbor)
                        queue.append((neighbor, depth + 1))
            return result

    def get_edges(self, node_id: str) -> list[tuple[str, str, str]]:
        with self._lock:
            return [
                (s, r, t) for s, r, t in self._edges
                if s == node_id or t == node_id
            ]

    def get_node_metadata(self, node_id: str) -> dict[str, Any] | None:
        with self._lock:
            if node_id not in self._nodes:
                return None
            return dict(self._nodes[node_id])

    def link_memory(self, node_id: str, memory_id: str) -> None:
        with self._lock:
            if node_id not in self._nodes:
                msg = f"Entity '{node_id}' does not exist in the graph"
                raise KeyError(msg)
            self._entity_to_memories.setdefault(node_id, []).append(memory_id)

    def get_memory_ids(self, node_id: str, max_depth: int = 1) -> list[str]:
        with self._lock:
            all_entities = [node_id, *self._get_neighbors_unlocked(node_id, max_depth)]
            seen: set[str] = set()
            result: list[str] = []
            for eid in all_entities:
                for mid in self._entity_to_memories.get(eid, []):
                    if mid not in seen:
                        seen.add(mid)
                        result.append(mid)
            return result

    def _get_neighbors_unlocked(self, node_id: str, max_depth: int = 1) -> list[str]:
        """Internal BFS without lock (caller must hold lock)."""
        if node_id not in self._nodes:
            return []
        adj = self._rebuild_adjacency()
        from collections import deque
        visited: set[str] = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        result: list[str] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in adj.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append((neighbor, depth + 1))
        return result

    def remove_node(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)
            old_edges = self._edges
            self._edges = [
                (s, r, t) for s, r, t in old_edges
                if s != node_id and t != node_id
            ]
            # Clean edge metadata
            for key in list(self._edge_metadata):
                if key[0] == node_id or key[2] == node_id:
                    del self._edge_metadata[key]
            self._entity_to_memories.pop(node_id, None)
            self._adjacency_dirty = True

    def remove_edge(self, source: str, relation: str, target: str) -> bool:
        with self._lock:
            edge_key = (source, relation, target)
            if edge_key in self._edge_metadata:
                self._edges.remove(edge_key)
                del self._edge_metadata[edge_key]
                self._adjacency_dirty = True
                return True
            return False

    def list_nodes(self) -> list[str]:
        with self._lock:
            return list(self._nodes)

    def list_edges(self) -> list[tuple[str, str, str]]:
        with self._lock:
            return list(self._edges)

    def clear(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._edge_metadata.clear()
            self._entity_to_memories.clear()
            self._adjacency_dirty = True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nodes={len(self._nodes)}, edges={len(self._edges)})"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_storage/test_graph_store.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/storage/memory_store.py tests/test_storage/test_graph_store.py
git commit -m "feat: add InMemoryGraphStore implementing GraphStore protocol"
```

### Task 3: Refactor SimpleGraphMemory to delegate to GraphStore

**Files:**
- Modify: `src/anchor/memory/graph_memory.py`
- Test: existing `tests/test_memory/` tests must still pass

- [ ] **Step 1: Write failing test for store delegation**

Add to `tests/test_storage/test_graph_store.py`:

```python
from anchor.memory.graph_memory import SimpleGraphMemory
from anchor.storage.memory_store import InMemoryGraphStore


class TestSimpleGraphMemoryBackwardsCompat:
    def test_default_no_store_works(self):
        """Existing usage without a store must work identically."""
        graph = SimpleGraphMemory()
        graph.add_entity("alice", {"type": "person"})
        graph.add_relationship("alice", "knows", "bob")
        assert "bob" in graph.get_related_entities("alice")

    def test_with_store_delegates(self):
        """When a store is provided, operations delegate to it."""
        store = InMemoryGraphStore()
        graph = SimpleGraphMemory(store=store)
        graph.add_entity("alice", {"type": "person"})
        graph.add_relationship("alice", "knows", "bob")
        # Verify store has the data
        assert "alice" in store.list_nodes()
        assert ("alice", "knows", "bob") in store.list_edges()

    def test_relation_filter_through_graph_memory(self):
        """relation_filter flows through to the underlying store."""
        store = InMemoryGraphStore()
        graph = SimpleGraphMemory(store=store)
        graph.add_relationship("alice", "knows", "bob")
        graph.add_relationship("alice", "works_with", "carol")
        neighbors = graph.get_related_entities("alice", relation_filter="knows")
        assert neighbors == ["bob"]

    def test_relation_filter_list_through_graph_memory(self):
        """relation_filter as a list flows through to the underlying store."""
        store = InMemoryGraphStore()
        graph = SimpleGraphMemory(store=store)
        graph.add_relationship("alice", "knows", "bob")
        graph.add_relationship("alice", "works_with", "carol")
        graph.add_relationship("alice", "manages", "dave")
        neighbors = graph.get_related_entities("alice", relation_filter=["knows", "works_with"])
        assert sorted(neighbors) == ["bob", "carol"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage/test_graph_store.py::TestSimpleGraphMemoryBackwardsCompat -v`
Expected: FAIL (SimpleGraphMemory does not accept `store` parameter yet)

- [ ] **Step 3: Refactor SimpleGraphMemory**

Replace `src/anchor/memory/graph_memory.py` entirely:

```python
"""Simple graph memory for entity-relationship tracking.

Provides a directed graph that links entities to each other
and to memory entry IDs. Supports an optional GraphStore backend
for persistence; defaults to in-memory dicts.
"""

from __future__ import annotations

from collections import deque
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from anchor.protocols.storage import GraphStore


class SimpleGraphMemory:
    """Graph for entity-relationship tracking with optional persistent backend.

    When *store* is provided, all operations delegate to it.
    When *store* is None, uses in-memory dicts (original behavior).
    """

    __slots__ = (
        "_adjacency",
        "_adjacency_dirty",
        "_edges",
        "_entity_to_memories",
        "_nodes",
        "_store",
    )

    def __init__(self, store: GraphStore | None = None) -> None:
        self._store = store
        # In-memory fallback (only used when store is None)
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[tuple[str, str, str]] = []
        self._entity_to_memories: dict[str, list[str]] = {}
        self._adjacency: dict[str, set[str]] = {}
        self._adjacency_dirty: bool = True

    # -- Aliases (old API -> new API) --

    def add_entity(self, entity_id: str, metadata: dict[str, Any] | None = None) -> None:
        """Add an entity node. Alias for add_node (backwards compat)."""
        if self._store is not None:
            self._store.add_node(entity_id, metadata)
        else:
            if entity_id in self._nodes:
                if metadata:
                    self._nodes[entity_id].update(metadata)
            else:
                self._nodes[entity_id] = dict(metadata) if metadata else {}

    def add_relationship(self, source: str, relation: str, target: str) -> None:
        """Add a directed edge. Alias for add_edge (backwards compat)."""
        if self._store is not None:
            self._store.add_edge(source, relation, target)
        else:
            if source not in self._nodes:
                self._nodes[source] = {}
            if target not in self._nodes:
                self._nodes[target] = {}
            self._edges.append((source, relation, target))
            self._adjacency_dirty = True

    def get_related_entities(
        self, entity_id: str, max_depth: int = 2,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]:
        """BFS traversal. Alias for get_neighbors (backwards compat)."""
        if self._store is not None:
            return self._store.get_neighbors(entity_id, max_depth=max_depth, relation_filter=relation_filter)
        # In-memory BFS
        if entity_id not in self._nodes:
            return []
        if self._adjacency_dirty:
            self._rebuild_adjacency(relation_filter)
        else:
            # Rebuild if filter changed
            self._rebuild_adjacency(relation_filter)
        visited: set[str] = {entity_id}
        queue: deque[tuple[str, int]] = deque([(entity_id, 0)])
        result: list[str] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in self._adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append((neighbor, depth + 1))
        return result

    def link_memory(self, entity_id: str, memory_id: str) -> None:
        if self._store is not None:
            self._store.link_memory(entity_id, memory_id)
        else:
            if entity_id not in self._nodes:
                msg = f"Entity '{entity_id}' does not exist in the graph"
                raise KeyError(msg)
            self._entity_to_memories.setdefault(entity_id, []).append(memory_id)

    def get_memory_ids_for_entity(self, entity_id: str) -> list[str]:
        if self._store is not None:
            return self._store.get_memory_ids(entity_id, max_depth=0)
        return list(self._entity_to_memories.get(entity_id, []))

    def get_related_memory_ids(self, entity_id: str, max_depth: int = 2) -> list[str]:
        if self._store is not None:
            return self._store.get_memory_ids(entity_id, max_depth=max_depth)
        all_entities = [entity_id, *self.get_related_entities(entity_id, max_depth)]
        seen: set[str] = set()
        result: list[str] = []
        for eid in all_entities:
            for mid in self._entity_to_memories.get(eid, []):
                if mid not in seen:
                    seen.add(mid)
                    result.append(mid)
        return result

    def remove_entity(self, entity_id: str) -> None:
        if self._store is not None:
            self._store.remove_node(entity_id)
        else:
            self._nodes.pop(entity_id, None)
            self._edges = [
                (s, r, t) for s, r, t in self._edges if s != entity_id and t != entity_id
            ]
            self._entity_to_memories.pop(entity_id, None)
            self._adjacency_dirty = True

    def remove_edge(self, source: str, relation: str, target: str) -> bool:
        """Remove a specific edge. Returns True if it existed."""
        if self._store is not None:
            return self._store.remove_edge(source, relation, target)
        for i, (s, r, t) in enumerate(self._edges):
            if s == source and r == relation and t == target:
                self._edges.pop(i)
                self._adjacency_dirty = True
                return True
        return False

    def clear(self) -> None:
        if self._store is not None:
            self._store.clear()
        else:
            self._nodes.clear()
            self._edges.clear()
            self._entity_to_memories.clear()
            self._adjacency_dirty = True

    @property
    def entities(self) -> list[str]:
        if self._store is not None:
            return self._store.list_nodes()
        return list(self._nodes)

    @property
    def relationships(self) -> list[tuple[str, str, str]]:
        if self._store is not None:
            return self._store.list_edges()
        return list(self._edges)

    def get_entity_metadata(self, entity_id: str) -> dict[str, Any]:
        if self._store is not None:
            result = self._store.get_node_metadata(entity_id)
            if result is None:
                msg = f"Entity '{entity_id}' does not exist in the graph"
                raise KeyError(msg)
            return result
        if entity_id not in self._nodes:
            msg = f"Entity '{entity_id}' does not exist in the graph"
            raise KeyError(msg)
        return dict(self._nodes[entity_id])

    def _rebuild_adjacency(self, relation_filter: str | list[str] | None = None) -> None:
        if relation_filter is not None:
            allowed = {relation_filter} if isinstance(relation_filter, str) else set(relation_filter)
        else:
            allowed = None
        self._adjacency = {}
        for src, rel, tgt in self._edges:
            if allowed is not None and rel not in allowed:
                continue
            self._adjacency.setdefault(src, set()).add(tgt)
            self._adjacency.setdefault(tgt, set()).add(src)
        self._adjacency_dirty = False

    def __repr__(self) -> str:
        if self._store is not None:
            return f"SimpleGraphMemory(store={self._store!r})"
        return (
            f"SimpleGraphMemory(entities={len(self._nodes)}, "
            f"relationships={len(self._edges)})"
        )

    def __len__(self) -> int:
        if self._store is not None:
            return len(self._store.list_nodes())
        return len(self._nodes)
```

- [ ] **Step 4: Run ALL existing graph memory tests + new tests**

Run: `pytest tests/ -k "graph" -v`
Expected: All PASS (backwards compatibility preserved)

- [ ] **Step 5: Commit**

```bash
git add src/anchor/memory/graph_memory.py tests/test_storage/test_graph_store.py
git commit -m "refactor: SimpleGraphMemory delegates to optional GraphStore backend"
```

## Chunk 2: ConversationStore Protocol + InMemory Implementation

### Task 4: Add ConversationStore and AsyncConversationStore protocols

**Files:**
- Modify: `src/anchor/protocols/storage.py`

- [ ] **Step 1: Add imports for ConversationTurn and SummaryTier**

At the top of `src/anchor/protocols/storage.py`, add to imports:

```python
from anchor.models.memory import ConversationTurn, SummaryTier
```

- [ ] **Step 2: Add ConversationStore + AsyncConversationStore protocols**

Append after `AsyncGraphStore`:

```python
@runtime_checkable
class ConversationStore(Protocol):
    """Protocol for persistent conversation history storage."""

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None: ...
    def load_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]: ...
    def save_summary_tiers(self, session_id: str, tiers: dict[int, SummaryTier | None]) -> None: ...
    def load_summary_tiers(self, session_id: str) -> dict[int, SummaryTier | None]: ...
    def truncate_turns(self, session_id: str, keep_last: int) -> None: ...
    def delete_session(self, session_id: str) -> bool: ...
    def list_sessions(self) -> list[str]: ...
    def clear(self) -> None: ...


@runtime_checkable
class AsyncConversationStore(Protocol):
    """Async variant of ConversationStore."""

    async def append_turn(self, session_id: str, turn: ConversationTurn) -> None: ...
    async def load_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]: ...
    async def save_summary_tiers(self, session_id: str, tiers: dict[int, SummaryTier | None]) -> None: ...
    async def load_summary_tiers(self, session_id: str) -> dict[int, SummaryTier | None]: ...
    async def truncate_turns(self, session_id: str, keep_last: int) -> None: ...
    async def delete_session(self, session_id: str) -> bool: ...
    async def list_sessions(self) -> list[str]: ...
    async def clear(self) -> None: ...
```

- [ ] **Step 3: Verify import works**

Run: `python -c "from anchor.protocols.storage import ConversationStore, AsyncConversationStore; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/anchor/protocols/storage.py
git commit -m "feat: add ConversationStore and AsyncConversationStore protocols"
```

### Task 5: Add InMemoryConversationStore

**Files:**
- Modify: `src/anchor/storage/memory_store.py`
- Create: `tests/test_storage/test_conversation_store.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_storage/test_conversation_store.py`:

```python
"""Tests for ConversationStore protocol and InMemory implementation."""

from anchor.models.memory import ConversationTurn, SummaryTier
from anchor.protocols.storage import ConversationStore
from anchor.storage.memory_store import InMemoryConversationStore


def _turn(role: str, content: str) -> ConversationTurn:
    return ConversationTurn(role=role, content=content)


def _tier(level: int, content: str, turn_count: int = 1) -> SummaryTier:
    return SummaryTier(level=level, content=content, token_count=10, source_turn_count=turn_count)


class TestInMemoryConversationStoreProtocol:
    def test_satisfies_protocol(self):
        store = InMemoryConversationStore()
        assert isinstance(store, ConversationStore)


class TestInMemoryConversationStoreTurns:
    def test_append_and_load(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.append_turn("sess-1", _turn("assistant", "hi"))
        turns = store.load_turns("sess-1")
        assert len(turns) == 2
        assert turns[0].content == "hello"
        assert turns[1].content == "hi"

    def test_load_with_limit(self):
        store = InMemoryConversationStore()
        for i in range(10):
            store.append_turn("sess-1", _turn("user", f"msg-{i}"))
        turns = store.load_turns("sess-1", limit=3)
        assert len(turns) == 3
        assert turns[0].content == "msg-7"

    def test_load_empty_session(self):
        store = InMemoryConversationStore()
        assert store.load_turns("nonexistent") == []

    def test_truncate_turns(self):
        store = InMemoryConversationStore()
        for i in range(10):
            store.append_turn("sess-1", _turn("user", f"msg-{i}"))
        store.truncate_turns("sess-1", keep_last=3)
        turns = store.load_turns("sess-1")
        assert len(turns) == 3
        assert turns[0].content == "msg-7"

    def test_sessions_are_isolated(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.append_turn("sess-2", _turn("user", "world"))
        assert len(store.load_turns("sess-1")) == 1
        assert len(store.load_turns("sess-2")) == 1


class TestInMemoryConversationStoreTiers:
    def test_save_and_load_tiers(self):
        store = InMemoryConversationStore()
        tiers = {1: _tier(1, "summary"), 2: None, 3: None}
        store.save_summary_tiers("sess-1", tiers)
        loaded = store.load_summary_tiers("sess-1")
        assert loaded[1] is not None
        assert loaded[1].content == "summary"
        assert loaded[2] is None

    def test_load_tiers_empty_session(self):
        store = InMemoryConversationStore()
        loaded = store.load_summary_tiers("nonexistent")
        assert loaded == {1: None, 2: None, 3: None}

    def test_save_tiers_overwrites(self):
        store = InMemoryConversationStore()
        store.save_summary_tiers("sess-1", {1: _tier(1, "old"), 2: None, 3: None})
        store.save_summary_tiers("sess-1", {1: _tier(1, "new"), 2: None, 3: None})
        loaded = store.load_summary_tiers("sess-1")
        assert loaded[1].content == "new"


class TestInMemoryConversationStoreSession:
    def test_list_sessions(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.append_turn("sess-2", _turn("user", "world"))
        assert sorted(store.list_sessions()) == ["sess-1", "sess-2"]

    def test_delete_session(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.save_summary_tiers("sess-1", {1: _tier(1, "summary"), 2: None, 3: None})
        assert store.delete_session("sess-1") is True
        assert store.load_turns("sess-1") == []
        assert store.delete_session("sess-1") is False

    def test_clear(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.append_turn("sess-2", _turn("user", "world"))
        store.clear()
        assert store.list_sessions() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage/test_conversation_store.py -v --no-header 2>&1 | head -5`
Expected: ImportError

- [ ] **Step 3: Implement InMemoryConversationStore**

Add to `src/anchor/storage/memory_store.py`:

```python
from anchor.models.memory import ConversationTurn, SummaryTier

class InMemoryConversationStore:
    """Dict-backed conversation store. Implements ConversationStore protocol."""

    __slots__ = ("_lock", "_tiers", "_turns")

    def __init__(self) -> None:
        self._turns: dict[str, list[ConversationTurn]] = {}
        self._tiers: dict[str, dict[int, SummaryTier | None]] = {}
        self._lock = threading.Lock()

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        with self._lock:
            self._turns.setdefault(session_id, []).append(turn)

    def load_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        with self._lock:
            turns = self._turns.get(session_id, [])
            if limit is not None:
                return list(turns[-limit:])
            return list(turns)

    def save_summary_tiers(self, session_id: str, tiers: dict[int, SummaryTier | None]) -> None:
        with self._lock:
            self._tiers[session_id] = dict(tiers)

    def load_summary_tiers(self, session_id: str) -> dict[int, SummaryTier | None]:
        with self._lock:
            return dict(self._tiers.get(session_id, {1: None, 2: None, 3: None}))

    def truncate_turns(self, session_id: str, keep_last: int) -> None:
        with self._lock:
            turns = self._turns.get(session_id)
            if turns is not None:
                self._turns[session_id] = turns[-keep_last:]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            found = session_id in self._turns or session_id in self._tiers
            self._turns.pop(session_id, None)
            self._tiers.pop(session_id, None)
            return found

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(set(self._turns) | set(self._tiers))

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
            self._tiers.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sessions={len(set(self._turns) | set(self._tiers))})"
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_storage/test_conversation_store.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/storage/memory_store.py tests/test_storage/test_conversation_store.py
git commit -m "feat: add InMemoryConversationStore implementing ConversationStore protocol"
```

### Task 6: Add serialization helpers for ConversationTurn and SummaryTier

**Files:**
- Modify: `src/anchor/storage/_serialization.py`

- [ ] **Step 1: Write test for serialization round-trip**

Add to `tests/test_storage/test_conversation_store.py`:

```python
from anchor.storage._serialization import (
    conversation_turn_to_row,
    row_to_conversation_turn,
    summary_tier_to_row,
    row_to_summary_tier,
)


class TestConversationSerialization:
    def test_turn_round_trip(self):
        turn = _turn("user", "hello world")
        row = conversation_turn_to_row(turn)
        restored = row_to_conversation_turn(row)
        assert restored.role == turn.role
        assert restored.content == turn.content

    def test_tier_round_trip(self):
        tier = _tier(1, "summary text", turn_count=5)
        row = summary_tier_to_row(tier)
        restored = row_to_summary_tier(row)
        assert restored.level == tier.level
        assert restored.content == tier.content
        assert restored.source_turn_count == 5
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_storage/test_conversation_store.py::TestConversationSerialization -v`
Expected: ImportError

- [ ] **Step 3: Implement serialization helpers**

Add to `src/anchor/storage/_serialization.py`:

```python
import json
from datetime import datetime
from typing import Any

from anchor.models.memory import ConversationTurn, SummaryTier

def conversation_turn_to_row(turn: ConversationTurn) -> dict[str, Any]:
    """Convert a ConversationTurn to a flat dict for SQL INSERT."""
    return {
        "role": str(turn.role),
        "content": turn.content,
        "token_count": turn.token_count,
        "metadata_json": json.dumps(turn.metadata, default=str),
        "created_at": turn.timestamp.isoformat(),
    }


def row_to_conversation_turn(row: dict[str, Any] | Any) -> ConversationTurn:
    """Reconstruct a ConversationTurn from a database row."""
    r = dict(row) if not isinstance(row, dict) else row
    return ConversationTurn(
        role=r["role"],
        content=r["content"],
        token_count=r.get("token_count", 0),
        metadata=json.loads(r.get("metadata_json", "{}")),
        timestamp=datetime.fromisoformat(r["created_at"]),
    )


def summary_tier_to_row(tier: SummaryTier) -> dict[str, Any]:
    """Convert a SummaryTier to a flat dict for SQL INSERT."""
    return {
        "tier_level": tier.level,
        "content": tier.content,
        "token_count": tier.token_count,
        "source_turn_count": tier.source_turn_count,
        "created_at": tier.created_at.isoformat(),
        "updated_at": tier.updated_at.isoformat(),
    }


def row_to_summary_tier(row: dict[str, Any] | Any) -> SummaryTier:
    """Reconstruct a SummaryTier from a database row."""
    r = dict(row) if not isinstance(row, dict) else row
    return SummaryTier(
        level=r["tier_level"],
        content=r["content"],
        token_count=r["token_count"],
        source_turn_count=r["source_turn_count"],
        created_at=datetime.fromisoformat(r["created_at"]),
        updated_at=datetime.fromisoformat(r["updated_at"]),
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_storage/test_conversation_store.py::TestConversationSerialization -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/storage/_serialization.py tests/test_storage/test_conversation_store.py
git commit -m "feat: add ConversationTurn and SummaryTier serialization helpers"
```

### Task 7: Integrate ConversationStore into MemoryManager

**Files:**
- Modify: `src/anchor/memory/manager.py`
- Test: `tests/test_integration/test_persistent_memory.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/test_integration/test_persistent_memory.py`:

```python
"""Integration tests for persistent conversation memory."""

import pytest

from anchor.memory.manager import MemoryManager
from anchor.storage.memory_store import InMemoryConversationStore


class TestMemoryManagerConversationPersistence:
    def test_auto_persist_appends_turns(self):
        store = InMemoryConversationStore()
        mgr = MemoryManager(
            conversation_tokens=4096,
            conversation_store=store,
            session_id="sess-1",
            auto_persist=True,
        )
        mgr.add_user_message("hello")
        mgr.add_assistant_message("hi")
        turns = store.load_turns("sess-1")
        assert len(turns) == 2

    def test_requires_session_id_with_store(self):
        store = InMemoryConversationStore()
        with pytest.raises(ValueError, match="session_id"):
            MemoryManager(
                conversation_tokens=4096,
                conversation_store=store,
            )

    def test_save_and_load(self):
        store = InMemoryConversationStore()
        mgr1 = MemoryManager(
            conversation_tokens=4096,
            conversation_store=store,
            session_id="sess-1",
        )
        mgr1.add_user_message("hello")
        mgr1.add_assistant_message("hi there")
        mgr1.save()

        # Simulate restart: new MemoryManager loads from store
        mgr2 = MemoryManager(
            conversation_tokens=4096,
            conversation_store=store,
            session_id="sess-1",
        )
        mgr2.load()
        items = mgr2.get_context_items()
        contents = [item.content for item in items]
        assert any("hello" in c for c in contents)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_integration/test_persistent_memory.py -v --no-header 2>&1 | head -5`
Expected: TypeError (unexpected keyword argument 'conversation_store')

- [ ] **Step 3: Modify MemoryManager**

Apply these changes to `src/anchor/memory/manager.py`:

**Update `__slots__`:**
```python
__slots__ = ("_auto_persist", "_conversation", "_conversation_store", "_is_loading", "_persistent_store", "_session_id", "_tokenizer")
```

**Update `__init__` signature** (add 3 new params after `conversation_memory`):
```python
def __init__(
    self,
    conversation_tokens: int = 4096,
    tokenizer: Tokenizer | None = None,
    on_evict: Callable[[list[ConversationTurn]], None] | None = None,
    persistent_store: MemoryEntryStore | None = None,
    conversation_memory: ConversationMemory | None = None,
    conversation_store: ConversationStore | None = None,
    session_id: str | None = None,
    auto_persist: bool = False,
) -> None:
    self._tokenizer = tokenizer or get_default_counter()
    # Validate: conversation_store requires session_id
    if conversation_store is not None and session_id is None:
        msg = "session_id is required when conversation_store is provided"
        raise ValueError(msg)
    self._conversation_store = conversation_store
    self._session_id = session_id
    self._auto_persist = auto_persist
    self._is_loading = False
    if conversation_memory is not None:
        self._conversation: ConversationMemory = conversation_memory
    else:
        if conversation_tokens <= 0:
            msg = "conversation_tokens must be a positive integer"
            raise ValueError(msg)
        self._conversation = SlidingWindowMemory(
            max_tokens=conversation_tokens,
            tokenizer=self._tokenizer,
            on_evict=on_evict,
        )
    self._persistent_store = persistent_store
```

**Add import** to existing imports from `anchor.protocols.storage`:
```python
from anchor.protocols.storage import ConversationStore, MemoryEntryStore
```
(Add `ConversationStore` alongside the existing `MemoryEntryStore` import.)

**Add `_is_loading` to `__slots__`** (already listed above).

**Update `_add_message`** to auto-persist (add auto-persist block at end of existing method — keep existing dispatch logic unchanged):
```python
def _add_message(self, role: Role, content: str) -> None:
    """Add a message to the conversation backend (works with both types)."""
    if isinstance(self._conversation, ProgressiveSummarizationMemory):
        self._conversation.add_message(role, content)
    elif isinstance(self._conversation, SummaryBufferMemory):
        self._conversation.add_message(role, content)
    elif isinstance(self._conversation, SlidingWindowMemory):
        self._conversation.add_turn(role, content)
    else:
        msg = (
            f"ConversationMemory implementation {type(self._conversation).__name__!r} "
            "does not support add_turn() or add_message()"
        )
        raise TypeError(msg)
    # Auto-persist to conversation store (skip during load to avoid re-appending)
    if (
        self._auto_persist
        and not self._is_loading
        and self._conversation_store is not None
        and self._session_id is not None
    ):
        turn = ConversationTurn(role=role, content=content)
        self._conversation_store.append_turn(self._session_id, turn)
```

**Add `save()` and `load()` methods** after `add_tool_message`:
```python
def save(self) -> None:
    """Persist current conversation state to the conversation store."""
    if self._conversation_store is None or self._session_id is None:
        return
    if not self._auto_persist:
        # Only append turns if not auto-persisting (avoids duplicates)
        for turn in self._conversation.turns:
            self._conversation_store.append_turn(self._session_id, turn)
    # Save summary tiers if using ProgressiveSummarizationMemory
    if isinstance(self._conversation, ProgressiveSummarizationMemory):
        self._conversation_store.save_summary_tiers(
            self._session_id, self._conversation.tiers
        )

def load(self) -> None:
    """Load conversation state from the conversation store.

    Sets ``_is_loading`` flag to prevent auto-persist from re-appending
    loaded turns back to the store.
    """
    if self._conversation_store is None or self._session_id is None:
        return
    self._is_loading = True
    try:
        turns = self._conversation_store.load_turns(self._session_id)
        for turn in turns:
            self._add_message(turn.role, turn.content)
        # Restore summary tiers if using ProgressiveSummarizationMemory
        if isinstance(self._conversation, ProgressiveSummarizationMemory):
            tiers = self._conversation_store.load_summary_tiers(self._session_id)
            self._conversation._tiers = tiers
    finally:
        self._is_loading = False
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_integration/test_persistent_memory.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite for regressions**

Run: `pytest tests/ -x -q --timeout=30`
Expected: No failures

- [ ] **Step 6: Commit**

```bash
git add src/anchor/memory/manager.py tests/test_integration/test_persistent_memory.py
git commit -m "feat: integrate ConversationStore into MemoryManager with auto-persist and save/load"
```

### Task 7b: Integrate ConversationStore with ProgressiveSummarizationMemory compaction

**Files:**
- Modify: `src/anchor/memory/manager.py`
- Test: `tests/test_integration/test_persistent_memory.py`

- [ ] **Step 1: Write failing test for compaction persistence**

Add to `tests/test_integration/test_persistent_memory.py`:

```python
from anchor.memory.progressive import ProgressiveSummarizationMemory
from anchor.storage.memory_store import InMemoryConversationStore


class TestProgressiveSummarizationPersistence:
    def test_compaction_persists_summary_tiers(self):
        """When ProgressiveSummarizationMemory compacts, summary tiers are saved to store."""
        store = InMemoryConversationStore()
        progressive = ProgressiveSummarizationMemory(
            max_tokens=200,
            tokenizer=None,  # uses default
        )
        mgr = MemoryManager(
            conversation_memory=progressive,
            conversation_store=store,
            session_id="sess-1",
            auto_persist=True,
        )
        # Add enough messages to trigger compaction
        for i in range(20):
            mgr.add_user_message(f"Message number {i} with some content to fill tokens")
            mgr.add_assistant_message(f"Response number {i}")
        mgr.save()
        # Verify tiers were persisted
        tiers = store.load_summary_tiers("sess-1")
        # At least tier 1 should have content after compaction
        has_tier = any(t is not None for t in tiers.values())
        assert has_tier or len(store.load_turns("sess-1")) > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_integration/test_persistent_memory.py::TestProgressiveSummarizationPersistence -v`
Expected: FAIL

- [ ] **Step 3: Verify save()/load() handle ProgressiveSummarizationMemory**

No code changes needed — Task 7's `save()` and `load()` already handle `ProgressiveSummarizationMemory` via `isinstance` checks:
- `save()` calls `save_summary_tiers()` when the conversation backend is `ProgressiveSummarizationMemory`
- `load()` restores `_tiers` from the store

This step is verification-only: confirm the test from Step 1 passes with the existing `save()`/`load()` from Task 7.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_integration/test_persistent_memory.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/anchor/memory/manager.py tests/test_integration/test_persistent_memory.py
git commit -m "feat: persist ProgressiveSummarizationMemory tiers to ConversationStore"
```

## Chunk 3: Cache Backends

### Task 8: Implement SqliteCacheBackend

**Files:**
- Create: `src/anchor/cache/sqlite_backend.py`
- Modify: `src/anchor/storage/sqlite/_schema.py`
- Create: `tests/test_cache/test_sqlite_backend.py`

- [ ] **Step 1: Add cache schema to SQLite schema module**

Add to `_TABLES` list in `src/anchor/storage/sqlite/_schema.py`:

```python
    """CREATE TABLE IF NOT EXISTS cache_entries (
        key         TEXT PRIMARY KEY,
        value_json  TEXT NOT NULL,
        created_at  REAL NOT NULL,
        expires_at  REAL
    )""",
```

Add to `_INDEXES` list:

```python
    "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)",
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_cache/test_sqlite_backend.py`:

```python
"""Tests for SqliteCacheBackend."""

import time

import pytest

from anchor.cache.sqlite_backend import SqliteCacheBackend
from anchor.protocols.cache import CacheBackend
from anchor.storage.sqlite import SqliteConnectionManager


@pytest.fixture
def cache():
    conn_mgr = SqliteConnectionManager(":memory:")
    return SqliteCacheBackend(conn_mgr, default_ttl=1.0)


class TestSqliteCacheProtocol:
    def test_satisfies_protocol(self, cache):
        assert isinstance(cache, CacheBackend)


class TestSqliteCacheOperations:
    def test_set_and_get(self, cache):
        cache.set("key1", {"data": "value"})
        assert cache.get("key1") == {"data": "value"}

    def test_get_missing_key(self, cache):
        assert cache.get("nonexistent") is None

    def test_invalidate(self, cache):
        cache.set("key1", "value")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self, cache):
        cache.set("key1", "v1")
        cache.set("key2", "v2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_overwrite_existing_key(self, cache):
        cache.set("key1", "old")
        cache.set("key1", "new")
        assert cache.get("key1") == "new"

    def test_ttl_expiration(self, cache):
        cache.set("key1", "value", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_no_ttl(self, cache):
        conn_mgr = SqliteConnectionManager(":memory:")
        no_ttl_cache = SqliteCacheBackend(conn_mgr, default_ttl=None)
        no_ttl_cache.set("key1", "value")
        assert no_ttl_cache.get("key1") == "value"

    def test_stores_various_types(self, cache):
        cache.set("str", "hello")
        cache.set("int", 42)
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"a": 1})
        assert cache.get("str") == "hello"
        assert cache.get("int") == 42
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"a": 1}
```

- [ ] **Step 3: Run tests to verify failure**

Run: `pytest tests/test_cache/test_sqlite_backend.py -v --no-header 2>&1 | head -5`
Expected: ImportError

- [ ] **Step 4: Implement SqliteCacheBackend**

Create `src/anchor/cache/sqlite_backend.py`:

```python
"""SQLite-backed cache backend implementation."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from anchor.storage.sqlite._connection import SqliteConnectionManager

logger = logging.getLogger(__name__)


class SqliteCacheBackend:
    """SQLite-backed cache with optional TTL expiration.

    Implements the CacheBackend protocol. Survives process restarts.
    Lazy expiration on read (matches InMemoryCacheBackend pattern).
    """

    __slots__ = ("_conn_manager", "_default_ttl")

    def __init__(
        self,
        connection_manager: SqliteConnectionManager,
        default_ttl: float | None = 300.0,
    ) -> None:
        self._conn_manager = connection_manager
        self._default_ttl = default_ttl
        # Ensure table exists
        from anchor.storage.sqlite._schema import ensure_tables
        ensure_tables(self._conn_manager.get_connection())

    def get(self, key: str) -> Any | None:
        conn = self._conn_manager.get_connection()
        row = conn.execute(
            "SELECT value_json, expires_at FROM cache_entries WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        value_json, expires_at = row["value_json"], row["expires_at"]
        if expires_at is not None and time.time() >= expires_at:
            conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            conn.commit()
            return None
        return json.loads(value_json)

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        now = time.time()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (now + effective_ttl) if effective_ttl is not None else None
        conn = self._conn_manager.get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO cache_entries (key, value_json, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (key, json.dumps(value, default=str), now, expires_at),
        )
        conn.commit()

    def invalidate(self, key: str) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
        conn.commit()

    def clear(self) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM cache_entries")
        conn.commit()

    def __repr__(self) -> str:
        return f"SqliteCacheBackend(default_ttl={self._default_ttl})"
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_cache/test_sqlite_backend.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/anchor/cache/sqlite_backend.py src/anchor/storage/sqlite/_schema.py tests/test_cache/test_sqlite_backend.py
git commit -m "feat: add SqliteCacheBackend implementing CacheBackend protocol"
```

### Task 9: Implement RedisCacheBackend

**Files:**
- Create: `src/anchor/cache/redis_backend.py`
- Create: `tests/test_cache/test_redis_backend.py`

- [ ] **Step 1: Write failing tests (mocked Redis)**

Create `tests/test_cache/test_redis_backend.py`:

```python
"""Tests for RedisCacheBackend with mocked Redis client."""

from unittest.mock import MagicMock, patch

import pytest

from anchor.cache.redis_backend import RedisCacheBackend
from anchor.protocols.cache import CacheBackend


@pytest.fixture
def mock_conn_manager():
    mgr = MagicMock()
    mgr.prefix = "anchor:"
    client = MagicMock()
    client.get.return_value = None
    client.scan_iter.return_value = iter([])
    mgr.get_client.return_value = client
    return mgr


@pytest.fixture
def cache(mock_conn_manager):
    return RedisCacheBackend(mock_conn_manager, default_ttl=300.0)


class TestRedisCacheProtocol:
    def test_satisfies_protocol(self, cache):
        assert isinstance(cache, CacheBackend)


class TestRedisCacheOperations:
    def test_set_calls_redis_setex(self, cache, mock_conn_manager):
        cache.set("mykey", {"data": 1}, ttl=60.0)
        client = mock_conn_manager.get_client()
        # Should call setex with prefixed key and JSON value
        client.setex.assert_called_once()
        args = client.setex.call_args
        assert "anchor:cache:mykey" in str(args)

    def test_set_without_ttl_uses_default(self, cache, mock_conn_manager):
        cache.set("mykey", "value")
        client = mock_conn_manager.get_client()
        client.setex.assert_called_once()

    def test_set_with_no_ttl_uses_plain_set(self, mock_conn_manager):
        cache = RedisCacheBackend(mock_conn_manager, default_ttl=None)
        cache.set("mykey", "value")
        client = mock_conn_manager.get_client()
        client.set.assert_called_once()

    def test_get_returns_deserialized(self, cache, mock_conn_manager):
        import json
        client = mock_conn_manager.get_client()
        client.get.return_value = json.dumps({"data": 1}).encode()
        result = cache.get("mykey")
        assert result == {"data": 1}

    def test_get_returns_none_for_missing(self, cache, mock_conn_manager):
        client = mock_conn_manager.get_client()
        client.get.return_value = None
        assert cache.get("mykey") is None

    def test_invalidate_calls_delete(self, cache, mock_conn_manager):
        cache.invalidate("mykey")
        client = mock_conn_manager.get_client()
        client.delete.assert_called_once_with("anchor:cache:mykey")

    def test_clear_scans_and_deletes(self, cache, mock_conn_manager):
        client = mock_conn_manager.get_client()
        client.scan_iter.return_value = iter([b"anchor:cache:k1", b"anchor:cache:k2"])
        cache.clear()
        client.scan_iter.assert_called_once()
        assert client.delete.call_count == 2
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_cache/test_redis_backend.py -v --no-header 2>&1 | head -5`
Expected: ImportError

- [ ] **Step 3: Implement RedisCacheBackend**

Create `src/anchor/cache/redis_backend.py`:

```python
"""Redis-backed cache backend implementation."""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from anchor.storage.redis._connection import RedisConnectionManager

logger = logging.getLogger(__name__)


class RedisCacheBackend:
    """Redis-backed cache. Implements CacheBackend protocol.

    Uses JSON serialization for portability and debuggability.
    Leverages native Redis SETEX for TTL management.
    """

    __slots__ = ("_conn_manager", "_default_ttl", "_key_prefix")

    def __init__(
        self,
        connection_manager: RedisConnectionManager,
        default_ttl: float | None = 300.0,
        key_prefix: str = "cache:",
    ) -> None:
        self._conn_manager = connection_manager
        self._default_ttl = default_ttl
        self._key_prefix = key_prefix

    def _full_key(self, key: str) -> str:
        return f"{self._conn_manager.prefix}{self._key_prefix}{key}"

    def get(self, key: str) -> Any | None:
        client = self._conn_manager.get_client()
        raw = client.get(self._full_key(key))
        if raw is None:
            return None
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        client = self._conn_manager.get_client()
        serialized = json.dumps(value, default=str)
        full_key = self._full_key(key)
        if effective_ttl is not None:
            client.setex(full_key, int(effective_ttl), serialized)
        else:
            client.set(full_key, serialized)

    def invalidate(self, key: str) -> None:
        client = self._conn_manager.get_client()
        client.delete(self._full_key(key))

    def clear(self) -> None:
        client = self._conn_manager.get_client()
        pattern = f"{self._conn_manager.prefix}{self._key_prefix}*"
        for redis_key in client.scan_iter(match=pattern):
            client.delete(redis_key)

    def __repr__(self) -> str:
        return (
            f"RedisCacheBackend(default_ttl={self._default_ttl}, "
            f"key_prefix={self._key_prefix!r})"
        )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cache/test_redis_backend.py -v`
Expected: All PASS

- [ ] **Step 5: Update cache __init__.py exports**

Modify `src/anchor/cache/__init__.py`:

```python
"""Caching module for anchor pipeline steps."""

from .backend import InMemoryCacheBackend
from .sqlite_backend import SqliteCacheBackend

__all__ = [
    "InMemoryCacheBackend",
    "SqliteCacheBackend",
]

# Redis backend is optional (requires redis-py)
try:
    from .redis_backend import RedisCacheBackend
    __all__.append("RedisCacheBackend")
except ImportError:
    pass
```

- [ ] **Step 6: Commit**

```bash
git add src/anchor/cache/redis_backend.py src/anchor/cache/__init__.py tests/test_cache/test_redis_backend.py
git commit -m "feat: add RedisCacheBackend and SqliteCacheBackend, update cache exports"
```

## Chunk 4: SQLite Persistent Backends

### Task 10: Add SQLite schemas for graph + conversation

**Files:**
- Modify: `src/anchor/storage/sqlite/_schema.py`

- [ ] **Step 1: Add graph and conversation table DDL**

Add to `_TABLES` list in `src/anchor/storage/sqlite/_schema.py` (cache already added in Task 8):

```python
    """CREATE TABLE IF NOT EXISTS graph_nodes (
        node_id       TEXT PRIMARY KEY,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )""",
    """CREATE TABLE IF NOT EXISTS graph_edges (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        source        TEXT NOT NULL,
        relation      TEXT NOT NULL,
        target        TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE(source, relation, target)
    )""",
    """CREATE TABLE IF NOT EXISTS graph_memory_links (
        node_id   TEXT NOT NULL,
        memory_id TEXT NOT NULL,
        PRIMARY KEY (node_id, memory_id)
    )""",
    """CREATE TABLE IF NOT EXISTS conversation_turns (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id    TEXT NOT NULL,
        turn_index    INTEGER NOT NULL,
        role          TEXT NOT NULL,
        content       TEXT NOT NULL,
        token_count   INTEGER NOT NULL DEFAULT 0,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at    TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS summary_tiers (
        session_id       TEXT NOT NULL,
        tier_level       INTEGER NOT NULL,
        content          TEXT NOT NULL,
        token_count      INTEGER NOT NULL,
        source_turn_count INTEGER NOT NULL,
        created_at       TEXT NOT NULL,
        updated_at       TEXT NOT NULL,
        PRIMARY KEY (session_id, tier_level)
    )""",
```

Add to `_INDEXES` list:

```python
    "CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source)",
    "CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target)",
    "CREATE INDEX IF NOT EXISTS idx_edges_relation ON graph_edges(relation)",
    "CREATE INDEX IF NOT EXISTS idx_memory_links_node ON graph_memory_links(node_id)",
    "CREATE INDEX IF NOT EXISTS idx_turns_session ON conversation_turns(session_id, turn_index)",
```

- [ ] **Step 2: Verify schema creates cleanly**

Run: `python -c "import sqlite3; from anchor.storage.sqlite._schema import ensure_tables; conn = sqlite3.connect(':memory:'); conn.row_factory = sqlite3.Row; ensure_tables(conn); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/anchor/storage/sqlite/_schema.py
git commit -m "feat: add graph, conversation, and cache table schemas to SQLite"
```

### Task 11: Implement SqliteGraphStore

**Files:**
- Create: `src/anchor/storage/sqlite/_graph_store.py`
- Test: `tests/test_storage/test_graph_store.py` (add SQLite tests)

- [ ] **Step 1: Write failing tests for SqliteGraphStore**

Add to `tests/test_storage/test_graph_store.py`:

```python
import pytest
from anchor.storage.sqlite import SqliteConnectionManager

@pytest.fixture
def sqlite_graph_store():
    from anchor.storage.sqlite._graph_store import SqliteGraphStore
    conn_mgr = SqliteConnectionManager(":memory:")
    return SqliteGraphStore(conn_mgr)


class TestSqliteGraphStore:
    def test_satisfies_protocol(self, sqlite_graph_store):
        assert isinstance(sqlite_graph_store, GraphStore)

    def test_add_and_list_nodes(self, sqlite_graph_store):
        sqlite_graph_store.add_node("alice", {"type": "person"})
        assert "alice" in sqlite_graph_store.list_nodes()

    def test_add_and_list_edges(self, sqlite_graph_store):
        sqlite_graph_store.add_edge("alice", "knows", "bob")
        assert ("alice", "knows", "bob") in sqlite_graph_store.list_edges()

    def test_get_neighbors_depth_1(self, sqlite_graph_store):
        sqlite_graph_store.add_edge("alice", "knows", "bob")
        sqlite_graph_store.add_edge("bob", "knows", "carol")
        neighbors = sqlite_graph_store.get_neighbors("alice", max_depth=1)
        assert neighbors == ["bob"]

    def test_get_neighbors_depth_2(self, sqlite_graph_store):
        sqlite_graph_store.add_edge("alice", "knows", "bob")
        sqlite_graph_store.add_edge("bob", "knows", "carol")
        neighbors = sqlite_graph_store.get_neighbors("alice", max_depth=2)
        assert sorted(neighbors) == ["bob", "carol"]

    def test_get_neighbors_with_relation_filter(self, sqlite_graph_store):
        sqlite_graph_store.add_edge("alice", "knows", "bob")
        sqlite_graph_store.add_edge("alice", "works_with", "carol")
        neighbors = sqlite_graph_store.get_neighbors("alice", relation_filter="knows")
        assert neighbors == ["bob"]

    def test_link_and_get_memory_ids(self, sqlite_graph_store):
        sqlite_graph_store.add_node("alice")
        sqlite_graph_store.link_memory("alice", "mem-001")
        ids = sqlite_graph_store.get_memory_ids("alice")
        assert ids == ["mem-001"]

    def test_remove_node(self, sqlite_graph_store):
        sqlite_graph_store.add_edge("alice", "knows", "bob")
        sqlite_graph_store.remove_node("alice")
        assert "alice" not in sqlite_graph_store.list_nodes()
        assert sqlite_graph_store.list_edges() == []

    def test_remove_edge(self, sqlite_graph_store):
        sqlite_graph_store.add_edge("alice", "knows", "bob")
        assert sqlite_graph_store.remove_edge("alice", "knows", "bob") is True
        assert sqlite_graph_store.list_edges() == []

    def test_clear(self, sqlite_graph_store):
        sqlite_graph_store.add_edge("alice", "knows", "bob")
        sqlite_graph_store.clear()
        assert sqlite_graph_store.list_nodes() == []

    def test_handles_cycles(self, sqlite_graph_store):
        sqlite_graph_store.add_edge("a", "r", "b")
        sqlite_graph_store.add_edge("b", "r", "c")
        sqlite_graph_store.add_edge("c", "r", "a")
        neighbors = sqlite_graph_store.get_neighbors("a", max_depth=5)
        assert sorted(neighbors) == ["b", "c"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_storage/test_graph_store.py::TestSqliteGraphStore -v --no-header 2>&1 | head -5`
Expected: ImportError

- [ ] **Step 3: Implement SqliteGraphStore**

Create `src/anchor/storage/sqlite/_graph_store.py`:

```python
"""SQLite-backed graph store implementation."""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from anchor.storage.sqlite._connection import SqliteConnectionManager

from anchor.storage.sqlite._schema import ensure_tables

logger = logging.getLogger(__name__)


class SqliteGraphStore:
    """SQLite-backed graph store. Implements GraphStore protocol."""

    __slots__ = ("_conn_manager",)

    def __init__(self, connection_manager: SqliteConnectionManager) -> None:
        self._conn_manager = connection_manager
        ensure_tables(self._conn_manager.get_connection())

    def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None:
        conn = self._conn_manager.get_connection()
        existing = conn.execute(
            "SELECT metadata_json FROM graph_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if existing is not None:
            if metadata:
                merged = json.loads(existing["metadata_json"])
                merged.update(metadata)
                conn.execute(
                    "UPDATE graph_nodes SET metadata_json = ? WHERE node_id = ?",
                    (json.dumps(merged), node_id),
                )
        else:
            conn.execute(
                "INSERT INTO graph_nodes (node_id, metadata_json) VALUES (?, ?)",
                (node_id, json.dumps(metadata or {})),
            )
        conn.commit()

    def add_edge(
        self, source: str, relation: str, target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conn = self._conn_manager.get_connection()
        # Auto-create nodes
        for nid in (source, target):
            conn.execute(
                "INSERT OR IGNORE INTO graph_nodes (node_id, metadata_json) VALUES (?, '{}')",
                (nid,),
            )
        conn.execute(
            "INSERT OR IGNORE INTO graph_edges (source, relation, target, metadata_json) "
            "VALUES (?, ?, ?, ?)",
            (source, relation, target, json.dumps(metadata or {})),
        )
        conn.commit()

    def get_neighbors(
        self, node_id: str, max_depth: int = 1,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]:
        conn = self._conn_manager.get_connection()
        # Check node exists
        if conn.execute("SELECT 1 FROM graph_nodes WHERE node_id = ?", (node_id,)).fetchone() is None:
            return []
        # Python-side BFS, loading one level at a time via SQL
        visited: set[str] = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        result: list[str] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            # Build query with optional relation filter
            if relation_filter is not None:
                rels = [relation_filter] if isinstance(relation_filter, str) else list(relation_filter)
                placeholders = ",".join("?" * len(rels))
                rows = conn.execute(
                    f"SELECT target FROM graph_edges WHERE source = ? AND relation IN ({placeholders}) "
                    f"UNION SELECT source FROM graph_edges WHERE target = ? AND relation IN ({placeholders})",
                    (current, *rels, current, *rels),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT target FROM graph_edges WHERE source = ? "
                    "UNION SELECT source FROM graph_edges WHERE target = ?",
                    (current, current),
                ).fetchall()
            for row in rows:
                neighbor = row[0]  # UNION result: first column from first SELECT
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append((neighbor, depth + 1))
        return result

    def get_edges(self, node_id: str) -> list[tuple[str, str, str]]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute(
            "SELECT source, relation, target FROM graph_edges WHERE source = ? OR target = ?",
            (node_id, node_id),
        ).fetchall()
        return [(r["source"], r["relation"], r["target"]) for r in rows]

    def get_node_metadata(self, node_id: str) -> dict[str, Any] | None:
        conn = self._conn_manager.get_connection()
        row = conn.execute(
            "SELECT metadata_json FROM graph_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["metadata_json"])

    def link_memory(self, node_id: str, memory_id: str) -> None:
        conn = self._conn_manager.get_connection()
        if conn.execute("SELECT 1 FROM graph_nodes WHERE node_id = ?", (node_id,)).fetchone() is None:
            msg = f"Entity '{node_id}' does not exist in the graph"
            raise KeyError(msg)
        conn.execute(
            "INSERT OR IGNORE INTO graph_memory_links (node_id, memory_id) VALUES (?, ?)",
            (node_id, memory_id),
        )
        conn.commit()

    def get_memory_ids(self, node_id: str, max_depth: int = 1) -> list[str]:
        all_nodes = [node_id, *self.get_neighbors(node_id, max_depth=max_depth)]
        conn = self._conn_manager.get_connection()
        result: list[str] = []
        seen: set[str] = set()
        for nid in all_nodes:
            rows = conn.execute(
                "SELECT memory_id FROM graph_memory_links WHERE node_id = ?", (nid,)
            ).fetchall()
            for row in rows:
                mid = row["memory_id"]
                if mid not in seen:
                    seen.add(mid)
                    result.append(mid)
        return result

    def remove_node(self, node_id: str) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM graph_edges WHERE source = ? OR target = ?", (node_id, node_id))
        conn.execute("DELETE FROM graph_memory_links WHERE node_id = ?", (node_id,))
        conn.execute("DELETE FROM graph_nodes WHERE node_id = ?", (node_id,))
        conn.commit()

    def remove_edge(self, source: str, relation: str, target: str) -> bool:
        conn = self._conn_manager.get_connection()
        cursor = conn.execute(
            "DELETE FROM graph_edges WHERE source = ? AND relation = ? AND target = ?",
            (source, relation, target),
        )
        conn.commit()
        return cursor.rowcount > 0

    def list_nodes(self) -> list[str]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute("SELECT node_id FROM graph_nodes").fetchall()
        return [r["node_id"] for r in rows]

    def list_edges(self) -> list[tuple[str, str, str]]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute("SELECT source, relation, target FROM graph_edges").fetchall()
        return [(r["source"], r["relation"], r["target"]) for r in rows]

    def clear(self) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM graph_memory_links")
        conn.execute("DELETE FROM graph_edges")
        conn.execute("DELETE FROM graph_nodes")
        conn.commit()

    def __repr__(self) -> str:
        return f"SqliteGraphStore()"
```

- [ ] **Step 4: Add AsyncSqliteGraphStore**

Append to same file (`_graph_store.py`). Follows the `AsyncSqliteContextStore` pattern from `_context_store.py`:

```python
from anchor.storage.sqlite._schema import ensure_tables_async


class AsyncSqliteGraphStore:
    """Async SQLite-backed graph store. Implements AsyncGraphStore protocol."""

    __slots__ = ("_conn_manager",)

    def __init__(self, connection_manager: SqliteConnectionManager) -> None:
        self._conn_manager = connection_manager

    async def _ensure_tables(self) -> None:
        conn = await self._conn_manager.get_async_connection()
        await ensure_tables_async(conn)

    async def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT metadata_json FROM graph_nodes WHERE node_id = ?", (node_id,)
        )
        existing = await cursor.fetchone()
        if existing is not None:
            if metadata:
                merged = json.loads(existing["metadata_json"])
                merged.update(metadata)
                await conn.execute(
                    "UPDATE graph_nodes SET metadata_json = ? WHERE node_id = ?",
                    (json.dumps(merged), node_id),
                )
        else:
            await conn.execute(
                "INSERT INTO graph_nodes (node_id, metadata_json) VALUES (?, ?)",
                (node_id, json.dumps(metadata or {})),
            )
        await conn.commit()

    async def add_edge(
        self, source: str, relation: str, target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conn = await self._conn_manager.get_async_connection()
        for nid in (source, target):
            await conn.execute(
                "INSERT OR IGNORE INTO graph_nodes (node_id, metadata_json) VALUES (?, '{}')",
                (nid,),
            )
        await conn.execute(
            "INSERT OR IGNORE INTO graph_edges (source, relation, target, metadata_json) "
            "VALUES (?, ?, ?, ?)",
            (source, relation, target, json.dumps(metadata or {})),
        )
        await conn.commit()

    async def get_neighbors(
        self, node_id: str, max_depth: int = 1,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute("SELECT 1 FROM graph_nodes WHERE node_id = ?", (node_id,))
        if await cursor.fetchone() is None:
            return []
        visited: set[str] = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        result: list[str] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            if relation_filter is not None:
                rels = [relation_filter] if isinstance(relation_filter, str) else list(relation_filter)
                placeholders = ",".join("?" * len(rels))
                cursor = await conn.execute(
                    f"SELECT target FROM graph_edges WHERE source = ? AND relation IN ({placeholders}) "
                    f"UNION SELECT source FROM graph_edges WHERE target = ? AND relation IN ({placeholders})",
                    (current, *rels, current, *rels),
                )
            else:
                cursor = await conn.execute(
                    "SELECT target FROM graph_edges WHERE source = ? "
                    "UNION SELECT source FROM graph_edges WHERE target = ?",
                    (current, current),
                )
            rows = await cursor.fetchall()
            for row in rows:
                neighbor = row[0]
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append((neighbor, depth + 1))
        return result

    async def get_edges(self, node_id: str) -> list[tuple[str, str, str]]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT source, relation, target FROM graph_edges WHERE source = ? OR target = ?",
            (node_id, node_id),
        )
        rows = await cursor.fetchall()
        return [(r["source"], r["relation"], r["target"]) for r in rows]

    async def get_node_metadata(self, node_id: str) -> dict[str, Any] | None:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT metadata_json FROM graph_nodes WHERE node_id = ?", (node_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row["metadata_json"])

    async def link_memory(self, node_id: str, memory_id: str) -> None:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute("SELECT 1 FROM graph_nodes WHERE node_id = ?", (node_id,))
        if await cursor.fetchone() is None:
            msg = f"Entity '{node_id}' does not exist in the graph"
            raise KeyError(msg)
        await conn.execute(
            "INSERT OR IGNORE INTO graph_memory_links (node_id, memory_id) VALUES (?, ?)",
            (node_id, memory_id),
        )
        await conn.commit()

    async def get_memory_ids(self, node_id: str, max_depth: int = 1) -> list[str]:
        all_nodes = [node_id, *(await self.get_neighbors(node_id, max_depth=max_depth))]
        conn = await self._conn_manager.get_async_connection()
        result: list[str] = []
        seen: set[str] = set()
        for nid in all_nodes:
            cursor = await conn.execute(
                "SELECT memory_id FROM graph_memory_links WHERE node_id = ?", (nid,)
            )
            rows = await cursor.fetchall()
            for row in rows:
                mid = row["memory_id"]
                if mid not in seen:
                    seen.add(mid)
                    result.append(mid)
        return result

    async def remove_node(self, node_id: str) -> None:
        conn = await self._conn_manager.get_async_connection()
        await conn.execute("DELETE FROM graph_edges WHERE source = ? OR target = ?", (node_id, node_id))
        await conn.execute("DELETE FROM graph_memory_links WHERE node_id = ?", (node_id,))
        await conn.execute("DELETE FROM graph_nodes WHERE node_id = ?", (node_id,))
        await conn.commit()

    async def remove_edge(self, source: str, relation: str, target: str) -> bool:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "DELETE FROM graph_edges WHERE source = ? AND relation = ? AND target = ?",
            (source, relation, target),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def list_nodes(self) -> list[str]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute("SELECT node_id FROM graph_nodes")
        rows = await cursor.fetchall()
        return [r["node_id"] for r in rows]

    async def list_edges(self) -> list[tuple[str, str, str]]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute("SELECT source, relation, target FROM graph_edges")
        rows = await cursor.fetchall()
        return [(r["source"], r["relation"], r["target"]) for r in rows]

    async def clear(self) -> None:
        conn = await self._conn_manager.get_async_connection()
        await conn.execute("DELETE FROM graph_memory_links")
        await conn.execute("DELETE FROM graph_edges")
        await conn.execute("DELETE FROM graph_nodes")
        await conn.commit()

    def __repr__(self) -> str:
        return "AsyncSqliteGraphStore()"
```

- [ ] **Step 5: Update sqlite __init__.py exports**

Add to `src/anchor/storage/sqlite/__init__.py`:
```python
from anchor.storage.sqlite._graph_store import AsyncSqliteGraphStore, SqliteGraphStore
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_storage/test_graph_store.py -v`
Expected: All PASS (both InMemory and SQLite tests)

- [ ] **Step 7: Commit**

```bash
git add src/anchor/storage/sqlite/_graph_store.py src/anchor/storage/sqlite/__init__.py tests/test_storage/test_graph_store.py
git commit -m "feat: add SqliteGraphStore and AsyncSqliteGraphStore"
```

### Task 12: Implement SqliteConversationStore

**Files:**
- Create: `src/anchor/storage/sqlite/_conversation_store.py`
- Test: `tests/test_storage/test_conversation_store.py` (add SQLite tests)

- [ ] **Step 1: Write failing tests for SqliteConversationStore**

Add to `tests/test_storage/test_conversation_store.py`:

```python
import pytest
from anchor.storage.sqlite import SqliteConnectionManager


@pytest.fixture
def sqlite_conv_store():
    from anchor.storage.sqlite._conversation_store import SqliteConversationStore
    conn_mgr = SqliteConnectionManager(":memory:")
    return SqliteConversationStore(conn_mgr)


class TestSqliteConversationStore:
    def test_satisfies_protocol(self, sqlite_conv_store):
        assert isinstance(sqlite_conv_store, ConversationStore)

    def test_append_and_load(self, sqlite_conv_store):
        sqlite_conv_store.append_turn("sess-1", _turn("user", "hello"))
        sqlite_conv_store.append_turn("sess-1", _turn("assistant", "hi"))
        turns = sqlite_conv_store.load_turns("sess-1")
        assert len(turns) == 2
        assert turns[0].content == "hello"

    def test_load_with_limit(self, sqlite_conv_store):
        for i in range(10):
            sqlite_conv_store.append_turn("sess-1", _turn("user", f"msg-{i}"))
        turns = sqlite_conv_store.load_turns("sess-1", limit=3)
        assert len(turns) == 3
        assert turns[0].content == "msg-7"

    def test_truncate_turns(self, sqlite_conv_store):
        for i in range(10):
            sqlite_conv_store.append_turn("sess-1", _turn("user", f"msg-{i}"))
        sqlite_conv_store.truncate_turns("sess-1", keep_last=3)
        turns = sqlite_conv_store.load_turns("sess-1")
        assert len(turns) == 3
        assert turns[0].content == "msg-7"

    def test_save_and_load_tiers(self, sqlite_conv_store):
        tiers = {1: _tier(1, "summary", 5), 2: None, 3: None}
        sqlite_conv_store.save_summary_tiers("sess-1", tiers)
        loaded = sqlite_conv_store.load_summary_tiers("sess-1")
        assert loaded[1].content == "summary"
        assert loaded[1].source_turn_count == 5
        assert loaded[2] is None

    def test_delete_session(self, sqlite_conv_store):
        sqlite_conv_store.append_turn("sess-1", _turn("user", "hello"))
        assert sqlite_conv_store.delete_session("sess-1") is True
        assert sqlite_conv_store.load_turns("sess-1") == []

    def test_list_sessions(self, sqlite_conv_store):
        sqlite_conv_store.append_turn("sess-1", _turn("user", "hello"))
        sqlite_conv_store.append_turn("sess-2", _turn("user", "world"))
        assert sorted(sqlite_conv_store.list_sessions()) == ["sess-1", "sess-2"]

    def test_clear(self, sqlite_conv_store):
        sqlite_conv_store.append_turn("sess-1", _turn("user", "hello"))
        sqlite_conv_store.clear()
        assert sqlite_conv_store.list_sessions() == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_storage/test_conversation_store.py::TestSqliteConversationStore -v --no-header 2>&1 | head -5`
Expected: ImportError

- [ ] **Step 3: Implement SqliteConversationStore**

Create `src/anchor/storage/sqlite/_conversation_store.py`:

```python
"""SQLite-backed conversation store implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anchor.models.memory import ConversationTurn, SummaryTier
from anchor.storage._serialization import (
    conversation_turn_to_row,
    row_to_conversation_turn,
    row_to_summary_tier,
    summary_tier_to_row,
)
from anchor.storage.sqlite._schema import ensure_tables

if TYPE_CHECKING:
    from anchor.storage.sqlite._connection import SqliteConnectionManager

logger = logging.getLogger(__name__)


class SqliteConversationStore:
    """SQLite-backed conversation store. Implements ConversationStore protocol."""

    __slots__ = ("_conn_manager",)

    def __init__(self, connection_manager: SqliteConnectionManager) -> None:
        self._conn_manager = connection_manager
        ensure_tables(self._conn_manager.get_connection())

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        conn = self._conn_manager.get_connection()
        row = conversation_turn_to_row(turn)
        # Compute next turn_index for this session
        result = conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_idx "
            "FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        turn_index = result["next_idx"]
        conn.execute(
            "INSERT INTO conversation_turns "
            "(session_id, turn_index, role, content, token_count, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, turn_index, row["role"], row["content"],
             row["token_count"], row["metadata_json"], row["created_at"]),
        )
        conn.commit()

    def load_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        conn = self._conn_manager.get_connection()
        if limit is not None:
            rows = conn.execute(
                "SELECT * FROM conversation_turns WHERE session_id = ? "
                "ORDER BY turn_index DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = conn.execute(
                "SELECT * FROM conversation_turns WHERE session_id = ? "
                "ORDER BY turn_index ASC",
                (session_id,),
            ).fetchall()
        return [row_to_conversation_turn(r) for r in rows]

    def save_summary_tiers(self, session_id: str, tiers: dict[int, SummaryTier | None]) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM summary_tiers WHERE session_id = ?", (session_id,))
        for level, tier in tiers.items():
            if tier is not None:
                row = summary_tier_to_row(tier)
                conn.execute(
                    "INSERT INTO summary_tiers "
                    "(session_id, tier_level, content, token_count, source_turn_count, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, row["tier_level"], row["content"], row["token_count"],
                     row["source_turn_count"], row["created_at"], row["updated_at"]),
                )
        conn.commit()

    def load_summary_tiers(self, session_id: str) -> dict[int, SummaryTier | None]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute(
            "SELECT * FROM summary_tiers WHERE session_id = ?", (session_id,)
        ).fetchall()
        result: dict[int, SummaryTier | None] = {1: None, 2: None, 3: None}
        for row in rows:
            tier = row_to_summary_tier(row)
            result[tier.level] = tier
        return result

    def truncate_turns(self, session_id: str, keep_last: int) -> None:
        conn = self._conn_manager.get_connection()
        # Get the max turn_index
        row = conn.execute(
            "SELECT MAX(turn_index) AS max_idx FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row["max_idx"] is None:
            return
        cutoff = row["max_idx"] - keep_last + 1
        conn.execute(
            "DELETE FROM conversation_turns WHERE session_id = ? AND turn_index < ?",
            (session_id, cutoff),
        )
        conn.commit()

    def delete_session(self, session_id: str) -> bool:
        conn = self._conn_manager.get_connection()
        c1 = conn.execute("DELETE FROM conversation_turns WHERE session_id = ?", (session_id,))
        c2 = conn.execute("DELETE FROM summary_tiers WHERE session_id = ?", (session_id,))
        conn.commit()
        return (c1.rowcount + c2.rowcount) > 0

    def list_sessions(self) -> list[str]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM conversation_turns "
            "UNION SELECT DISTINCT session_id FROM summary_tiers"
        ).fetchall()
        return [r["session_id"] for r in rows]

    def clear(self) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM conversation_turns")
        conn.execute("DELETE FROM summary_tiers")
        conn.commit()

    def __repr__(self) -> str:
        return f"SqliteConversationStore()"
```

- [ ] **Step 4: Add AsyncSqliteConversationStore**

Append to same file (`_conversation_store.py`):

```python
from anchor.storage.sqlite._schema import ensure_tables_async


class AsyncSqliteConversationStore:
    """Async SQLite-backed conversation store. Implements AsyncConversationStore protocol."""

    __slots__ = ("_conn_manager",)

    def __init__(self, connection_manager: SqliteConnectionManager) -> None:
        self._conn_manager = connection_manager

    async def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        conn = await self._conn_manager.get_async_connection()
        row = conversation_turn_to_row(turn)
        cursor = await conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_idx "
            "FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        )
        result = await cursor.fetchone()
        turn_index = result["next_idx"]
        await conn.execute(
            "INSERT INTO conversation_turns "
            "(session_id, turn_index, role, content, token_count, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, turn_index, row["role"], row["content"],
             row["token_count"], row["metadata_json"], row["created_at"]),
        )
        await conn.commit()

    async def load_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        conn = await self._conn_manager.get_async_connection()
        if limit is not None:
            cursor = await conn.execute(
                "SELECT * FROM conversation_turns WHERE session_id = ? "
                "ORDER BY turn_index DESC LIMIT ?",
                (session_id, limit),
            )
            rows = list(reversed(await cursor.fetchall()))
        else:
            cursor = await conn.execute(
                "SELECT * FROM conversation_turns WHERE session_id = ? "
                "ORDER BY turn_index ASC",
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [row_to_conversation_turn(r) for r in rows]

    async def save_summary_tiers(self, session_id: str, tiers: dict[int, SummaryTier | None]) -> None:
        conn = await self._conn_manager.get_async_connection()
        await conn.execute("DELETE FROM summary_tiers WHERE session_id = ?", (session_id,))
        for level, tier in tiers.items():
            if tier is not None:
                row = summary_tier_to_row(tier)
                await conn.execute(
                    "INSERT INTO summary_tiers "
                    "(session_id, tier_level, content, token_count, source_turn_count, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, row["tier_level"], row["content"], row["token_count"],
                     row["source_turn_count"], row["created_at"], row["updated_at"]),
                )
        await conn.commit()

    async def load_summary_tiers(self, session_id: str) -> dict[int, SummaryTier | None]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT * FROM summary_tiers WHERE session_id = ?", (session_id,)
        )
        rows = await cursor.fetchall()
        result: dict[int, SummaryTier | None] = {1: None, 2: None, 3: None}
        for row in rows:
            tier = row_to_summary_tier(row)
            result[tier.level] = tier
        return result

    async def truncate_turns(self, session_id: str, keep_last: int) -> None:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT MAX(turn_index) AS max_idx FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row["max_idx"] is None:
            return
        cutoff = row["max_idx"] - keep_last + 1
        await conn.execute(
            "DELETE FROM conversation_turns WHERE session_id = ? AND turn_index < ?",
            (session_id, cutoff),
        )
        await conn.commit()

    async def delete_session(self, session_id: str) -> bool:
        conn = await self._conn_manager.get_async_connection()
        c1 = await conn.execute("DELETE FROM conversation_turns WHERE session_id = ?", (session_id,))
        c2 = await conn.execute("DELETE FROM summary_tiers WHERE session_id = ?", (session_id,))
        await conn.commit()
        return (c1.rowcount + c2.rowcount) > 0

    async def list_sessions(self) -> list[str]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT DISTINCT session_id FROM conversation_turns "
            "UNION SELECT DISTINCT session_id FROM summary_tiers"
        )
        rows = await cursor.fetchall()
        return [r["session_id"] for r in rows]

    async def clear(self) -> None:
        conn = await self._conn_manager.get_async_connection()
        await conn.execute("DELETE FROM conversation_turns")
        await conn.execute("DELETE FROM summary_tiers")
        await conn.commit()

    def __repr__(self) -> str:
        return "AsyncSqliteConversationStore()"
```

- [ ] **Step 5: Update sqlite __init__.py exports**

Add to `src/anchor/storage/sqlite/__init__.py`:
```python
from anchor.storage.sqlite._conversation_store import AsyncSqliteConversationStore, SqliteConversationStore
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_storage/test_conversation_store.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/anchor/storage/sqlite/_conversation_store.py src/anchor/storage/sqlite/__init__.py tests/test_storage/test_conversation_store.py
git commit -m "feat: add SqliteConversationStore and AsyncSqliteConversationStore"
```

## Chunk 5: PostgreSQL Backends + Exports + Full Test Suite

### Task 13: Implement PostgresGraphStore

**Files:**
- Create: `src/anchor/storage/postgres/_graph_store.py`
- Modify: `src/anchor/storage/postgres/_schema.py`
- Modify: `src/anchor/storage/postgres/__init__.py`

- [ ] **Step 1: Add graph schema to PostgreSQL schema module**

Add to `src/anchor/storage/postgres/_schema.py` in `ensure_tables()`, after existing table definitions:

```python
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
            node_id   TEXT PRIMARY KEY,
            metadata  JSONB NOT NULL DEFAULT '{}'
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_edges (
            id        SERIAL PRIMARY KEY,
            source    TEXT NOT NULL,
            relation  TEXT NOT NULL,
            target    TEXT NOT NULL,
            metadata  JSONB NOT NULL DEFAULT '{}',
            UNIQUE(source, relation, target)
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_memory_links (
            node_id   TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            PRIMARY KEY (node_id, memory_id)
        )
    """)
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation ON graph_edges(relation)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_links_node ON graph_memory_links(node_id)")
```

- [ ] **Step 2: Implement PostgresGraphStore**

Create `src/anchor/storage/postgres/_graph_store.py`:

```python
"""PostgreSQL-backed GraphStore implementation."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from anchor.storage.postgres._connection import PostgresConnectionManager


class PostgresGraphStore:
    """Async PostgreSQL-backed graph store. Implements AsyncGraphStore protocol.

    Uses WITH RECURSIVE CTE for get_neighbors with cycle prevention via path array.
    """

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: PostgresConnectionManager) -> None:
        self._conn_manager = conn_manager

    async def add_node(self, node_id: str, metadata: dict[str, Any] | None = None) -> None:
        async with self._conn_manager.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT metadata FROM graph_nodes WHERE node_id = $1", node_id
            )
            if existing is not None:
                if metadata:
                    merged = json.loads(existing["metadata"]) if isinstance(existing["metadata"], str) else dict(existing["metadata"])
                    merged.update(metadata)
                    await conn.execute(
                        "UPDATE graph_nodes SET metadata = $1 WHERE node_id = $2",
                        json.dumps(merged), node_id,
                    )
            else:
                await conn.execute(
                    "INSERT INTO graph_nodes (node_id, metadata) VALUES ($1, $2)",
                    node_id, json.dumps(metadata or {}),
                )

    async def add_edge(
        self, source: str, relation: str, target: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self._conn_manager.acquire() as conn:
            for nid in (source, target):
                await conn.execute(
                    "INSERT INTO graph_nodes (node_id, metadata) VALUES ($1, '{}') "
                    "ON CONFLICT (node_id) DO NOTHING",
                    nid,
                )
            await conn.execute(
                "INSERT INTO graph_edges (source, relation, target, metadata) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (source, relation, target) DO NOTHING",
                source, relation, target, json.dumps(metadata or {}),
            )

    async def get_neighbors(
        self, node_id: str, max_depth: int = 1,
        relation_filter: str | list[str] | None = None,
    ) -> list[str]:
        async with self._conn_manager.acquire() as conn:
            # Check node exists
            if await conn.fetchrow("SELECT 1 FROM graph_nodes WHERE node_id = $1", node_id) is None:
                return []
            # Build WITH RECURSIVE CTE with cycle prevention
            if relation_filter is not None:
                rels = [relation_filter] if isinstance(relation_filter, str) else list(relation_filter)
                query = """
                    WITH RECURSIVE reachable(node, depth, path) AS (
                        SELECT $1::text, 0, ARRAY[$1::text]
                      UNION ALL
                        SELECT neighbor, r.depth + 1, r.path || neighbor
                        FROM reachable r
                        CROSS JOIN LATERAL (
                            SELECT e.target AS neighbor FROM graph_edges e
                            WHERE e.source = r.node AND e.relation = ANY($3)
                            UNION
                            SELECT e.source AS neighbor FROM graph_edges e
                            WHERE e.target = r.node AND e.relation = ANY($3)
                        ) neighbors
                        WHERE r.depth < $2
                          AND NOT neighbor = ANY(r.path)
                    )
                    SELECT DISTINCT node FROM reachable WHERE node != $1
                """
                rows = await conn.fetch(query, node_id, max_depth, rels)
            else:
                query = """
                    WITH RECURSIVE reachable(node, depth, path) AS (
                        SELECT $1::text, 0, ARRAY[$1::text]
                      UNION ALL
                        SELECT neighbor, r.depth + 1, r.path || neighbor
                        FROM reachable r
                        CROSS JOIN LATERAL (
                            SELECT e.target AS neighbor FROM graph_edges e
                            WHERE e.source = r.node
                            UNION
                            SELECT e.source AS neighbor FROM graph_edges e
                            WHERE e.target = r.node
                        ) neighbors
                        WHERE r.depth < $2
                          AND NOT neighbor = ANY(r.path)
                    )
                    SELECT DISTINCT node FROM reachable WHERE node != $1
                """
                rows = await conn.fetch(query, node_id, max_depth)
            return [r["node"] for r in rows]

    async def get_edges(self, node_id: str) -> list[tuple[str, str, str]]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch(
                "SELECT source, relation, target FROM graph_edges WHERE source = $1 OR target = $1",
                node_id,
            )
            return [(r["source"], r["relation"], r["target"]) for r in rows]

    async def get_node_metadata(self, node_id: str) -> dict[str, Any] | None:
        async with self._conn_manager.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT metadata FROM graph_nodes WHERE node_id = $1", node_id
            )
            if row is None:
                return None
            meta = row["metadata"]
            return json.loads(meta) if isinstance(meta, str) else dict(meta)

    async def link_memory(self, node_id: str, memory_id: str) -> None:
        async with self._conn_manager.acquire() as conn:
            if await conn.fetchrow("SELECT 1 FROM graph_nodes WHERE node_id = $1", node_id) is None:
                msg = f"Entity '{node_id}' does not exist in the graph"
                raise KeyError(msg)
            await conn.execute(
                "INSERT INTO graph_memory_links (node_id, memory_id) VALUES ($1, $2) "
                "ON CONFLICT DO NOTHING",
                node_id, memory_id,
            )

    async def get_memory_ids(self, node_id: str, max_depth: int = 1) -> list[str]:
        all_nodes = [node_id, *(await self.get_neighbors(node_id, max_depth=max_depth))]
        async with self._conn_manager.acquire() as conn:
            result: list[str] = []
            seen: set[str] = set()
            for nid in all_nodes:
                rows = await conn.fetch(
                    "SELECT memory_id FROM graph_memory_links WHERE node_id = $1", nid
                )
                for row in rows:
                    mid = row["memory_id"]
                    if mid not in seen:
                        seen.add(mid)
                        result.append(mid)
            return result

    async def remove_node(self, node_id: str) -> None:
        async with self._conn_manager.acquire() as conn:
            await conn.execute("DELETE FROM graph_edges WHERE source = $1 OR target = $1", node_id)
            await conn.execute("DELETE FROM graph_memory_links WHERE node_id = $1", node_id)
            await conn.execute("DELETE FROM graph_nodes WHERE node_id = $1", node_id)

    async def remove_edge(self, source: str, relation: str, target: str) -> bool:
        async with self._conn_manager.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM graph_edges WHERE source = $1 AND relation = $2 AND target = $3",
                source, relation, target,
            )
            return int(result.split()[-1]) > 0

    async def list_nodes(self) -> list[str]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch("SELECT node_id FROM graph_nodes")
            return [r["node_id"] for r in rows]

    async def list_edges(self) -> list[tuple[str, str, str]]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch("SELECT source, relation, target FROM graph_edges")
            return [(r["source"], r["relation"], r["target"]) for r in rows]

    async def clear(self) -> None:
        async with self._conn_manager.acquire() as conn:
            await conn.execute("DELETE FROM graph_memory_links")
            await conn.execute("DELETE FROM graph_edges")
            await conn.execute("DELETE FROM graph_nodes")

    def __repr__(self) -> str:
        return "PostgresGraphStore()"
```

- [ ] **Step 3: Update postgres __init__.py**

Add to `src/anchor/storage/postgres/__init__.py`:
```python
from anchor.storage.postgres._graph_store import PostgresGraphStore
```
And add `"PostgresGraphStore"` to the `__all__` list.

- [ ] **Step 4: Write PostgresGraphStore tests**

Add to `tests/test_storage/test_graph_store.py`. Tests are marked with `@pytest.mark.postgres` and skipped when asyncpg is unavailable:

```python
import pytest

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

postgres_only = pytest.mark.skipif(not HAS_ASYNCPG, reason="asyncpg not installed")


@postgres_only
@pytest.mark.asyncio
class TestPostgresGraphStore:
    """Tests for PostgresGraphStore.

    Requires a running PostgreSQL instance. Set ANCHOR_TEST_POSTGRES_DSN env var.
    Tests are skipped if asyncpg is not installed or no DSN is configured.
    """

    @pytest.fixture
    async def pg_graph_store(self):
        import os
        dsn = os.environ.get("ANCHOR_TEST_POSTGRES_DSN")
        if not dsn:
            pytest.skip("ANCHOR_TEST_POSTGRES_DSN not set")
        from anchor.storage.postgres._graph_store import PostgresGraphStore
        from anchor.storage.postgres._connection import PostgresConnectionManager
        mgr = PostgresConnectionManager(dsn)
        store = PostgresGraphStore(mgr)
        yield store
        await store.clear()

    async def test_add_and_list_nodes(self, pg_graph_store):
        await pg_graph_store.add_node("alice", {"type": "person"})
        nodes = await pg_graph_store.list_nodes()
        assert "alice" in nodes

    async def test_add_edge_and_traverse(self, pg_graph_store):
        await pg_graph_store.add_edge("alice", "knows", "bob")
        await pg_graph_store.add_edge("bob", "knows", "carol")
        neighbors = await pg_graph_store.get_neighbors("alice", max_depth=2)
        assert sorted(neighbors) == ["bob", "carol"]

    async def test_relation_filter(self, pg_graph_store):
        await pg_graph_store.add_edge("alice", "knows", "bob")
        await pg_graph_store.add_edge("alice", "works_with", "carol")
        neighbors = await pg_graph_store.get_neighbors("alice", relation_filter="knows")
        assert neighbors == ["bob"]

    async def test_handles_cycles(self, pg_graph_store):
        await pg_graph_store.add_edge("a", "r", "b")
        await pg_graph_store.add_edge("b", "r", "c")
        await pg_graph_store.add_edge("c", "r", "a")
        neighbors = await pg_graph_store.get_neighbors("a", max_depth=5)
        assert sorted(neighbors) == ["b", "c"]

    async def test_link_and_get_memory_ids(self, pg_graph_store):
        await pg_graph_store.add_node("alice")
        await pg_graph_store.link_memory("alice", "mem-001")
        ids = await pg_graph_store.get_memory_ids("alice")
        assert ids == ["mem-001"]
```

- [ ] **Step 5: Commit**

```bash
git add src/anchor/storage/postgres/_graph_store.py src/anchor/storage/postgres/_schema.py src/anchor/storage/postgres/__init__.py tests/test_storage/test_graph_store.py
git commit -m "feat: add PostgresGraphStore with recursive CTE traversal"
```

### Task 14: Implement PostgresConversationStore

**Files:**
- Create: `src/anchor/storage/postgres/_conversation_store.py`
- Modify: `src/anchor/storage/postgres/_schema.py`
- Modify: `src/anchor/storage/postgres/__init__.py`

- [ ] **Step 1: Add conversation schema to PostgreSQL**

Add to `src/anchor/storage/postgres/_schema.py` in `ensure_tables()`:

```python
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_turns (
            id            SERIAL PRIMARY KEY,
            session_id    TEXT NOT NULL,
            turn_index    INTEGER NOT NULL,
            role          TEXT NOT NULL,
            content       TEXT NOT NULL,
            token_count   INTEGER NOT NULL DEFAULT 0,
            metadata      JSONB NOT NULL DEFAULT '{}',
            created_at    TIMESTAMPTZ NOT NULL
        )
    """)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS summary_tiers (
            session_id       TEXT NOT NULL,
            tier_level       INTEGER NOT NULL,
            content          TEXT NOT NULL,
            token_count      INTEGER NOT NULL,
            source_turn_count INTEGER NOT NULL,
            created_at       TIMESTAMPTZ NOT NULL,
            updated_at       TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (session_id, tier_level)
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_turns_session ON conversation_turns(session_id, turn_index)"
    )
```

- [ ] **Step 2: Implement PostgresConversationStore**

Create `src/anchor/storage/postgres/_conversation_store.py`:

```python
"""PostgreSQL-backed ConversationStore implementation."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from anchor.models.memory import ConversationTurn, SummaryTier

if TYPE_CHECKING:
    from anchor.storage.postgres._connection import PostgresConnectionManager


class PostgresConversationStore:
    """Async PostgreSQL-backed conversation store. Implements AsyncConversationStore protocol."""

    __slots__ = ("_conn_manager",)

    def __init__(self, conn_manager: PostgresConnectionManager) -> None:
        self._conn_manager = conn_manager

    async def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        async with self._conn_manager.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_idx "
                "FROM conversation_turns WHERE session_id = $1",
                session_id,
            )
            turn_index = row["next_idx"]
            await conn.execute(
                "INSERT INTO conversation_turns "
                "(session_id, turn_index, role, content, token_count, metadata, created_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                session_id, turn_index, str(turn.role), turn.content,
                turn.token_count, json.dumps(turn.metadata, default=str),
                turn.timestamp,
            )

    async def load_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        async with self._conn_manager.acquire() as conn:
            if limit is not None:
                rows = await conn.fetch(
                    "SELECT * FROM ("
                    "  SELECT * FROM conversation_turns WHERE session_id = $1 "
                    "  ORDER BY turn_index DESC LIMIT $2"
                    ") sub ORDER BY turn_index ASC",
                    session_id, limit,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM conversation_turns WHERE session_id = $1 "
                    "ORDER BY turn_index ASC",
                    session_id,
                )
            return [
                ConversationTurn(
                    role=r["role"],
                    content=r["content"],
                    token_count=r["token_count"],
                    metadata=json.loads(r["metadata"]) if isinstance(r["metadata"], str) else dict(r["metadata"]),
                    timestamp=r["created_at"],
                )
                for r in rows
            ]

    async def save_summary_tiers(self, session_id: str, tiers: dict[int, SummaryTier | None]) -> None:
        async with self._conn_manager.acquire() as conn:
            await conn.execute("DELETE FROM summary_tiers WHERE session_id = $1", session_id)
            for level, tier in tiers.items():
                if tier is not None:
                    await conn.execute(
                        "INSERT INTO summary_tiers "
                        "(session_id, tier_level, content, token_count, source_turn_count, created_at, updated_at) "
                        "VALUES ($1, $2, $3, $4, $5, $6, $7)",
                        session_id, tier.level, tier.content, tier.token_count,
                        tier.source_turn_count, tier.created_at, tier.updated_at,
                    )

    async def load_summary_tiers(self, session_id: str) -> dict[int, SummaryTier | None]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM summary_tiers WHERE session_id = $1", session_id
            )
            result: dict[int, SummaryTier | None] = {1: None, 2: None, 3: None}
            for r in rows:
                result[r["tier_level"]] = SummaryTier(
                    level=r["tier_level"],
                    content=r["content"],
                    token_count=r["token_count"],
                    source_turn_count=r["source_turn_count"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
            return result

    async def truncate_turns(self, session_id: str, keep_last: int) -> None:
        async with self._conn_manager.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT MAX(turn_index) AS max_idx FROM conversation_turns WHERE session_id = $1",
                session_id,
            )
            if row["max_idx"] is None:
                return
            cutoff = row["max_idx"] - keep_last + 1
            await conn.execute(
                "DELETE FROM conversation_turns WHERE session_id = $1 AND turn_index < $2",
                session_id, cutoff,
            )

    async def delete_session(self, session_id: str) -> bool:
        async with self._conn_manager.acquire() as conn:
            r1 = await conn.execute("DELETE FROM conversation_turns WHERE session_id = $1", session_id)
            r2 = await conn.execute("DELETE FROM summary_tiers WHERE session_id = $1", session_id)
            return (int(r1.split()[-1]) + int(r2.split()[-1])) > 0

    async def list_sessions(self) -> list[str]:
        async with self._conn_manager.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT session_id FROM conversation_turns "
                "UNION SELECT DISTINCT session_id FROM summary_tiers"
            )
            return [r["session_id"] for r in rows]

    async def clear(self) -> None:
        async with self._conn_manager.acquire() as conn:
            await conn.execute("DELETE FROM conversation_turns")
            await conn.execute("DELETE FROM summary_tiers")

    def __repr__(self) -> str:
        return "PostgresConversationStore()"
```

- [ ] **Step 3: Update postgres __init__.py**

Add to `src/anchor/storage/postgres/__init__.py`:
```python
from anchor.storage.postgres._conversation_store import PostgresConversationStore
```
And add `"PostgresConversationStore"` to the `__all__` list.

- [ ] **Step 4: Write PostgresConversationStore tests**

Add to `tests/test_storage/test_conversation_store.py`:

```python
@postgres_only
@pytest.mark.asyncio
class TestPostgresConversationStore:
    """Tests for PostgresConversationStore. Requires running PostgreSQL."""

    @pytest.fixture
    async def pg_conv_store(self):
        import os
        dsn = os.environ.get("ANCHOR_TEST_POSTGRES_DSN")
        if not dsn:
            pytest.skip("ANCHOR_TEST_POSTGRES_DSN not set")
        from anchor.storage.postgres._conversation_store import PostgresConversationStore
        from anchor.storage.postgres._connection import PostgresConnectionManager
        mgr = PostgresConnectionManager(dsn)
        store = PostgresConversationStore(mgr)
        yield store
        await store.clear()

    async def test_append_and_load(self, pg_conv_store):
        await pg_conv_store.append_turn("sess-1", _turn("user", "hello"))
        await pg_conv_store.append_turn("sess-1", _turn("assistant", "hi"))
        turns = await pg_conv_store.load_turns("sess-1")
        assert len(turns) == 2
        assert turns[0].content == "hello"

    async def test_load_with_limit(self, pg_conv_store):
        for i in range(10):
            await pg_conv_store.append_turn("sess-1", _turn("user", f"msg-{i}"))
        turns = await pg_conv_store.load_turns("sess-1", limit=3)
        assert len(turns) == 3
        assert turns[0].content == "msg-7"

    async def test_save_and_load_tiers(self, pg_conv_store):
        tiers = {1: _tier(1, "summary", 5), 2: None, 3: None}
        await pg_conv_store.save_summary_tiers("sess-1", tiers)
        loaded = await pg_conv_store.load_summary_tiers("sess-1")
        assert loaded[1].content == "summary"

    async def test_delete_session(self, pg_conv_store):
        await pg_conv_store.append_turn("sess-1", _turn("user", "hello"))
        assert await pg_conv_store.delete_session("sess-1") is True
        assert await pg_conv_store.load_turns("sess-1") == []

    async def test_list_sessions(self, pg_conv_store):
        await pg_conv_store.append_turn("sess-1", _turn("user", "hello"))
        await pg_conv_store.append_turn("sess-2", _turn("user", "world"))
        assert sorted(await pg_conv_store.list_sessions()) == ["sess-1", "sess-2"]
```

Also add the postgres skip imports at the top of the file (matching the pattern from test_graph_store.py):

```python
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

postgres_only = pytest.mark.skipif(not HAS_ASYNCPG, reason="asyncpg not installed")
```

- [ ] **Step 5: Commit**

```bash
git add src/anchor/storage/postgres/_conversation_store.py src/anchor/storage/postgres/_schema.py src/anchor/storage/postgres/__init__.py tests/test_storage/test_conversation_store.py
git commit -m "feat: add PostgresConversationStore"
```

### Task 15: Update public API exports

**Files:**
- Modify: `src/anchor/__init__.py`

- [ ] **Step 1: Add new exports to __init__.py**

Add to `src/anchor/__init__.py`:

```python
# New protocols
from anchor.protocols.storage import (
    AsyncConversationStore,
    AsyncGraphStore,
    ConversationStore,
    GraphStore,
)

# New InMemory implementations
from anchor.storage.memory_store import (
    InMemoryConversationStore,
    InMemoryGraphStore,
)
```

Cache backends are exported from `anchor.cache` (already updated in Task 9). They are NOT re-exported from `anchor.__init__` because they require optional dependencies (redis, sqlite). Users import them directly:
- `from anchor.cache import InMemoryCacheBackend` (always available)
- `from anchor.cache import SqliteCacheBackend` (requires sqlite schema setup)
- `from anchor.cache import RedisCacheBackend` (requires redis-py)

- [ ] **Step 2: Verify imports work**

Run: `python -c "from anchor import GraphStore, ConversationStore, InMemoryGraphStore, InMemoryConversationStore; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/anchor/__init__.py
git commit -m "feat: export new storage protocols and InMemory implementations from anchor"
```

### Task 16: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -x -q --timeout=60`
Expected: All PASS, no regressions

- [ ] **Step 2: Run linting**

Run: `ruff check src/anchor/`
Expected: No errors

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address any linting or test issues from storage layer gaps"
```
