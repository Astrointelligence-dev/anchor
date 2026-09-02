#!/usr/bin/env python3
"""Knowledge graph: entities, evidenced relations and scope-aware navigation.

Run with:  python examples/graph_memory.py

Demonstrates KnowledgeGraph over memory entries — no API keys, no external
services: build the graph, walk it, explain a path, hide a namespace with a
RetrievalScope, and feed memory into a pipeline through graph_retrieval_step.
"""

from __future__ import annotations

from anchor import (
    ContextPipeline,
    InMemoryEntryStore,
    KnowledgeGraph,
    MemoryEntry,
    MemoryManager,
    RetrievalScope,
    graph_retrieval_step,
)


class WhitespaceTokenizer:
    """Minimal tokenizer that counts whitespace-separated words."""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


def header(number: int, title: str) -> None:
    print(f"\n=== {number}. {title} ===\n")


def demo_build() -> tuple[KnowledgeGraph, InMemoryEntryStore]:
    header(1, "Build the graph from memory entries")
    graph = KnowledgeGraph()
    store = InMemoryEntryStore()
    facts = {
        "mem-python": ("/lang", "Python is a high-level language known for readability."),
        "mem-fastapi": ("/web", "FastAPI is one of the fastest Python web frameworks."),
        "mem-pydantic": ("/web", "Pydantic validates data with Python type annotations."),
        "mem-sqlalchemy": ("/data", "SQLAlchemy is the Python SQL toolkit and ORM."),
        "mem-postgres": ("/data", "PostgreSQL scales with read replicas and partitioning."),
    }
    for entry_id, (_, content) in facts.items():
        store.add(MemoryEntry(id=entry_id, content=content))

    # Nodes first, each evidenced by the memory entry that talks about it
    # (the item id IS the memory id — one currency), under a namespace.
    for name, entry_id in (
        ("Python", "mem-python"),
        ("FastAPI", "mem-fastapi"),
        ("Pydantic", "mem-pydantic"),
        ("SQLAlchemy", "mem-sqlalchemy"),
        ("PostgreSQL", "mem-postgres"),
    ):
        graph.add_node(name)
        graph.link_item(name, entry_id, facts[entry_id][0])

    # Typed, evidenced relations with the fact they came from.
    for source, relation, target, evidence, fact in (
        ("FastAPI", "uses", "Pydantic", "mem-fastapi", "FastAPI validates with Pydantic"),
        ("FastAPI", "written in", "Python", "mem-fastapi", "FastAPI is a Python framework"),
        ("Pydantic", "written in", "Python", "mem-pydantic", None),
        ("SQLAlchemy", "written in", "Python", "mem-sqlalchemy", None),
        ("SQLAlchemy", "connects to", "PostgreSQL", "mem-sqlalchemy", "the ORM talks to Postgres"),
    ):
        edge = graph.add_edge(source, relation, target, evidence=[evidence], fact=fact)
        print(f"  {edge.source} --[{edge.relation}]--> {edge.target}  (evidence {edge.evidence})")
    print(f"\n  nodes: {graph.nodes()}")
    return graph, store


def demo_navigate(graph: KnowledgeGraph) -> None:
    header(2, "Navigate: neighbors, path, explain, hubs")
    print(f"  neighbors(FastAPI, depth 1): {graph.neighbors('FastAPI')}")
    print(f"  neighbors(FastAPI, depth 2): {graph.neighbors('FastAPI', max_depth=2)}")
    print(f"  path(FastAPI -> PostgreSQL): {graph.path('FastAPI', 'PostgreSQL')}")
    for hop in graph.explain("FastAPI", "PostgreSQL"):
        print(f"    {hop.source} --[{hop.relation}]--> {hop.target}: {hop.fact or '(no fact)'}")
    print(f"  hubs: {graph.hubs(k=3)}")
    print(f"  items around Python: {graph.related_items('Python', max_depth=1)}")


def demo_scope(graph: KnowledgeGraph) -> None:
    header(3, "Scope: a hidden namespace is a wall, not a bridge")
    no_data = RetrievalScope(exclude=("/data",))
    print(f"  nodes(exclude /data): {graph.nodes(scope=no_data)}")
    print(
        f"  path(FastAPI -> PostgreSQL, exclude /data): {graph.path('FastAPI', 'PostgreSQL', scope=no_data)}"
    )
    print(f"  query(seeds=[Python]) top 3: {[i for i, _ in graph.query(['Python'], top_k=3)]}")
    print(
        f"  ...under exclude /data: {[i for i, _ in graph.query(['Python'], top_k=3, scope=no_data)]}"
    )


def demo_pipeline(graph: KnowledgeGraph, store: InMemoryEntryStore) -> None:
    header(4, "graph_retrieval_step in a ContextPipeline")
    keywords = {
        "fastapi": "FastAPI",
        "postgres": "PostgreSQL",
        "python": "Python",
        "orm": "SQLAlchemy",
    }

    def extract_entities(query: str) -> list[str]:
        q = query.lower()
        return [name for word, name in keywords.items() if word in q]

    tokenizer = WhitespaceTokenizer()
    manager = MemoryManager(conversation_tokens=200, tokenizer=tokenizer)
    manager.add_user_message("I want to learn about FastAPI and databases.")
    pipeline = (
        ContextPipeline(max_tokens=500, tokenizer=tokenizer)
        .with_memory(manager)
        .add_system_prompt("You are a Python web development expert.")
        .add_step(graph_retrieval_step(graph, store, extract_entities, max_depth=2, max_items=5))
    )
    for query in ("Tell me about FastAPI performance", "How does Postgres scale?"):
        result = pipeline.build(query)
        print(
            f"  '{query}' -> {len(result.window.items)} items, {result.window.used_tokens} tokens"
        )
        for item in result.window.items:
            tag = (
                "[GRAPH]"
                if item.metadata.get("source") == "graph_retrieval"
                else f"[{item.source.value}]"
            )
            print(f"    {tag:<14} {item.content[:60]}")


def main() -> None:
    graph, store = demo_build()
    demo_navigate(graph)
    demo_scope(graph)
    demo_pipeline(graph, store)


if __name__ == "__main__":
    main()
