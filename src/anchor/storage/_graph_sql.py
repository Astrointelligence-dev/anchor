"""What the SQL-backed GraphStores share: row shapes and the visibility filter.

The rules live once, in Python, and every backend applies them to rows it
fetched with cheap indexed queries — the same semantics as
``InMemoryGraphStore``, which is the reference:

- an item is visible when ``scope`` is ``None`` or matches its namespace;
- a node is visible when ``scope`` is ``None`` or one of its items is;
- an edge is visible when live at ``as_of``, both endpoints visible and,
  if it carries evidence, at least one evidence item is visible.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from anchor.models.graph import GraphEdge, GraphNode, best_provenance

EDGE_COLUMNS = (
    "id",
    "source",
    "target",
    "relation",
    "fact",
    "provenance",
    "confidence",
    "valid_from",
    "valid_to",
    "created_at",
    "invalidated_at",
    "metadata_json",
)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _dt(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def node_to_row(node: GraphNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "label": node.label,
        "aliases_json": json.dumps(list(node.aliases)),
        "metadata_json": json.dumps(node.metadata, default=str),
    }


def row_to_node(row: Any) -> GraphNode:
    return GraphNode(
        id=row["id"],
        label=row["label"],
        aliases=tuple(_loads(row["aliases_json"], [])),
        metadata=_loads(row["metadata_json"], {}),
    )


def edge_to_row(edge: GraphEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "source": edge.source,
        "target": edge.target,
        "relation": edge.relation,
        "fact": edge.fact,
        "provenance": edge.provenance,
        "confidence": edge.confidence,
        "valid_from": _iso(edge.valid_from),
        "valid_to": _iso(edge.valid_to),
        "created_at": _iso(edge.created_at),
        "invalidated_at": _iso(edge.invalidated_at),
        "metadata_json": json.dumps(edge.metadata, default=str),
    }


def row_to_edge(row: Any, evidence: Sequence[str] = ()) -> GraphEdge:
    return GraphEdge(
        id=row["id"],
        source=row["source"],
        target=row["target"],
        relation=row["relation"],
        fact=row["fact"],
        provenance=row["provenance"],
        confidence=row["confidence"],
        evidence=tuple(evidence),
        valid_from=_dt(row["valid_from"]),
        valid_to=_dt(row["valid_to"]),
        created_at=_dt(row["created_at"]) or datetime.now().astimezone(),
        invalidated_at=_dt(row["invalidated_at"]),
        metadata=_loads(row["metadata_json"], {}),
    )


def _loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def merge_node(current: GraphNode, incoming: GraphNode) -> GraphNode:
    """Upsert semantics: aliases and metadata merge, the first label wins."""
    return current.model_copy(
        update={
            "aliases": tuple(dict.fromkeys(current.aliases + incoming.aliases)),
            "metadata": {**current.metadata, **incoming.metadata},
        }
    )


def merge_edge(current: GraphEdge, incoming: GraphEdge) -> GraphEdge:
    """Reinforce a live edge: evidence union, max confidence, best provenance, first fact."""
    return current.model_copy(
        update={
            "evidence": tuple(dict.fromkeys(current.evidence + incoming.evidence)),
            "confidence": max(current.confidence, incoming.confidence),
            "provenance": best_provenance(current.provenance, incoming.provenance),
            "fact": current.fact if current.fact is not None else incoming.fact,
            "metadata": {**current.metadata, **incoming.metadata},
        }
    )


def visible_nodes(
    node_ids: Iterable[str],
    node_items: Mapping[str, Sequence[str]],
    visible: set[str] | None,
) -> list[str]:
    """Node ids that exist for the scope (``visible`` is ``None`` when unscoped)."""
    if visible is None:
        return list(node_ids)
    return [n for n in node_ids if any(i in visible for i in node_items.get(n, ()))]


def visible_edges(
    edges: Iterable[GraphEdge],
    node_items: Mapping[str, Sequence[str]],
    visible: set[str] | None,
    as_of: datetime | None,
) -> list[GraphEdge]:
    """Edges live at *as_of* whose endpoints and evidence survive the scope."""
    out: list[GraphEdge] = []
    for edge in edges:
        if not edge.is_live(as_of):
            continue
        if visible is not None:
            ends = (edge.source, edge.target)
            if not all(any(i in visible for i in node_items.get(n, ())) for n in ends):
                continue
            if edge.evidence and not any(i in visible for i in edge.evidence):
                continue
        out.append(edge)
    return out
