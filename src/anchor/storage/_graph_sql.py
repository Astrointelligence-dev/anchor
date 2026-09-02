"""What every GraphStore shares: row shapes, merge rules and the visibility filter.

The rules live once, in Python, and every backend (in-memory included)
applies them to rows it fetched with cheap indexed queries. The visibility
rule itself is stated in :mod:`anchor.models.graph`.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping, Sequence
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
    """Upsert semantics: aliases and metadata merge, the first REAL label wins.

    A node born as an edge endpoint carries its key as a placeholder label;
    the first upsert that brings a display name replaces it.
    """
    label = current.label
    if label == current.id and incoming.label != incoming.id:
        label = incoming.label
    return current.model_copy(
        update={
            "label": label,
            "aliases": tuple(dict.fromkeys(current.aliases + incoming.aliases)),
            "metadata": {**current.metadata, **incoming.metadata},
        }
    )


def check_new_edge(edge: GraphEdge) -> None:
    """Edges are born live: closing one is ``invalidate_edge``, never ``add_edge``."""
    if edge.invalidated_at is not None:
        msg = f"edge {edge.id!r} carries invalidated_at; use invalidate_edge to close a live edge"
        raise ValueError(msg)


def merge_edge(current: GraphEdge, incoming: GraphEdge) -> GraphEdge:
    """Reinforce a live edge: evidence union, max confidence, best provenance, first fact.

    World time follows one rule for both bounds — the incoming value when it
    brings one, else the current (re-asserting a fact with ``valid_to`` is how
    the bi-temporal model ends it; a later ``valid_from`` is a correction).
    """
    return current.model_copy(
        update={
            "evidence": tuple(dict.fromkeys(current.evidence + incoming.evidence)),
            "confidence": max(current.confidence, incoming.confidence),
            "provenance": best_provenance(current.provenance, incoming.provenance),
            "fact": current.fact if current.fact is not None else incoming.fact,
            "valid_from": incoming.valid_from
            if incoming.valid_from is not None
            else current.valid_from,
            "valid_to": incoming.valid_to if incoming.valid_to is not None else current.valid_to,
            "metadata": {**current.metadata, **incoming.metadata},
        }
    )


def node_visible(
    node_id: str,
    node_items: Mapping[str, Collection[str]],
    visible: set[str] | None,
    root_visible: bool,
) -> bool:
    """The rule from :mod:`anchor.models.graph`: one visible item, or no items and root visible."""
    if visible is None:
        return True
    items = node_items.get(node_id, ())
    if not items:
        return root_visible
    return any(i in visible for i in items)


def visible_nodes(
    node_ids: Iterable[str],
    node_items: Mapping[str, Collection[str]],
    visible: set[str] | None,
    root_visible: bool = True,
) -> list[str]:
    """Node ids that exist for the scope (``visible`` is ``None`` when unscoped)."""
    return [n for n in node_ids if node_visible(n, node_items, visible, root_visible)]


def visible_edges(
    edges: Iterable[GraphEdge],
    node_items: Mapping[str, Collection[str]],
    visible: set[str] | None,
    as_of: datetime | None,
    root_visible: bool = True,
) -> list[GraphEdge]:
    """Edges live at *as_of* whose endpoints and evidence survive the scope."""
    out: list[GraphEdge] = []
    for edge in edges:
        if not edge.is_live(as_of):
            continue
        if visible is not None:
            ends = (edge.source, edge.target)
            if not all(node_visible(n, node_items, visible, root_visible) for n in ends):
                continue
            if edge.evidence and not any(i in visible for i in edge.evidence):
                continue
        out.append(edge)
    return out
