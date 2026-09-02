"""Knowledge-graph models (roadmap #4).

Nodes are entities or notes under a canonical key; edges carry a free,
normalized relation label, provenance, confidence, the evidence
(``ContextItem`` ids) that supports them, and bi-temporal validity.

Design, fixed in docs/plans/2026-08-28-v020-4-grafo-de-conhecimento.md:

- ``ContextItem`` is the single currency: evidence is a tuple of item ids,
  never memory ids. For memory, the item id *is* ``MemoryEntry.id``.
- ``relation`` is a free label normalized by :func:`normalize_key`
  (``"Works On"`` → ``"works_on"``), no closed enum. Built-in extractors
  use ``links_to`` (wikilink), ``contains`` (structure) and ``similar_to``
  (embedding); an LLM extractor emits its own predicates.
- An obsolete edge is invalidated, never deleted: ``invalidated_at`` is
  transaction time, ``valid_from``/``valid_to`` is world time.
- Visibility under a scope is decided by evidence: a node without a
  visible evidence item does not exist for that scope, and an edge whose
  evidence is all hidden is hidden too — exclusion filters the node, and
  an excluded node is a wall, not a bridge.
"""

from __future__ import annotations

import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Provenance = Literal["extracted", "inferred", "ambiguous"]
"""Where an edge came from: the source says it, the system inferred it, or
the extractor could not resolve it (graphify's three tags)."""

_PROVENANCE_RANK: dict[str, int] = {"extracted": 2, "inferred": 1, "ambiguous": 0}


def normalize_key(text: str) -> str:
    """Canonical key for node ids and relation labels.

    NFKC + casefold + whitespace collapsed to ``_``: ``"Project X"``,
    ``"project x"`` and ``"PROJECT  X"`` converge on ``"project_x"``, so a
    wikilink target and an LLM-extracted entity land on the same node.
    Raises ``ValueError`` on empty input.
    """
    key = "_".join(unicodedata.normalize("NFKC", text).casefold().split())
    if not key:
        msg = "graph key must not be empty"
        raise ValueError(msg)
    return key


def best_provenance(a: Provenance, b: Provenance) -> Provenance:
    """The stronger of two provenance tags (extracted > inferred > ambiguous)."""
    return a if _PROVENANCE_RANK[a] >= _PROVENANCE_RANK[b] else b


class GraphNode(BaseModel):
    """An entity or note. ``id`` is the canonical key, ``label`` the display name."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str = ""
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _default_label(cls, data: Any) -> Any:
        # The label defaults to the name as given (before normalization),
        # so "Project X" stays readable while its id becomes "project_x".
        if isinstance(data, dict) and not data.get("label") and isinstance(data.get("id"), str):
            data = {**data, "label": data["id"].strip()}
        return data

    @field_validator("id")
    @classmethod
    def _normalize_id(cls, v: str) -> str:
        return normalize_key(v)


class GraphEdge(BaseModel):
    """A directed, labeled, evidenced, bi-temporal relation between two nodes."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    target: str
    relation: str
    fact: str | None = None
    provenance: Provenance = "extracted"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    invalidated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source", "target", "relation")
    @classmethod
    def _normalize(cls, v: str) -> str:
        return normalize_key(v)

    @field_validator("evidence")
    @classmethod
    def _dedupe(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(v))

    @property
    def weight(self) -> int:
        """How much support the edge has — the evidence count (min 1)."""
        return max(1, len(self.evidence))

    def is_live(self, as_of: datetime | None = None) -> bool:
        """Not invalidated, and valid in world time at *as_of* (default: now)."""
        if self.invalidated_at is not None:
            return False
        at = as_of if as_of is not None else datetime.now(UTC)
        if self.valid_from is not None and self.valid_from > at:
            return False
        return not (self.valid_to is not None and self.valid_to <= at)


@dataclass(frozen=True, slots=True)
class Subgraph:
    """The part of a graph visible under a scope at a time — what algorithms consume."""

    nodes: dict[str, GraphNode]
    edges: list[GraphEdge]
    items: dict[str, tuple[str, ...]]
    """Node id → visible evidence item ids."""
