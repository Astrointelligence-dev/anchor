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
  transaction time, ``valid_from``/``valid_to`` is world time. Every
  datetime is timezone-aware — a naive one is rejected at construction, never
  on the read after the write.
- **Visibility under a scope is decided by evidence** (the one rule, stated
  here and referenced everywhere): an item is visible when the scope matches
  its namespace; a node is visible when one of its evidence items is, and a
  node with no evidence at all lives at the root namespace ``/`` (visible
  under an empty include, hidden by ``exclude=("/",)`` or a narrower
  include — the same rule memory follows); an edge is visible when it is live
  at ``as_of``, both endpoints are visible and, if it carries evidence, one
  evidence item is visible. Exclusion filters the node, and an excluded node
  is a wall, not a bridge. An edge's evidence evidences both its endpoints
  (the item mentions them), so an empty ``RetrievalScope()`` is the identity.
"""

from __future__ import annotations

import re
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
_SEPARATORS = re.compile(r"[\s\-_]+")


def normalize_key(text: str) -> str:
    """Canonical key for node ids and relation labels.

    NFKC + casefold, then every run of whitespace, hyphens and underscores
    becomes one ``_`` (leading/trailing runs dropped): ``"Project X"``,
    ``"project-x"`` and ``"PROJECT_X"`` converge on ``"project_x"``, so a
    wikilink target, a file stem and an LLM-extracted entity land on the
    same node. Other punctuation is kept — ``C++``, ``C#`` and ``.NET`` stay
    distinct. Raises ``ValueError`` on empty input.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    key = _SEPARATORS.sub("_", folded).strip("_")
    if not key:
        msg = "graph key must not be empty"
        raise ValueError(msg)
    return key


def best_provenance(a: Provenance, b: Provenance) -> Provenance:
    """The stronger of two provenance tags (extracted > inferred > ambiguous)."""
    return a if _PROVENANCE_RANK[a] >= _PROVENANCE_RANK[b] else b


def require_aware(value: datetime | None, name: str = "datetime") -> datetime | None:
    """Refuse a naive datetime: world/transaction time must carry a timezone."""
    if value is not None and value.tzinfo is None:
        msg = f"{name} must be timezone-aware (got naive {value.isoformat()})"
        raise ValueError(msg)
    return value


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

    @field_validator("valid_from", "valid_to", "created_at", "invalidated_at")
    @classmethod
    def _aware(cls, v: datetime | None, info: Any) -> datetime | None:
        return require_aware(v, info.field_name)

    def is_live(self, as_of: datetime | None = None) -> bool:
        """Not invalidated, and valid in world time at *as_of* (default: now)."""
        if self.invalidated_at is not None:
            return False
        at = require_aware(as_of, "as_of") if as_of is not None else datetime.now(UTC)
        assert at is not None  # noqa: S101 -- narrowed above
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
