"""Deterministic graph extraction from ``ContextItem``s (roadmap #4, phase B).

Two extractors that cost nothing — no LLM, no embedding — and the indexer
that writes their output into a ``KnowledgeGraph``:

- :class:`StructureExtractor`: the note a chunk belongs to becomes a node
  (Obsidian identity = the file name), evidenced by the chunk; frontmatter
  ``aliases`` become node aliases and ``tags`` become ``tagged`` edges.
- :class:`WikilinkExtractor`: every ``[[target]]`` in the chunk becomes a
  ``links_to`` edge from the chunk's note to the target note, evidenced by
  the chunk, and the target node is linked to the chunk as a mention.

An LLM extractor (phase D) plugs into the same :class:`GraphExtractor`
protocol.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

from anchor.graph.knowledge_graph import KnowledgeGraph
from anchor.models.context import ContextItem
from anchor.models.graph import GraphEdge, GraphNode, normalize_key

logger = logging.getLogger(__name__)

WIKILINK_RE = re.compile(r"(!?)\[\[([^\[\]]+?)\]\]")
"""``[[target]]``, ``[[target|alias]]``, ``[[target#heading]]``, ``[[target#^block]]``,
``![[embed]]`` — the Obsidian grammar."""


@dataclass(frozen=True, slots=True)
class Wikilink:
    """One parsed ``[[...]]``. ``target`` is the note name (basename, no extension)."""

    target: str
    anchor: str | None = None
    alias: str | None = None
    embed: bool = False


def parse_wikilink(inner: str, *, embed: bool = False) -> Wikilink | None:
    """Parse the text between ``[[`` and ``]]``; ``None`` for a same-note link (``[[#h]]``)."""
    body, _, alias = inner.partition("|")
    path, _, anchor = body.partition("#")
    name = PurePosixPath(path.strip()).name
    if name.lower().endswith(".md"):
        name = name[:-3]
    if not name:
        return None
    return Wikilink(
        target=name,
        anchor=anchor.strip() or None,
        alias=alias.strip() or None,
        embed=embed,
    )


def wikilinks(text: str) -> list[Wikilink]:
    """Every wikilink in *text*, in order."""
    out: list[Wikilink] = []
    for embed, inner in WIKILINK_RE.findall(text):
        link = parse_wikilink(inner, embed=bool(embed))
        if link is not None:
            out.append(link)
    return out


def note_name(item: ContextItem) -> str | None:
    """The note a chunk belongs to — its file stem (Obsidian identity), else its title."""
    filename = item.metadata.get("doc_filename")
    if isinstance(filename, str) and filename:
        return PurePosixPath(filename).stem
    title = item.metadata.get("doc_title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def _as_list(value: Any) -> list[str]:
    """Frontmatter ``tags``/``aliases`` come as a list or a comma/space string."""
    if isinstance(value, str):
        return [v.strip().lstrip("#") for v in re.split(r"[,\s]+", value) if v.strip()]
    if isinstance(value, list | tuple):
        return [str(v).strip().lstrip("#") for v in value if str(v).strip()]
    return []


@dataclass(slots=True)
class Extraction:
    """What one extractor found in one item.

    ``nodes`` are linked to the item as evidence; ``edges`` get the item as
    evidence when they carry none.
    """

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


@runtime_checkable
class GraphExtractor(Protocol):
    """Anything that turns a ``ContextItem`` into nodes and edges."""

    def extract(self, item: ContextItem) -> Extraction: ...


class StructureExtractor:
    """The chunk's note as a node (+ frontmatter aliases and ``tagged`` edges)."""

    __slots__ = ()

    def extract(self, item: ContextItem) -> Extraction:
        name = note_name(item)
        if name is None:
            return Extraction()
        aliases = _as_list(item.metadata.get("doc_aliases"))
        note = GraphNode(
            id=name,
            label=name,
            aliases=tuple(aliases),
            metadata={"kind": "note", "doc_id": item.metadata.get("parent_doc_id")},
        )
        result = Extraction(nodes=[note])
        for tag in _as_list(item.metadata.get("doc_tags")):
            result.nodes.append(GraphNode(id=tag, label=tag, metadata={"kind": "tag"}))
            result.edges.append(
                GraphEdge(
                    source=name,
                    target=tag,
                    relation="tagged",
                    metadata={"extractor": "structure"},
                )
            )
        return result

    def __repr__(self) -> str:
        return "StructureExtractor()"


class WikilinkExtractor:
    """``[[target]]`` → ``links_to`` edge from the chunk's note; the target is a mention.

    Resolution is by basename, case-insensitive, like Obsidian's default.
    """

    # ponytail: two notes with the same basename in different folders merge
    # into one node; add a vault-wide note index when a real vault has them.
    __slots__ = ()

    def extract(self, item: ContextItem) -> Extraction:
        source = note_name(item)
        result = Extraction()
        seen: set[tuple[str, str]] = set()
        for link in wikilinks(item.content):
            target_key = normalize_key(link.target)
            if source is not None and normalize_key(source) == target_key:
                continue
            result.nodes.append(
                GraphNode(
                    id=link.target,
                    label=link.target,
                    aliases=(link.alias,) if link.alias else (),
                    metadata={"kind": "note"},
                )
            )
            if source is None:
                continue
            key = (normalize_key(source), target_key)
            if key in seen:
                continue
            seen.add(key)
            meta: dict[str, Any] = {"extractor": "wikilink"}
            if link.anchor:
                meta["anchor"] = link.anchor
            if link.embed:
                meta["embed"] = True
            result.edges.append(
                GraphEdge(source=source, target=link.target, relation="links_to", metadata=meta)
            )
        return result

    def __repr__(self) -> str:
        return "WikilinkExtractor()"


@dataclass(frozen=True, slots=True)
class IndexStats:
    items: int
    nodes: int
    edges: int


class GraphIndexer:
    """Run extractors over items and write the result into a ``KnowledgeGraph``.

    Add-only and idempotent: indexing the same item twice reinforces the
    same nodes and edges (evidence merges, nothing duplicates). Removing an
    item is ``graph.unlink_item(item_id)`` — edges left without evidence are
    invalidated, which is the incremental contract of roadmap #4.
    """

    __slots__ = ("_extractors", "_graph")

    def __init__(
        self,
        graph: KnowledgeGraph,
        extractors: Sequence[GraphExtractor] | None = None,
    ) -> None:
        self._graph = graph
        self._extractors: tuple[GraphExtractor, ...] = (
            tuple(extractors)
            if extractors is not None
            else (StructureExtractor(), WikilinkExtractor())
        )

    @property
    def extractors(self) -> tuple[GraphExtractor, ...]:
        return self._extractors

    def index(self, items: Iterable[ContextItem]) -> IndexStats:
        """Extract and write every item. Items from another vault are refused."""
        store = self._graph.store
        n_items = n_nodes = n_edges = 0
        for item in items:
            if item.vault != store.vault:
                msg = (
                    f"item {item.id!r} lives in vault {item.vault!r}, "
                    f"graph is mounted on {store.vault!r}"
                )
                raise ValueError(msg)
            n_items += 1
            for extractor in self._extractors:
                found = extractor.extract(item)
                for node in found.nodes:
                    store.upsert_node(node)
                    store.link_item(node.id, item.id, item.namespace)
                    n_nodes += 1
                for edge in found.edges:
                    if not edge.evidence:
                        edge = edge.model_copy(update={"evidence": (item.id,)})
                    store.add_edge(edge)
                    n_edges += 1
        return IndexStats(items=n_items, nodes=n_nodes, edges=n_edges)

    def __repr__(self) -> str:
        return f"GraphIndexer(extractors={list(self._extractors)!r})"
