"""Deterministic graph extraction from ``ContextItem``s (roadmap #4, phase B).

Two extractors that cost nothing — no LLM, no embedding — and the indexer
that writes their output into a ``KnowledgeGraph``:

- :class:`StructureExtractor`: the note a chunk belongs to becomes a node
  (Obsidian identity = the file name), evidenced by the chunk; frontmatter
  ``aliases`` become node aliases and ``tags`` become ``tagged`` edges.
- :class:`WikilinkExtractor`: every ``[[target]]`` in the chunk becomes a
  ``links_to`` edge from the chunk's note to the target note, evidenced by
  the chunk, and the target node is linked to the chunk as a mention.

:class:`LLMGraphExtractor` is the opt-in third source (phase D): one
model call per chunk, entities and typed relations with fact, provenance
and confidence — for text without links, memory above all.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from anchor._text import ask_json
from anchor.graph.knowledge_graph import KnowledgeGraph
from anchor.models.context import ContextItem, SourceType
from anchor.models.graph import GraphEdge, GraphNode, normalize_key
from anchor.models.memory import MemoryEntry
from anchor.models.scope import ROOT_NAMESPACE
from anchor.protocols.storage import MemoryEntryStore

if TYPE_CHECKING:
    from anchor.llm.base import LLMProvider

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
        raw: list[str] = re.split(r"[,\s]+", value)
    elif isinstance(value, list | tuple):
        raw = [str(v) for v in value]
    else:
        return []
    cleaned = (v.strip().lstrip("#").strip() for v in raw)
    return [v for v in cleaned if v]


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


DEFAULT_RELATIONS = (
    "depends_on",
    "part_of",
    "uses",
    "owns",
    "works_on",
    "member_of",
    "located_in",
    "caused_by",
    "decided_by",
    "related_to",
)
"""Suggested predicates: the prompt lists them to contain label drift; nothing validates them."""

_EXTRACTION_PROMPT = """Extract the knowledge graph from the text below.

Return ONLY a JSON object with two keys:
- "entities": a list of {{"name": str, "type": str, "aliases": [str]}} — the concrete
  things the text is about (people, systems, places, concepts). Use the most
  specific, consistent name; put other spellings in "aliases".
- "relations": a list of {{"source": str, "target": str, "relation": str, "fact": str,
  "provenance": "extracted" | "inferred", "confidence": number between 0 and 1}}.

Rules:
- "relation" is a short snake_case predicate. Prefer one of: {relations}. Invent a
  new one only when none fits.
- "provenance" is "extracted" when the text states the relation, "inferred" when
  you deduce it. "fact" quotes the evidence in the text's own words.
- Never invent entities the text does not mention. Return
  {{"entities": [], "relations": []}} when there is nothing.

TEXT:
{content}"""

_PROVENANCE: frozenset[str] = frozenset({"extracted", "inferred", "ambiguous"})


class LLMGraphExtractor:
    """Entities and typed relations from a chunk, in one model call (no gleaning).

    Opt-in: the corpus A/B (plan doc, phase B) showed the wikilink graph does
    not beat hybrid retrieval, so this is for text without links — memory
    entries first. The provider is injected (the agent's own), never a new
    client; a malformed or failed response yields nothing and logs a
    warning, the same fail-soft contract as ``TierCompactor``.
    """

    __slots__ = ("_llm", "_relations")

    def __init__(self, llm: LLMProvider, *, relations: Iterable[str] = DEFAULT_RELATIONS) -> None:
        self._llm = llm
        self._relations = tuple(relations)

    def extract(self, item: ContextItem) -> Extraction:
        prompt = _EXTRACTION_PROMPT.format(
            relations=", ".join(self._relations), content=item.content
        )
        data = ask_json(
            self._llm,
            prompt,
            expect=dict,
            log=logger,
            what=f"LLM graph extraction failed for item {item.id}",
        )
        if data is None:
            return Extraction()
        try:
            return self._parse(data)
        except Exception as exc:
            # Fail-soft (the TierCompactor contract): a malformed answer costs
            # this chunk its graph, never the index run. The model can return
            # any shape (`"entities": true`), so parsing is guarded too.
            logger.warning("LLM graph extraction failed for item %s: %s", item.id, exc)
            return Extraction()

    def _parse(self, data: dict[str, Any]) -> Extraction:
        result = Extraction()
        nodes: dict[str, GraphNode] = {}
        for ent in data.get("entities") or []:
            if isinstance(ent, dict):
                _add_node(nodes, ent.get("name"), ent.get("type"), ent.get("aliases") or ())
        seen: set[tuple[str, str, str]] = set()
        for rel in data.get("relations") or []:
            edge = _parse_relation(rel, nodes)
            if edge is None:
                continue
            key = (edge.source, edge.relation, edge.target)
            if key not in seen:
                seen.add(key)
                result.edges.append(edge)
        result.nodes = list(nodes.values())
        return result

    def __repr__(self) -> str:
        return f"LLMGraphExtractor(llm={self._llm!r})"


def _add_node(
    nodes: dict[str, GraphNode], name: Any, kind: Any = None, aliases: Any = ()
) -> str | None:
    """Register an entity by canonical key (first spelling wins). Returns the key."""
    if not isinstance(name, str) or not name.strip():
        return None
    key = normalize_key(name)
    if key not in nodes:
        alias_list = aliases if isinstance(aliases, list | tuple) else ()
        meta: dict[str, Any] = {"kind": "entity", "extractor": "llm"}
        if kind is not None:
            meta["type"] = kind
        nodes[key] = GraphNode(
            id=name,
            label=name.strip(),
            aliases=tuple(a for a in alias_list if isinstance(a, str) and a.strip()),
            metadata=meta,
        )
    return key


def _parse_relation(rel: Any, nodes: dict[str, GraphNode]) -> GraphEdge | None:
    """One relation dict → edge (endpoints registered as nodes), or ``None`` if unusable."""
    if not isinstance(rel, dict) or not isinstance(rel.get("relation"), str):
        return None
    src, tgt = rel.get("source"), rel.get("target")
    if not (isinstance(src, str) and isinstance(tgt, str)):
        return None
    try:
        relation = normalize_key(rel["relation"])
        if normalize_key(src) == normalize_key(tgt):
            return None
    except ValueError:
        return None
    source, target = _add_node(nodes, src), _add_node(nodes, tgt)
    if source is None or target is None:
        return None
    provenance = rel.get("provenance")
    confidence = rel.get("confidence", 0.9)
    if provenance not in _PROVENANCE:
        provenance = "ambiguous"
    if not isinstance(confidence, int | float):
        confidence = 0.9
    confidence = max(0.0, min(1.0, float(confidence)))
    if provenance == "ambiguous":
        confidence = min(confidence, 0.3)
    fact = rel.get("fact")
    return GraphEdge(
        source=source,
        target=target,
        relation=relation,
        fact=fact if isinstance(fact, str) and fact.strip() else None,
        provenance=provenance,  # type: ignore[arg-type]
        confidence=confidence,
        metadata={"extractor": "llm"},
    )


def entry_to_item(entry: MemoryEntry, *, vault: str) -> ContextItem:
    """A memory entry as the item the graph evidences — same id, root namespace."""
    return ContextItem(
        id=entry.id,
        content=entry.content,
        source=SourceType.MEMORY,
        score=entry.relevance_score,
        metadata={
            "memory_id": entry.id,
            "memory_type": str(entry.memory_type),
            "tags": list(entry.tags),
        },
        created_at=entry.created_at,
        vault=vault,
        namespace=ROOT_NAMESPACE,
    )


class GraphIndexingEntryStore:
    """A ``MemoryEntryStore`` that keeps a knowledge graph in step with itself.

    Wrap the persistent store once and every writer — ``MemoryManager``,
    the pipeline's consolidation step, ``MemoryGarbageCollector`` — keeps
    the graph honest without knowing it exists: ``add`` re-extracts the
    entry (unless its content is unchanged), while ``add`` of an expired
    entry (a soft delete), ``delete`` and ``clear`` take the evidence with
    them. Everything else forwards to the wrapped store.
    """

    __slots__ = ("_indexer", "_inner")

    def __init__(self, inner: MemoryEntryStore, indexer: GraphIndexer) -> None:
        self._inner = inner
        self._indexer = indexer

    @property
    def inner(self) -> MemoryEntryStore:
        return self._inner

    @property
    def graph(self) -> KnowledgeGraph:
        return self._indexer.graph

    def add(self, entry: MemoryEntry) -> None:
        if entry.is_expired:
            # A soft delete (or an entry arriving already expired): hidden from
            # list_all, so its evidence leaves the graph — same as delete.
            self._inner.add(entry)
            self._indexer.graph.unlink_item(entry.id)
            return
        current = getattr(self._inner, "get", lambda _id: None)(entry.id)
        self._inner.add(entry)
        if current is not None and not current.is_expired and current.content == entry.content:
            return  # same live text: the graph already has this evidence, no re-extraction
        self._indexer.graph.unlink_item(entry.id)  # a restored (un-expired) entry re-links here
        self._indexer.index_entries([entry])

    def delete(self, entry_id: str) -> bool:
        deleted = self._inner.delete(entry_id)
        if deleted:
            self._indexer.graph.unlink_item(entry_id)
        return deleted

    def clear(self) -> None:
        # Expired entries are hidden from list_all but still evidence the graph.
        everything = getattr(self._inner, "list_all_unfiltered", self._inner.list_all)()
        for entry in everything:
            self._indexer.graph.unlink_item(entry.id)
        self._inner.clear()

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        return self._inner.search(query, top_k=top_k)

    def list_all(self) -> list[MemoryEntry]:
        return self._inner.list_all()

    def __getattr__(self, name: str) -> Any:  # get, list_all_unfiltered, search_filtered, ...
        if name.startswith("_"):  # copy/pickle probe attributes before __init__ ran
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"GraphIndexingEntryStore({self._inner!r})"


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Items indexed, distinct nodes touched, distinct edges touched."""

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
    def graph(self) -> KnowledgeGraph:
        return self._graph

    @property
    def extractors(self) -> tuple[GraphExtractor, ...]:
        return self._extractors

    def index_entries(self, entries: Iterable[MemoryEntry]) -> IndexStats:
        """Index memory entries: the item id IS the entry id (one currency)."""
        vault = self._graph.store.vault
        return self.index(entry_to_item(e, vault=vault) for e in entries)

    def index(self, items: Iterable[ContextItem]) -> IndexStats:
        """Extract and write every item. Items from another vault are refused.

        Every node an extractor returns, and both endpoints of every edge it
        returns, are linked to the item as evidence: the chunk mentions them.
        ``IndexStats`` counts distinct nodes and edges touched, not emissions.
        """
        store = self._graph.store
        n_items = 0
        nodes_seen: set[str] = set()
        edges_seen: set[tuple[str, str, str]] = set()
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
                    nodes_seen.add(node.id)
                for edge in found.edges:
                    for endpoint in (edge.source, edge.target):
                        if endpoint not in nodes_seen and store.get_node(endpoint) is None:
                            store.upsert_node(GraphNode(id=endpoint))
                        store.link_item(endpoint, item.id, item.namespace)
                        nodes_seen.add(endpoint)
                    if not edge.evidence:
                        edge = edge.model_copy(update={"evidence": (item.id,)})
                    store.add_edge(edge)
                    edges_seen.add((edge.source, edge.relation, edge.target))
        return IndexStats(items=n_items, nodes=len(nodes_seen), edges=len(edges_seen))

    def __repr__(self) -> str:
        return f"GraphIndexer(extractors={list(self._extractors)!r})"
