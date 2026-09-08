"""Graph retrieval: personalized PageRank over the knowledge graph (roadmap #4)."""

from __future__ import annotations

import logging
from collections.abc import Callable

from anchor.exceptions import RetrieverError
from anchor.graph.knowledge_graph import KnowledgeGraph
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.models.scope import RetrievalScope, same_vault, scope_kwargs
from anchor.protocols.embeddings import EmbeddingProvider
from anchor.protocols.storage import ContextStore, VectorStore
from anchor.protocols.tokenizer import Tokenizer
from anchor.tokens.counter import get_default_counter

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieves items by spreading activation from the nodes a query mentions.

    Seeds are the graph nodes named in the query (``entity_extractor``, or
    the graph's own alias-aware mention matcher) plus, when a vector store
    is given, the ``seed_k`` nearest items (HippoRAG-2 style passage
    seeding). Personalized PageRank over the entity-item bipartite graph
    scores every visible item; the ranking is resolved through the context
    store and normalized to ``[0, 1]`` by the top score.

    First-class ``Retriever``: it plugs into ``retriever_step`` (scope
    intersection), ``HybridRetriever`` (RRF fusion by item id — which is
    why items keep their canonical ids) and the evaluation harness.
    """

    __slots__ = (
        "_context_store",
        "_damping",
        "_embeddings",
        "_entity_extractor",
        "_graph",
        "_seed_k",
        "_tokenizer",
        "_vector_store",
    )

    def __init__(
        self,
        graph: KnowledgeGraph,
        context_store: ContextStore,
        entity_extractor: Callable[[str], list[str]] | None = None,
        *,
        vector_store: VectorStore | None = None,
        embeddings: EmbeddingProvider | None = None,
        seed_k: int = 5,
        damping: float = 0.5,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        same_vault(graph.store, context_store, vector_store)
        self._graph = graph
        self._context_store = context_store
        self._entity_extractor = entity_extractor
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._seed_k = seed_k
        self._damping = damping
        self._tokenizer = tokenizer or get_default_counter()

    def __repr__(self) -> str:
        return (
            f"GraphRetriever(graph={self._graph!r}, dense_seeding={self._vector_store is not None})"
        )

    def _item_seeds(self, query: QueryBundle, scope: RetrievalScope | None) -> list[str]:
        if self._vector_store is None:
            return []
        if query.embedding is not None:
            embedding = query.embedding
        elif self._embeddings is not None:
            embedding = self._embeddings.embed_query(query.query_str)
        else:
            msg = "GraphRetriever has a vector_store but no embeddings and query.embedding is None"
            raise RetrieverError(msg)
        hits = self._vector_store.search(embedding, top_k=self._seed_k, **scope_kwargs(scope))
        return [item_id for item_id, _ in hits]

    def retrieve(
        self,
        query: QueryBundle,
        top_k: int = 10,
        *,
        scope: RetrievalScope | None = None,
    ) -> list[ContextItem]:
        """Items ranked by activation from the query's seeds, within *scope*."""
        # One scoped load serves both the mention matcher and the walk.
        sub = self._graph.store.subgraph(scope=scope)
        if self._entity_extractor is not None:
            names = self._entity_extractor(query.query_str)
        else:
            names = self._graph.mentions(query.query_str, scope=scope, subgraph=sub)
        ranked = self._graph.query(
            names,
            item_seeds=self._item_seeds(query, scope),
            top_k=top_k,
            scope=scope,
            damping=self._damping,
            subgraph=sub,
        )
        # Resolve first, normalize after: an id the context store does not
        # hold (a memory entry indexed via index_entries, say) must not be the
        # score everything else is divided by.
        resolved = [
            (item, score)
            for item_id, score in ranked
            if (item := self._context_store.get(item_id)) is not None
        ]
        if not resolved:
            if ranked:
                logger.debug("graph ranked %d items, none resolve", len(ranked))
            return []
        top = resolved[0][1]
        items: list[ContextItem] = []
        for item, score in resolved:
            items.append(
                item.model_copy(
                    update={
                        "source": SourceType.RETRIEVAL,
                        "score": score / top,
                        "token_count": item.token_count
                        or self._tokenizer.count_tokens(item.content),
                        "metadata": {
                            **item.metadata,
                            "retrieval_method": "graph",
                            "raw_score": score,
                            "graph_seeds": list(names),
                        },
                    }
                )
            )
        return items
