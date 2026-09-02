"""Front #3: recall benchmark — scope pre-filter vs post-filter reference.

The risk the 2026-09-01 research flagged: a selective filter applied
AFTER the KNN (Pinecone-style post-filter, or an ``OR`` pushed into the
vec0 KNN ``WHERE``) silently loses recall. This pins the number on every
vector backend: with a scope keeping 5% of the corpus, scoped search must
keep recall@10 ≈ 1.0 while the post-filter reference collapses. Run with
``-s`` to see the table; the plan doc records it.
"""

from __future__ import annotations

import math
import random

from anchor.evaluation import GoldenCase, evaluate_retriever
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle
from anchor.models.scope import RetrievalScope
from tests.test_storage.conftest import DIM

N_ITEMS = 500
N_NAMESPACES = 20
N_QUERIES = 30
K = 10

SCOPES = {
    "include 1/20 (5%)": RetrievalScope(include=("/ns-3",)),
    "include 4/20 (20%)": RetrievalScope(
        include=("/ns-3", "/ns-7", "/ns-11", "/ns-19"),
    ),
    "exclude 1/20 (95%)": RetrievalScope(exclude=("/ns-3",)),
}


def _unit(rng: random.Random) -> list[float]:
    # Gaussian, not conftest.make_embedding: that family spans a 2-D
    # plane (sin(φ+i) = sinφ·cos i + cosφ·sin i), so neighbours are
    # near-ties on a circle. Random directions give a real top-k.
    v = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class _StoreRetriever:
    """Golden-set ``retrieve`` over a vector store.

    ``post_filter=True`` is the reference the research warns about: an
    unscoped KNN whose hits are filtered afterwards.
    """

    def __init__(self, store, queries, namespaces, scope, *, post_filter):
        self._store = store
        self._queries = queries
        self._ns = namespaces
        self._scope = scope
        self._post_filter = post_filter

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        emb = self._queries[query.query_str]
        if self._post_filter:
            hits = self._store.search(emb, top_k=top_k)
            hits = [(i, s) for i, s in hits if self._scope.matches(self._ns[i])]
        else:
            hits = self._store.search(emb, top_k=top_k, scope=self._scope)
        return [
            ContextItem(
                id=i,
                content=i,
                source=SourceType.RETRIEVAL,
                score=min(1.0, max(0.0, s)),
            )
            for i, s in hits
        ]


def test_scoped_search_keeps_recall_under_selective_filters(make_vector_store):
    rng = random.Random(3)  # noqa: S311 -- deterministic corpus, not crypto
    items = {f"d{i}": _unit(rng) for i in range(N_ITEMS)}
    namespaces = {i: f"/ns-{n % N_NAMESPACES}" for n, i in enumerate(items)}
    queries = {f"q{j}": _unit(rng) for j in range(N_QUERIES)}

    store = make_vector_store()
    for item_id, emb in items.items():
        store.add_embedding(item_id, emb, namespace=namespaces[item_id])

    print()
    for label, scope in SCOPES.items():
        cases = []
        for q, qemb in queries.items():
            in_scope = [i for i in items if scope.matches(namespaces[i])]
            ranked = sorted(in_scope, key=lambda i: -_dot(qemb, items[i]))
            cases.append(GoldenCase(query=q, relevant=ranked[:K]))

        pre = evaluate_retriever(
            _StoreRetriever(store, queries, namespaces, scope, post_filter=False),
            cases,
            k=K,
        ).mean("recall_at_k")
        post = evaluate_retriever(
            _StoreRetriever(store, queries, namespaces, scope, post_filter=True),
            cases,
            k=K,
        ).mean("recall_at_k")
        print(f"  {label:<20} recall@{K}  pre-filter={pre:.3f}  post-filter={post:.3f}")

        assert pre >= 0.99, f"{label}: scoped search lost recall ({pre:.3f})"
        if "include" in label:
            assert post <= 0.6, f"{label}: post-filter reference unexpectedly fine ({post:.3f})"
