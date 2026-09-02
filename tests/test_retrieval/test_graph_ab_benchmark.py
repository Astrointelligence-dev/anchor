"""Roadmap #4 gate: graph retrieval vs hybrid+RRF on a hand-made corpus.

Three conditions over ``tests/fixtures/graph_corpus`` (40 wiki notes with
wikilinks, 30 golden queries in three strata): hybrid (BM25 + a hashed
bag-of-words "dense" — no model in CI), graph alone (mention seeds + dense
passage seeds → PPR), and RRF(hybrid, graph). Run with ``-s`` to see the
table; the plan doc records it. The plan's rule: invest in the LLM
extractor only if the graph moves multi-hop/explain, not the average.
"""

from __future__ import annotations

import math
import re
import zlib
from pathlib import Path

import pytest

from anchor.embeddings._base import as_embedding_provider
from anchor.evaluation import GoldenCase, evaluate_retriever, load_golden_set
from anchor.graph import KnowledgeGraph
from anchor.ingestion import DocumentIngester, GraphIndexer
from anchor.models.context import ContextItem
from anchor.models.scope import RetrievalScope
from anchor.retrieval import DenseRetriever, GraphRetriever, HybridRetriever, SparseRetriever
from anchor.storage.memory_store import InMemoryContextStore, InMemoryVectorStore

CORPUS = Path(__file__).resolve().parent.parent / "fixtures" / "graph_corpus"
K = 5
DIM = 256
STRATA = ("fact", "multi-hop", "explain")


def _bow(text: str) -> list[float]:
    """Hashed bag-of-words: a deterministic stand-in for an embedding model."""
    v = [0.0] * DIM
    for tok in re.findall(r"\w+", text.casefold()):
        v[zlib.crc32(tok.encode()) % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _load_items() -> list[ContextItem]:
    pytest.importorskip("rank_bm25")
    ingester = DocumentIngester()
    items: list[ContextItem] = []
    for folder in sorted(p for p in CORPUS.iterdir() if p.is_dir()):
        for item in ingester.ingest_directory(folder, extensions=[".md"]):
            items.append(item.model_copy(update={"namespace": f"/{folder.name}"}))
    return items


def _cases(items: list[ContextItem]) -> list[GoldenCase]:
    by_stem: dict[str, list[str]] = {}
    for item in items:
        by_stem.setdefault(Path(item.metadata["doc_filename"]).stem, []).append(item.id)
    cases = []
    for case in load_golden_set(CORPUS / "golden.jsonl"):
        assert isinstance(case.relevant, list)
        ids = [cid for stem in case.relevant for cid in by_stem[stem]]
        cases.append(case.model_copy(update={"relevant": ids}))
    return cases


@pytest.fixture(scope="module")
def world() -> dict[str, object]:
    items = _load_items()
    context_store = InMemoryContextStore()
    vectors = InMemoryVectorStore()
    dense = DenseRetriever(vectors, context_store, embed_fn=_bow)
    dense.index(items)
    sparse = SparseRetriever()
    sparse.index(items)
    graph = KnowledgeGraph()
    GraphIndexer(graph).index(items)
    hybrid = HybridRetriever([sparse, dense])
    graph_ret = GraphRetriever(
        graph, context_store, vector_store=vectors, embeddings=as_embedding_provider(_bow), seed_k=3
    )
    return {
        "items": items,
        "cases": _cases(items),
        "graph": graph,
        "hybrid": hybrid,
        "graph_ret": graph_ret,
        "fusion": HybridRetriever([hybrid, graph_ret]),
        "fusion2:1": HybridRetriever([hybrid, graph_ret], weights=[1.0, 0.5]),
    }


def _by_stratum(cases: list[GoldenCase]) -> dict[str, list[GoldenCase]]:
    return {s: [c for c in cases if c.name.startswith(s + ":")] for s in STRATA}


def test_graph_vs_hybrid_ab(world: dict[str, object]) -> None:
    cases = world["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 30
    conditions = {
        "hybrid": world["hybrid"],
        "graph": world["graph_ret"],
        "fusion": world["fusion"],
        "fusion2:1": world["fusion2:1"],
    }
    table: dict[str, dict[str, tuple[float, float]]] = {}
    for label, retriever in conditions.items():
        table[label] = {}
        for stratum, subset in {**_by_stratum(cases), "all": cases}.items():
            report = evaluate_retriever(retriever, subset, k=K)  # type: ignore[arg-type]
            table[label][stratum] = (report.mean("recall_at_k"), report.mean("mrr"))
    print()
    print(f"  {'condition':<10}" + "".join(f"{s:>22}" for s in (*STRATA, "all")))
    for label, row in table.items():
        print(f"  {label:<10}" + "".join(f"  recall={r:.2f} mrr={m:.2f}" for r, m in row.values()))

    hybrid, graph, fusion = table["hybrid"], table["graph"], table["fusion"]
    # Session 11 numbers (recall@5 / mrr): hybrid 0.94/0.85, graph 0.85/0.66,
    # fusion 0.94/0.73 — on this corpus hybrid+RRF is at the ceiling and the
    # wikilink graph does NOT beat it: fusion matches recall and costs MRR.
    # These pins are regression floors, not a victory claim; the gate result
    # is recorded in the plan doc (LLM extractor stays opt-in).
    assert fusion["all"][0] >= hybrid["all"][0]
    assert fusion["multi-hop"][0] >= hybrid["multi-hop"][0]
    assert fusion["all"][1] >= 0.65
    assert graph["multi-hop"][0] >= 0.75
    assert graph["all"][0] >= 0.8


def test_graph_retrieval_never_leaks_an_excluded_folder(world: dict[str, object]) -> None:
    cases = world["cases"]
    graph = world["graph"]
    assert isinstance(cases, list)
    assert isinstance(graph, KnowledgeGraph)
    no_incidents = RetrievalScope(exclude=("/incidents",))
    incident_ids = {i.id for i in world["items"] if i.namespace == "/incidents"}  # type: ignore[attr-defined]
    for retriever in (world["graph_ret"], world["fusion"]):
        report = evaluate_retriever(retriever, cases, k=10, scope=no_incidents)  # type: ignore[arg-type]
        for result in report.results:
            assert not incident_ids & set(result.retrieved_ids), result.case.name
    # Incident notes are still NAMED by public notes (a wikilink is public
    # knowledge), so the nodes survive — but none of their hidden chunks do.
    for node in graph.nodes(scope=no_incidents):
        assert not incident_ids & set(graph.items(node, scope=no_incidents)), node
    assert graph.path("payments-service", "bruno-costa", scope=no_incidents) is not None
