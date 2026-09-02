"""LLMGraphExtractor with a canned provider: parsing, normalization, fail-soft."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from anchor.graph import KnowledgeGraph
from anchor.ingestion import GraphIndexer, LLMGraphExtractor
from anchor.ingestion.graph_extractors import _EXTRACTION_PROMPT, DEFAULT_RELATIONS
from anchor.models.context import ContextItem, SourceType
from anchor.models.memory import MemoryEntry

PAYLOAD = {
    "entities": [
        {"name": "Ana Lima", "type": "person", "aliases": ["Ana"]},
        {"name": "Team Payments", "type": "team", "aliases": []},
        {"name": "billing service", "type": "service"},
    ],
    "relations": [
        {
            "source": "Ana Lima",
            "target": "Team Payments",
            "relation": "member of",
            "fact": "Ana Lima leads Team Payments",
            "provenance": "extracted",
            "confidence": 1.0,
        },
        {
            "source": "Ana Lima",
            "target": "Billing Service",
            "relation": "on_call_for",
            "fact": "on call for Billing",
            "provenance": "inferred",
            "confidence": 0.8,
        },
        {"source": "Ana Lima", "target": "Team Payments", "relation": "member_of"},  # duplicate
        {"source": "Ledger", "target": "Ledger", "relation": "uses"},  # self-loop
        {"source": "", "target": "x", "relation": "uses"},  # empty endpoint
        {
            "source": "Ana Lima",
            "target": "Mystery",
            "relation": "??",
            "provenance": "guess",
            "confidence": 5,
        },
    ],
}


class FakeLLM:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, messages, **kwargs):
        self.prompts.append(messages[0].content)
        return SimpleNamespace(content=self.content)


def _item(text: str = "Ana Lima leads Team Payments and is on call for Billing.") -> ContextItem:
    return ContextItem(id="m1", content=text, source=SourceType.MEMORY)


class TestLLMGraphExtractor:
    def test_parses_entities_and_relations(self) -> None:
        llm = FakeLLM("```json\n" + json.dumps(PAYLOAD) + "\n```")
        found = LLMGraphExtractor(llm).extract(_item())
        assert [n.id for n in found.nodes] == [
            "ana_lima",
            "team_payments",
            "billing_service",
            "mystery",
        ]
        ana = found.nodes[0]
        assert ana.label == "Ana Lima"
        assert ana.aliases == ("Ana",)
        assert ana.metadata == {"kind": "entity", "extractor": "llm", "type": "person"}
        edges = [(e.source, e.relation, e.target, e.provenance, e.confidence) for e in found.edges]
        assert edges == [
            ("ana_lima", "member_of", "team_payments", "extracted", 1.0),
            ("ana_lima", "on_call_for", "billing_service", "inferred", 0.8),
            ("ana_lima", "??", "mystery", "ambiguous", 0.3),
        ]
        assert found.edges[0].fact == "Ana Lima leads Team Payments"
        assert found.edges[0].metadata == {"extractor": "llm"}

    def test_prompt_carries_text_and_vocabulary(self) -> None:
        llm = FakeLLM('{"entities": [], "relations": []}')
        LLMGraphExtractor(llm, relations=["depends_on", "owns"]).extract(_item("hello graph"))
        assert "hello graph" in llm.prompts[0]
        assert "depends_on, owns" in llm.prompts[0]
        assert "depends_on" in DEFAULT_RELATIONS
        assert "{content}" in _EXTRACTION_PROMPT

    def test_fail_soft_on_bad_json_or_provider_error(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            assert LLMGraphExtractor(FakeLLM("not json")).extract(_item()).nodes == []
            assert LLMGraphExtractor(FakeLLM("[1, 2]")).extract(_item()).nodes == []
            assert LLMGraphExtractor(FakeLLM(None)).extract(_item()).nodes == []

            class Boom:
                def invoke(self, messages, **kwargs):
                    raise RuntimeError("provider down")

            assert LLMGraphExtractor(Boom()).extract(_item()).edges == []
        assert "graph extraction" in caplog.text

    def test_indexes_memory_entries_under_the_entry_id(self) -> None:
        graph = KnowledgeGraph()
        extractor = LLMGraphExtractor(FakeLLM(json.dumps(PAYLOAD)))
        indexer = GraphIndexer(graph, extractors=[extractor])
        entry = MemoryEntry(id="mem-ana", content="Ana Lima leads Team Payments", tags=["people"])
        stats = indexer.index_entries([entry])
        assert stats.items == 1
        assert graph.items("ana lima") == ["mem-ana"]
        assert graph.neighbors("Ana") == ["team_payments", "billing_service", "mystery"]  # alias
        assert graph.mentions("what does Ana own?") == ["ana_lima"]
        hop = graph.explain("ana lima", "billing service")[0]
        assert (hop.relation, hop.provenance, hop.evidence) == (
            "on_call_for",
            "inferred",
            ("mem-ana",),
        )
        assert graph.query(["team payments"], top_k=1) == [
            ("mem-ana", graph.query(["team payments"])[0][1])
        ]


class TestReviewRegressions:
    @pytest.mark.parametrize("payload", ['{"entities": true}', '{"relations": 7}', "[1, 2]"])
    def test_unexpected_json_shapes_are_fail_soft(self, payload: str, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            found = LLMGraphExtractor(FakeLLM(payload)).extract(_item())
        assert found.nodes == []
        assert found.edges == []
        assert "graph extraction failed" in caplog.text

    def test_non_dict_entities_are_skipped_quietly(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            found = LLMGraphExtractor(FakeLLM('{"entities": [1, 2]}')).extract(_item())
        assert found.nodes == []
        assert caplog.text == ""
