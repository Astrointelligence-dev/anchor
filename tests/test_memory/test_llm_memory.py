"""LLMExtractor and LLMConsolidator against a scripted provider (roadmap #5)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from anchor.memory.consolidator import LLMConsolidator
from anchor.memory.extractor import LLMExtractor
from anchor.models.memory import ConversationTurn, MemoryEntry, MemoryType, _compute_content_hash
from anchor.pipeline.memory_steps import _store_with_consolidation
from anchor.protocols.memory import MemoryOperation
from anchor.storage.json_memory_store import InMemoryEntryStore

ADD, UPDATE, DELETE, NONE = (
    MemoryOperation.ADD,
    MemoryOperation.UPDATE,
    MemoryOperation.DELETE,
    MemoryOperation.NONE,
)


class FakeLLM:
    """Answers with *content* (a str, or a list of str consumed in order)."""

    def __init__(self, content: str | list[str] | None) -> None:
        self._answers = content if isinstance(content, list) else [content]
        self.prompts: list[str] = []

    def invoke(self, messages, **kwargs):
        self.prompts.append(messages[0].content)
        answer = self._answers.pop(0) if len(self._answers) > 1 else self._answers[0]
        return SimpleNamespace(content=answer)


class Boom:
    def invoke(self, messages, **kwargs):
        raise RuntimeError("provider down")


def _turn(role: str, content: str) -> ConversationTurn:
    return ConversationTurn(role=role, content=content)


def _ops(results):
    return [(op, e.id if e is not None else None) for op, e in results]


# ---------------------------------------------------------------------------
# LLMExtractor
# ---------------------------------------------------------------------------


class TestLLMExtractor:
    def test_facts_become_entries_and_tool_turns_are_ignored(self) -> None:
        llm = FakeLLM(
            "```json\n"
            + json.dumps(
                [
                    {
                        "content": "User lives in Rio",
                        "tags": ["location", 3],
                        "memory_type": "semantic",
                    },
                    {"content": "User ran a marathon", "memory_type": "episodic"},
                    {"content": "User lives in Rio"},  # repeated by the model
                    {"content": "  "},  # empty
                    {"tags": ["orphan"]},  # no content
                    "not a dict",
                    {"content": "User likes tea", "memory_type": "bogus"},
                ]
            )
            + "\n```"
        )
        turns = [
            _turn("user", "I moved to Rio last month"),
            _turn("tool", "[Tool: search] Input: rio → Result: ..."),
            _turn("assistant", "Nice, how is it going?"),
        ]
        entries = LLMExtractor(llm).extract(turns)

        assert [e.content for e in entries] == [
            "User lives in Rio",
            "User ran a marathon",
            "User likes tea",
        ]
        assert entries[0].tags == ["location"]
        assert entries[1].memory_type is MemoryType.EPISODIC
        assert entries[2].memory_type is MemoryType.SEMANTIC  # bogus type falls back
        kept = (turns[0], turns[2])
        assert entries[0].source_turns == [t.timestamp.isoformat() for t in kept]
        prompt = llm.prompts[0]
        assert "user: I moved to Rio last month" in prompt
        assert "assistant: Nice" in prompt
        assert "[Tool: search]" not in prompt

    def test_only_excluded_roles_means_no_call(self) -> None:
        llm = FakeLLM("[]")
        assert LLMExtractor(llm).extract([_turn("tool", "x"), _turn("system", "y")]) == []
        assert llm.prompts == []
        assert LLMExtractor(llm, roles=("system",)).extract([_turn("system", "y")]) == []
        assert len(llm.prompts) == 1

    def test_fail_soft(self, caplog) -> None:
        turns = [_turn("user", "hi")]
        with caplog.at_level(logging.WARNING):
            assert LLMExtractor(FakeLLM("not json")).extract(turns) == []
            assert LLMExtractor(FakeLLM('{"content": "obj"}')).extract(turns) == []
            assert LLMExtractor(FakeLLM(None)).extract(turns) == []
            assert LLMExtractor(Boom()).extract(turns) == []
        assert caplog.text.count("LLM memory extraction failed") == 4
        assert "LLMExtractor(" in repr(LLMExtractor(Boom()))


# ---------------------------------------------------------------------------
# LLMConsolidator
# ---------------------------------------------------------------------------


def _existing() -> list[MemoryEntry]:
    """m1 is the most recently updated, so without embed_fn it is candidate [0]."""
    now = datetime.now(UTC)
    return [
        MemoryEntry(id="m1", content="User lives in São Paulo", tags=["location"], updated_at=now),
        MemoryEntry(id="m2", content="User dislikes coffee", updated_at=now - timedelta(minutes=1)),
    ]


class TestLLMConsolidatorDecisions:
    def test_move_becomes_an_update_under_the_same_id(self) -> None:
        llm = FakeLLM(
            json.dumps([{"fact": 0, "op": "update", "target": 0, "content": "User lives in Rio"}])
        )
        fact = MemoryEntry(content="User moved to Rio last month", tags=["move"])
        results = LLMConsolidator(llm).consolidate([fact], _existing())

        assert _ops(results) == [(UPDATE, "m1")]
        updated = results[0][1]
        assert updated.content == "User lives in Rio"
        assert updated.content_hash == _compute_content_hash("User lives in Rio")
        assert updated.metadata["previous_content"] == "User lives in São Paulo"
        assert updated.tags == ["location", "move"]
        prompt = llm.prompts[0]
        assert "[0] User lives in São Paulo" in prompt
        assert "[1] User dislikes coffee" in prompt
        assert "[0] User moved to Rio last month" in prompt

    def test_applied_through_the_pipeline_one_live_entry_remains(self) -> None:
        store = InMemoryEntryStore()
        for e in _existing():
            store.add(e)
        decision = {"fact": 0, "op": "update", "target": 1, "content": "User likes coffee now"}
        llm = FakeLLM(json.dumps([decision]))
        _store_with_consolidation(
            [MemoryEntry(content="User started liking coffee")], store, LLMConsolidator(llm)
        )
        by_id = {e.id: e for e in store.list_all()}
        assert set(by_id) == {"m1", "m2"}
        assert by_id["m2"].content == "User likes coffee now"

    def test_delete_invalidates_and_links_to_a_replacing_fact(self) -> None:
        llm = FakeLLM(
            json.dumps(
                [
                    {"fact": 0, "op": "delete", "target": 1},
                    {"fact": 1, "op": "delete", "target": 0},
                    {"fact": 1, "op": "add"},
                ]
            )
        )
        sold = MemoryEntry(content="User stopped drinking coffee entirely")
        moved = MemoryEntry(content="User now lives abroad")
        results = LLMConsolidator(llm).consolidate([sold, moved], _existing())

        assert _ops(results) == [(DELETE, "m2"), (DELETE, "m1"), (ADD, moved.id)]
        gone, replaced = results[0][1], results[1][1]
        assert gone.is_expired
        assert "invalidated_by" not in gone.metadata
        assert replaced.is_expired
        assert replaced.metadata["invalidated_by"] == moved.id

    def test_none_and_omitted_and_unknown_targets(self, caplog) -> None:
        llm = FakeLLM(
            json.dumps(
                [
                    {"fact": 0, "op": "none"},
                    {"fact": 2, "op": "update", "target": 9, "content": "x"},
                    {"fact": 3, "op": "delete", "target": -1},
                    {"fact": 7, "op": "add"},
                    {"fact": 0, "op": "teleport"},
                    "garbage",
                ]
            )
        )
        facts = [MemoryEntry(content=f"fact {i}") for i in range(4)]
        with caplog.at_level(logging.WARNING):
            results = LLMConsolidator(llm).consolidate(facts, _existing())

        assert _ops(results) == [(NONE, None), (ADD, facts[1].id), (ADD, facts[2].id)]
        assert "unknown memory 9: added as-is" in caplog.text
        assert "unknown memory -1: ignored" in caplog.text
        assert "named fact 7 of 4" in caplog.text
        assert "unusable decision" in caplog.text

    def test_fail_soft_adds_everything(self, caplog) -> None:
        facts = [MemoryEntry(content="a"), MemoryEntry(content="b")]
        with caplog.at_level(logging.WARNING):
            for llm in (FakeLLM("not json"), FakeLLM('{"fact": 0}'), FakeLLM(None), Boom()):
                assert _ops(LLMConsolidator(llm).consolidate(facts, _existing())) == [
                    (ADD, facts[0].id),
                    (ADD, facts[1].id),
                ]
        assert caplog.text.count("adding 2 fact(s) as-is") == 4

    def test_results_keep_the_order_of_new_entries(self) -> None:
        dup = MemoryEntry(content="User lives in São Paulo")  # exact hash → NONE, never asked
        fresh = MemoryEntry(content="User plays chess")
        llm = FakeLLM(json.dumps([{"fact": 0, "op": "add"}]))
        results = LLMConsolidator(llm).consolidate([dup, fresh], _existing())
        assert _ops(results) == [(NONE, None), (ADD, fresh.id)]
        assert "[0] User plays chess" in llm.prompts[0]
        assert "São Paulo" not in llm.prompts[0].split("NEW FACTS")[1]


class TestLLMConsolidatorGates:
    def test_hash_duplicates_and_empty_store_never_call_the_model(self) -> None:
        llm = FakeLLM("[]")
        existing = _existing()
        dup = MemoryEntry(content="User dislikes coffee")
        assert _ops(LLMConsolidator(llm).consolidate([dup], existing)) == [(NONE, None)]
        fresh = MemoryEntry(content="User plays chess")
        assert _ops(LLMConsolidator(llm).consolidate([fresh], [])) == [(ADD, fresh.id)]
        assert _ops(LLMConsolidator(llm).consolidate([], existing)) == []
        assert llm.prompts == []

    def test_embed_fn_skips_the_model_for_novel_facts_and_picks_top_k(self) -> None:
        vectors = {
            "User lives in São Paulo": [1.0, 0.0, 0.0],
            "User dislikes coffee": [0.0, 1.0, 0.0],
            "User lives in Rio now": [0.9, 0.1, 0.0],  # near m1 → asked, candidates = top_k
            "User plays chess": [0.0, 0.0, 1.0],  # orthogonal → plain ADD, no call
        }
        decision = {"fact": 0, "op": "update", "target": 0, "content": "User lives in Rio"}
        llm = FakeLLM(json.dumps([decision]))
        consolidator = LLMConsolidator(llm, embed_fn=vectors.__getitem__, top_k=1)
        moved = MemoryEntry(content="User lives in Rio now")
        chess = MemoryEntry(content="User plays chess")

        results = consolidator.consolidate([chess, moved], _existing())

        assert _ops(results) == [(ADD, chess.id), (UPDATE, "m1")]
        assert len(llm.prompts) == 1
        assert "[0] User lives in São Paulo" in llm.prompts[0]
        assert "coffee" not in llm.prompts[0]  # top_k=1: only the nearest candidate is shown

    def test_without_embed_fn_the_most_recent_candidates_are_shown(self) -> None:
        now = datetime.now(UTC)
        existing = [
            MemoryEntry(id=f"m{i}", content=f"memory {i}", updated_at=now - timedelta(minutes=i))
            for i in range(5)
        ]
        llm = FakeLLM("[]")
        LLMConsolidator(llm, max_candidates=2).consolidate([MemoryEntry(content="new")], existing)
        shown = llm.prompts[0].split("NEW FACTS")[0]
        assert "[0] memory 0" in shown
        assert "[1] memory 1" in shown
        assert "memory 2" not in shown

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            LLMConsolidator(FakeLLM("[]"), new_threshold=1.5)
        with pytest.raises(ValueError):
            LLMConsolidator(FakeLLM("[]"), top_k=0)
        assert "LLMConsolidator(" in repr(LLMConsolidator(FakeLLM("[]")))
