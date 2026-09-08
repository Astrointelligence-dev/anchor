"""LLMExtractor and LLMConsolidator against a scripted provider (roadmap #5)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from anchor.memory.consolidator import LLMConsolidator, apply_consolidation
from anchor.memory.extractor import LLMExtractor
from anchor.models.memory import ConversationTurn, MemoryEntry, MemoryType, _compute_content_hash
from anchor.protocols.memory import MemoryOperation
from anchor.storage.json_memory_store import InMemoryEntryStore
from tests.conftest import FakeLLM

ADD, UPDATE, DELETE, NONE = (
    MemoryOperation.ADD,
    MemoryOperation.UPDATE,
    MemoryOperation.DELETE,
    MemoryOperation.NONE,
)


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
                    {"content": "User lives in Rio "},  # repeated by the model, extra space
                    {"content": "  "},  # empty
                    {"tags": ["orphan"]},  # no content
                    "not a dict",
                    {"content": "User likes tea", "memory_type": "bogus"},
                    {"content": "User owns a boat", "memory_type": ["semantic"]},  # unhashable
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
            "User owns a boat",
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
        # keyword overlap ranks m2 ("coffee") first, so it is candidate [0]
        decision = {"fact": 0, "op": "update", "target": 0, "content": "User likes coffee now"}
        llm = FakeLLM(json.dumps([decision]))
        apply_consolidation(
            [MemoryEntry(content="User started liking coffee")], store, LLMConsolidator(llm)
        )
        by_id = {e.id: e for e in store.list_all()}
        assert set(by_id) == {"m1", "m2"}
        assert by_id["m2"].content == "User likes coffee now"

    def test_delete_invalidates_and_links_to_a_replacing_fact(self) -> None:
        # candidates are the union of each fact's ranking: m2 (coffee) first, then m1
        llm = FakeLLM(
            json.dumps(
                [
                    {"fact": 0, "op": "delete", "target": 0},
                    {"fact": 1, "op": "delete", "target": 1},
                    {"fact": 1, "op": "add"},
                ]
            )
        )
        sold = MemoryEntry(content="User stopped drinking coffee entirely")
        moved = MemoryEntry(content="User now lives abroad")
        results = LLMConsolidator(llm).consolidate([sold, moved], _existing())

        # per fact, adds/updates come first, then its deletes
        assert _ops(results) == [(DELETE, "m2"), (ADD, moved.id), (DELETE, "m1")]
        gone, replaced = results[0][1], results[2][1]
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
        assert "memory 9: added as-is" in caplog.text
        assert "memory -1: ignored" in caplog.text
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
        assert caplog.text.count("LLM consolidation failed") == 4

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

    def test_without_embed_fn_keyword_overlap_ranks_candidates_recency_breaks_ties(self) -> None:
        now = datetime.now(UTC)
        existing = [
            MemoryEntry(
                id="old_sp", content="User lives in São Paulo", updated_at=now - timedelta(days=30)
            ),
            *[
                MemoryEntry(
                    id=f"m{i}", content=f"memory {i}", updated_at=now - timedelta(minutes=i)
                )
                for i in range(5)
            ],
        ]
        llm = FakeLLM("[]")
        LLMConsolidator(llm, max_candidates=2).consolidate(
            [MemoryEntry(content="User moved to Rio; lives there now")], existing
        )
        shown = llm.prompts[0].split("NEW FACTS")[0]
        assert "[0] User lives in São Paulo" in shown  # overlap ("user", "lives") beats recency
        assert "[1] memory 0" in shown  # ties broken by updated_at
        assert "memory 1" not in shown

    def test_a_fact_whose_candidates_miss_the_cap_is_added_without_asking(self) -> None:
        vectors: dict[str, list[float]] = {}
        existing = []
        for cluster in range(3):
            for j in range(2):
                vec = [0.0] * 3
                vec[cluster] = 1.0 - 0.01 * j
                entry = MemoryEntry(id=f"c{cluster}_{j}", content=f"cluster {cluster} memory {j}")
                vectors[entry.content] = vec
                existing.append(entry)
        facts = []
        for cluster in range(3):
            vec = [0.0] * 3
            vec[cluster] = 1.0
            fact = MemoryEntry(content=f"fact about cluster {cluster}")
            vectors[fact.content] = vec
            facts.append(fact)
        llm = FakeLLM("[]")
        consolidator = LLMConsolidator(llm, embed_fn=vectors.__getitem__, top_k=2, max_candidates=4)

        results = consolidator.consolidate(facts, existing)

        prompt = llm.prompts[0]
        assert "fact about cluster 0" in prompt
        assert "fact about cluster 1" in prompt
        assert "fact about cluster 2" not in prompt  # its candidates fell past the cap
        assert _ops(results)[2] == (ADD, facts[2].id)


class TestLLMConsolidatorBatches:
    """Decisions in one answer apply against a working copy, in order."""

    def test_update_then_delete_keeps_the_rewrite_in_history(self) -> None:
        llm = FakeLLM(
            json.dumps(
                [
                    {"fact": 0, "op": "update", "target": 0, "content": "User lives in Rio"},
                    {"fact": 1, "op": "delete", "target": 0},
                ]
            )
        )
        facts = [MemoryEntry(content="moved to Rio"), MemoryEntry(content="left the country")]
        results = LLMConsolidator(llm).consolidate(facts, _existing())
        assert _ops(results) == [(UPDATE, "m1"), (DELETE, "m1")]
        gone = results[1][1]
        assert gone.content == "User lives in Rio"  # the merged version, not the stale snapshot
        assert gone.is_expired
        assert "invalidated_by" not in gone.metadata

    def test_delete_then_update_does_not_resurrect(self, caplog) -> None:
        llm = FakeLLM(
            json.dumps(
                [
                    {"fact": 0, "op": "delete", "target": 0},
                    {"fact": 1, "op": "update", "target": 0, "content": "User lives in Rio"},
                ]
            )
        )
        facts = [MemoryEntry(content="no longer in SP"), MemoryEntry(content="works from Rio")]
        store = InMemoryEntryStore()
        for e in _existing():
            store.add(e)
        with caplog.at_level(logging.WARNING):
            outcomes = apply_consolidation(facts, store, LLMConsolidator(llm))
        assert [op for op, _ in outcomes] == [DELETE, ADD]
        assert sorted(e.id for e in store.list_all()) == sorted(["m2", facts[1].id])
        assert "deleted memory 0" in caplog.text

    def test_two_updates_on_one_memory_chain(self) -> None:
        llm = FakeLLM(
            json.dumps(
                [
                    {"fact": 0, "op": "update", "target": 0, "content": "User lives in Rio"},
                    {
                        "fact": 1,
                        "op": "update",
                        "target": 0,
                        "content": "User lives in Rio, Botafogo",
                    },
                ]
            )
        )
        facts = [MemoryEntry(content="moved to Rio"), MemoryEntry(content="in Botafogo")]
        results = LLMConsolidator(llm).consolidate(facts, _existing())
        assert [e.content for _, e in results] == [
            "User lives in Rio",
            "User lives in Rio, Botafogo",
        ]
        assert results[1][1].metadata["previous_content"] == "User lives in Rio"

    def test_invalidated_by_points_at_the_id_actually_written(self) -> None:
        llm = FakeLLM(
            json.dumps(
                [
                    {"fact": 0, "op": "update", "target": 0, "content": "User lives in Rio"},
                    {"fact": 0, "op": "delete", "target": 1},
                ]
            )
        )
        fact = MemoryEntry(content="moved to Rio and quit coffee")  # overlap: m2 ("coffee") is [0]
        results = LLMConsolidator(llm).consolidate([fact], _existing())
        assert _ops(results) == [(UPDATE, "m2"), (DELETE, "m1")]
        assert results[1][1].metadata["invalidated_by"] == "m2"  # the update's id, not fact.id


class TestApplyConsolidation:
    def test_delete_of_an_entry_not_in_the_store_is_ignored(self, caplog) -> None:
        store = InMemoryEntryStore()
        store.add(MemoryEntry(id="m1", content="User lives in SP"))

        class DropNew:
            def consolidate(self, new_entries, existing):
                return [(DELETE, new_entries[0])]  # the pre-#5 reading: "drop this fact"

        with caplog.at_level(logging.WARNING):
            outcomes = apply_consolidation([MemoryEntry(content="noise")], store, DropNew())
        assert outcomes == []
        assert [e.id for e in store.list_all_unfiltered()] == ["m1"]
        assert "not in the store" in caplog.text

    def test_update_callback_receives_the_previous_entry(self) -> None:
        seen = []

        class Recorder:
            def on_consolidation(self, action, new_entry, existing_entry):
                seen.append(
                    (
                        action,
                        new_entry and new_entry.content,
                        existing_entry and existing_entry.content,
                    )
                )

        store = InMemoryEntryStore()
        store.add(MemoryEntry(id="m1", content="User lives in SP"))

        class Rewrite:
            def consolidate(self, new_entries, existing):
                return [(UPDATE, existing[0].model_copy(update={"content": "User lives in Rio"}))]

        apply_consolidation([MemoryEntry(content="x")], store, Rewrite(), callbacks=[Recorder()])
        assert seen == [("update", "User lives in Rio", "User lives in SP")]

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            LLMConsolidator(FakeLLM("[]"), new_threshold=1.5)
        with pytest.raises(ValueError):
            LLMConsolidator(FakeLLM("[]"), top_k=0)
        assert "LLMConsolidator(" in repr(LLMConsolidator(FakeLLM("[]")))
