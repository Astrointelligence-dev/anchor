"""MemoryManager.remember() / after_turn(): the conversation becomes facts (roadmap #5)."""

from __future__ import annotations

import logging

import pytest

from anchor.memory.manager import MemoryManager
from anchor.models.memory import ConversationTurn, MemoryEntry
from anchor.protocols.memory import MemoryOperation
from anchor.storage.json_memory_store import InMemoryEntryStore
from tests.conftest import FakeTokenizer

ADD, DELETE = MemoryOperation.ADD, MemoryOperation.DELETE


class LastUserFactExtractor:
    """One fact per call: the content of the last user turn, prefixed."""

    def __init__(self) -> None:
        self.calls: list[list[ConversationTurn]] = []

    def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]:
        self.calls.append(list(turns))
        users = [t for t in turns if t.role == "user"]
        return [MemoryEntry(content=f"fact: {users[-1].content}")] if users else []


class ReplaceConsolidator:
    """Every new fact replaces whatever is in the store: DELETE old + ADD new."""

    def consolidate(self, new_entries, existing):
        return [(DELETE, e) for e in existing] + [(ADD, e) for e in new_entries]


class Recorder:
    def __init__(self) -> None:
        self.extractions: list[tuple[int, int]] = []
        self.consolidations: list[tuple[str, bool, bool]] = []

    def on_extraction(self, turns, entries) -> None:
        self.extractions.append((len(turns), len(entries)))

    def on_consolidation(self, action, new_entry, existing_entry) -> None:
        self.consolidations.append((action, new_entry is not None, existing_entry is not None))


def _manager(**kwargs) -> MemoryManager:
    return MemoryManager(tokenizer=FakeTokenizer(), persistent_store=InMemoryEntryStore(), **kwargs)


class TestRemember:
    def test_extracts_the_window_and_applies_the_operations(self) -> None:
        extractor, recorder = LastUserFactExtractor(), Recorder()
        manager = _manager(
            extractor=extractor,
            consolidator=ReplaceConsolidator(),
            extract_window=2,
            callbacks=[recorder],
        )
        for i in range(3):
            manager.add_user_message(f"u{i}")
            manager.add_assistant_message(f"a{i}")

        ops = manager.remember()

        assert [t.content for t in extractor.calls[0]] == ["u2", "a2"]  # the window
        assert [(op, e.content) for op, e in ops] == [(ADD, "fact: u2")]
        assert [e.content for e in manager.get_all_facts()] == ["fact: u2"]

        manager.add_user_message("u3")
        manager.add_assistant_message("a3")
        ops = manager.remember()

        assert [(op, e.content) for op, e in ops] == [(DELETE, "fact: u2"), (ADD, "fact: u3")]
        assert ops[0][1].is_expired  # the soft delete came back marked
        assert [e.content for e in manager.get_all_facts()] == ["fact: u3"]
        assert recorder.extractions == [(2, 1), (2, 1)]
        assert recorder.consolidations == [
            ("add", True, False),
            ("delete", False, True),
            ("add", True, False),
        ]

    def test_without_extractor_store_or_turns_nothing_happens(self) -> None:
        assert _manager().remember() == []
        no_store = MemoryManager(tokenizer=FakeTokenizer(), extractor=LastUserFactExtractor())
        no_store.add_user_message("x")
        assert no_store.remember() == []
        extractor = LastUserFactExtractor()
        assert _manager(extractor=extractor).remember() == []
        assert extractor.calls == []

    def test_without_consolidator_every_fact_is_added(self) -> None:
        manager = _manager(extractor=LastUserFactExtractor())
        manager.add_user_message("u0")
        manager.remember()
        manager.add_user_message("u1")
        assert [(op, e.content) for op, e in manager.remember()] == [(ADD, "fact: u1")]
        assert sorted(e.content for e in manager.get_all_facts()) == ["fact: u0", "fact: u1"]


class TestAfterTurn:
    def test_runs_every_n_turns_and_remember_resets_the_count(self) -> None:
        extractor = LastUserFactExtractor()
        manager = _manager(extractor=extractor, remember_every=2)
        manager.add_user_message("u0")

        assert manager.after_turn() == []  # 1 of 2
        assert len(manager.after_turn()) == 1  # 2 of 2 → remember
        assert manager.after_turn() == []  # 1 of 2
        manager.add_user_message("u1")
        manager.remember()  # an explicit flush resets the count
        assert manager.after_turn() == []  # 1 of 2 again
        assert len(extractor.calls) == 2


    def test_zero_means_manual_only(self) -> None:
        extractor = LastUserFactExtractor()
        manager = _manager(extractor=extractor, remember_every=0)
        manager.add_user_message("u0")
        assert manager.after_turn() == []
        assert manager.after_turn() == []
        assert extractor.calls == []
        assert len(manager.remember()) == 1

    def test_failures_are_logged_not_raised(self, caplog) -> None:
        class Broken:
            def extract(self, turns):
                raise RuntimeError("boom")

        manager = _manager(extractor=Broken())
        manager.add_user_message("u0")
        with caplog.at_level(logging.ERROR):
            assert manager.after_turn() == []
        assert "remember() failed" in caplog.text
        with pytest.raises(RuntimeError, match="boom"):
            manager.remember()

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="extract_window"):
            _manager(extract_window=0)
        with pytest.raises(ValueError, match="remember_every"):
            _manager(remember_every=-1)


class TestCursor:
    """A turn is extracted once: remember() reads only what came after the last call."""

    def test_no_re_extraction_and_nothing_new_means_no_call(self) -> None:
        extractor = LastUserFactExtractor()
        manager = _manager(extractor=extractor, extract_window=10)
        manager.add_user_message("u0")
        manager.add_assistant_message("a0")
        manager.remember()
        assert manager.remember() == []  # nothing new: no extractor call
        manager.add_user_message("u1")
        manager.remember()
        assert [[t.content for t in call] for call in extractor.calls] == [["u0", "a0"], ["u1"]]

    def test_provider_failure_keeps_the_turns_for_the_next_attempt(self, caplog) -> None:
        calls = {"n": 0}

        class Flaky:
            def extract(self, turns):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("429")
                return [MemoryEntry(content=f"fact: {turns[-1].content}")]

        manager = _manager(extractor=Flaky())
        manager.add_user_message("u0")
        with caplog.at_level(logging.ERROR):
            assert manager.after_turn() == []  # failed, logged, cursor not advanced
        assert [e.content for op, e in manager.after_turn()] == ["fact: u0"]  # retried

    def test_tool_turns_never_eat_the_window(self) -> None:
        extractor = LastUserFactExtractor()
        manager = _manager(extractor=extractor, extract_window=2)
        manager.add_user_message("u0")
        for i in range(10):
            manager.add_tool_message(f"[Tool: t] Input: {i} → Result: ok")
        manager.add_assistant_message("a0")
        manager.remember()
        assert [t.content for t in extractor.calls[0]] == ["u0", "a0"]

    def test_rebuilt_turn_objects_still_match_the_cursor(self) -> None:
        extractor = LastUserFactExtractor()
        manager = _manager(extractor=extractor)
        manager.add_user_message("u0")
        manager.remember()
        copies = [t.model_copy() for t in manager.conversation.turns]  # equal, not identical
        manager._last_remembered = copies[-1]
        manager.add_user_message("u1")
        manager.remember()
        assert [[t.content for t in call] for call in extractor.calls] == [["u0"], ["u1"]]

    def test_window_caps_the_fresh_turns_and_survives_eviction(self) -> None:
        extractor = LastUserFactExtractor()
        manager = _manager(extractor=extractor, extract_window=2, conversation_tokens=6)
        for i in range(3):
            manager.add_user_message(f"u{i}")
        manager.remember()  # fresh = u0,u1,u2 → capped to the newest 2
        assert [t.content for t in extractor.calls[0]] == ["u1", "u2"]
        for i in range(3, 12):  # evicts everything remembered so far
            manager.add_user_message(f"u{i}")
        manager.remember()  # cursor gone from the window: take the fresh tail
        assert [t.content for t in extractor.calls[1]] == ["u10", "u11"]
