"""Consolidation golden-set harness (roadmap #5) and the deterministic baselines."""

from __future__ import annotations

import zlib
from pathlib import Path

import pytest

from anchor.evaluation.consolidation import (
    ConsolidationCase,
    evaluate_consolidator,
    load_consolidation_set,
)
from anchor.evaluation.golden import assert_metric_floor
from anchor.memory.consolidator import SimilarityConsolidator
from anchor.memory.extractor import CallbackExtractor
from anchor.models.memory import ConversationTurn, MemoryEntry
from anchor.protocols.memory import MemoryOperation

GOLDEN = Path(__file__).parents[1] / "fixtures" / "consolidation_golden.jsonl"

# the deterministic extractor: every user turn, verbatim, is a fact
VERBATIM = CallbackExtractor(
    lambda turns: [{"content": t.content} for t in turns if t.role == "user"]
)


def _bow(text: str) -> list[float]:
    """Bag-of-words embedding: a stand-in for a real embedder in CI."""
    vec = [0.0] * 64
    for word in text.lower().split():
        vec[zlib.crc32(word.strip(".,!?;:").encode()) % 64] += 1.0
    return vec


class Replace:
    """DELETE everything in the store, ADD everything new."""

    def consolidate(self, new_entries, existing):
        gone = [(MemoryOperation.DELETE, e) for e in existing]
        return gone + [(MemoryOperation.ADD, e) for e in new_entries]


def _case(**kwargs) -> ConsolidationCase:
    base = {
        "name": "move",
        "existing": ["User lives in SP"],
        "turns": [{"role": "user", "content": "I moved to Rio"}],
        "live_contains": ["Rio"],
        "live_not_contains": ["SP"],
        "expected_live": 1,
    }
    return ConsolidationCase.model_validate({**base, **kwargs})


class TestLoader:
    def test_flat_turns_are_one_step_and_nested_are_many(self, tmp_path: Path) -> None:
        path = tmp_path / "g.jsonl"
        path.write_text(
            '{"name": "a", "turns": [{"role": "user", "content": "x"}], "expected_live": 1}\n'
            "\n"
            '{"name": "b", "turns": [[{"role": "user", "content": "x"}], '
            '[{"role": "assistant", "content": "y"}]], "expected_live": [0, 2]}\n'
        )
        a, b = load_consolidation_set(path)
        assert [len(step) for step in a.turns] == [1]
        assert isinstance(a.turns[0][0], ConversationTurn)
        assert [len(step) for step in b.turns] == [1, 1]
        assert b.expected_live == (0, 2)

    def test_invalid_line_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"name": "ok", "turns": [], "expected_live": 0}\n{"name": "no turns"}\n')
        with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
            load_consolidation_set(path)

    def test_python_built_cases_and_empty_steps(self) -> None:
        case = ConsolidationCase(
            name="c", turns=[ConversationTurn(role="user", content="hi")], expected_live=1
        )
        assert [len(step) for step in case.turns] == [1]
        with pytest.raises(ValueError, match="at least one turn"):
            ConsolidationCase(name="c", turns=[[]], expected_live=1)


class TestEvaluate:
    def test_state_size_and_probes_score_the_store_after_remember(self) -> None:
        case = _case(
            probes=[
                {"query": "Rio", "must_hit": ["Rio"]},
                {"query": "lives", "must_not_hit": ["SP"]},  # the stale fact must not come back
            ]
        )
        good = evaluate_consolidator(VERBATIM, Replace(), [case])
        assert good.summary() == {"passed": 1.0, "state_ok": 1.0, "size_ok": 1.0, "probes_ok": 1.0}
        assert good.results[0].live == ("I moved to Rio",)
        assert good.results[0].operations == ("delete", "add")

        naive = evaluate_consolidator(VERBATIM, None, [case])  # adds everything
        assert naive.summary() == {"passed": 0.0, "state_ok": 0.0, "size_ok": 0.0, "probes_ok": 0.0}
        assert naive.failures() == ["move"]
        with pytest.raises(AssertionError, match=r"'move'=0\.000"):
            assert_metric_floor(naive, "passed", 0.5)

    def test_steps_are_remembered_one_at_a_time(self) -> None:
        case = _case(
            turns=[
                [{"role": "user", "content": "I moved to Rio"}],
                [{"role": "user", "content": "Back to SP for good"}],
            ],
            live_contains=["SP"],
            live_not_contains=["Rio"],
        )
        report = evaluate_consolidator(VERBATIM, Replace(), [case])
        assert report.results[0].live == ("Back to SP for good",)
        assert report.results[0].operations == ("delete", "add", "delete", "add")
        assert report.mean("passed") == 1.0

    def test_expected_live_range_and_seeded_ids(self) -> None:
        seen: list[list[str]] = []

        class Spy:
            def consolidate(self, new_entries, existing):
                seen.append([e.id for e in existing])
                return [(MemoryOperation.ADD, e) for e in new_entries]

        case = _case(
            existing=["a", "b"], live_contains=[], live_not_contains=[], expected_live=[3, 4]
        )
        assert evaluate_consolidator(VERBATIM, Spy(), [case]).mean("size_ok") == 1.0
        assert seen == [["e0", "e1"]]
        assert MemoryEntry(content="x").is_expired is False


class TestGoldenSetBaselines:
    """The deterministic path cannot see a contradiction: these are regression floors,
    and the ceiling the LLM pair has to beat (tests/live/test_memory_llm_live.py)."""

    def test_add_everything_baseline(self) -> None:
        report = evaluate_consolidator(VERBATIM, None, load_consolidation_set(GOLDEN))
        print(report.summary())
        assert_metric_floor(report, "passed", 0.25)  # measured 0.271: only the ADD scenarios

    def test_similarity_consolidator_baseline(self) -> None:
        cases = load_consolidation_set(GOLDEN)
        report = evaluate_consolidator(
            VERBATIM, SimilarityConsolidator(_bow, similarity_threshold=0.5), cases
        )
        print(report.summary())
        assert_metric_floor(report, "passed", 0.33)  # measured 0.354: + a few paraphrases
        assert len(cases) == 48
