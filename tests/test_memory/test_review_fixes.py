"""Regression tests for the 2026-08-31 release-engineering review fixes
(progressive memory). Each test failed against the pre-fix code — see
docs/plans/2026-08-31-release-engineering.md, findings 14-17.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock

from anchor.memory.progressive import ProgressiveSummarizationMemory
from anchor.models.memory import TierConfig
from tests.conftest import FakeTokenizer
from tests.test_memory.test_progressive import _make_llm_response, _make_mock_llm

_SMALL_TIERS = [
    TierConfig(level=0, max_tokens=5),
    TierConfig(level=1, max_tokens=1000, target_tokens=50),
    TierConfig(level=2, max_tokens=500, target_tokens=20),
    TierConfig(level=3, max_tokens=200, target_tokens=5),
]


def _make_memory(mock_llm=None) -> ProgressiveSummarizationMemory:
    return ProgressiveSummarizationMemory(
        max_tokens=200,
        llm=mock_llm or _make_mock_llm(summary="S"),
        tier_config=_SMALL_TIERS,
        tokenizer=FakeTokenizer(),
    )


class TestEvictionCallbackRunsOutsideWindowLock:
    """Finding 15: the eviction cascade ran under SlidingWindowMemory's
    non-reentrant lock — a callback reading the memory back deadlocked."""

    def test_callback_reading_memory_does_not_deadlock(self):
        observed: list[int] = []

        class ReadBack:
            def on_tier_cascade(self, from_level, to_level, before, after):
                # Re-enters the window (total_tokens takes its lock).
                observed.append(mem.total_tokens)

        mem = ProgressiveSummarizationMemory(
            max_tokens=200,
            llm=_make_mock_llm(summary="S"),
            tier_config=_SMALL_TIERS,
            tokenizer=FakeTokenizer(),
            callbacks=[ReadBack()],
        )

        def overflow():
            for i in range(6):
                mem.add_message("user", f"w{i} w{i} w{i}")

        t = threading.Thread(target=overflow)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive(), "eviction deadlocked under the window lock"
        assert observed  # the cascade fired and the read-back succeeded


class TestSwapRestoreUnderLock:
    """Finding 14: the _on_evict swap/restore ran outside the RLock;
    interleaved callers permanently installed a dead capture lambda."""

    def test_callback_restored_after_gated_concurrent_adds(self):
        """A third thread holds mem._lock while two aadd threads start —
        pre-fix both swapped BEFORE blocking on the lock, so the loser's
        restore installed the other's capture lambda permanently."""
        import time

        for _round in range(12):
            mock_llm = _make_mock_llm(summary="S")
            mock_llm.ainvoke = AsyncMock(return_value=_make_llm_response("S"))
            mem = _make_memory(mock_llm)
            original = mem._window._on_evict

            held = threading.Event()
            release = threading.Event()

            def holder(m=mem, held=held, release=release):
                with m._lock:
                    held.set()
                    release.wait(timeout=5)

            def run_add(tag, m=mem):
                asyncio.run(m.aadd_message("user", f"{tag} {tag} {tag}"))

            t_hold = threading.Thread(target=holder)
            t_hold.start()
            assert held.wait(timeout=5)
            t1 = threading.Thread(target=run_add, args=("a",))
            t2 = threading.Thread(target=run_add, args=("b",))
            t1.start()
            t2.start()
            time.sleep(0.05)  # both adds reach (pre-fix: pass) the swap
            release.set()
            for t in (t1, t2, t_hold):
                t.join(timeout=5)

            assert mem._window._on_evict is original, (
                f"round {_round}: a capture lambda leaked as the callback"
            )

        # And the surviving callback must still work: sync evictions
        # reach the tiers instead of extending a dead list.
        for i in range(6):
            mem.add_message("user", f"post{i} post{i} post{i}")
        assert mem.tiers[1] is not None


class TestAsyncEvictionSerialized:
    """Finding 16: the async read-summarize-write spanned awaits without
    serialization — concurrent evictions lost the first one's turns."""

    def test_concurrent_evictions_accumulate_turn_count(self):
        mock_llm = _make_mock_llm(summary="S")

        async def slow_ainvoke(*args, **kwargs):
            await asyncio.sleep(0.05)
            return _make_llm_response("S")

        mock_llm.ainvoke = AsyncMock(side_effect=slow_ainvoke)
        mem = _make_memory(mock_llm)

        async def run():
            # Seed the window, then two adds that each force an eviction.
            await mem.aadd_message("user", "a1 a2 a3 a4")
            await asyncio.gather(
                mem.aadd_message("user", "b1 b2 b3 b4"),
                mem.aadd_message("user", "c1 c2 c3 c4"),
            )

        asyncio.run(run())
        tier1 = mem.tiers[1]
        assert tier1 is not None
        # Both evictions' turns must be accounted — the lost-update bug
        # left only the second batch (source_turn_count == 1).
        assert tier1.source_turn_count == 2


class TestCompactionErrorCarriesRootCause:
    """Finding 17: a synthetic Exception('compaction failed') replaced the
    real error — callback consumers could never see the cause."""

    def test_real_exception_reaches_callback(self):
        seen: list[Exception] = []

        class Recorder:
            def on_compaction_error(self, tier, error):
                seen.append(error)

        mock_llm = _make_mock_llm(summary="S")
        mem = ProgressiveSummarizationMemory(
            max_tokens=200,
            llm=mock_llm,
            tier_config=_SMALL_TIERS,
            tokenizer=FakeTokenizer(),
            callbacks=[Recorder()],
        )
        boom = ValueError("boom: tokenizer exploded")

        class BoomCompactor:
            def summarize(self, *a, **k):
                raise boom

            def extract_facts(self, *a, **k):
                return []

        mem._compactor = BoomCompactor()

        for i in range(6):
            mem.add_message("user", f"w{i} w{i} w{i}")

        assert seen, "on_compaction_error never fired"
        assert seen[0] is boom
