"""Regression tests for the 2026-08-31 release-engineering review fixes.

Each test failed against the pre-fix code (see
docs/plans/2026-08-31-release-engineering.md, findings 6, 40, 41, 15/7).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from anchor.llm.base import BaseLLMProvider
from anchor.llm.errors import ServerError
from anchor.llm.models import LLMResponse, Message, Role, StopReason, StreamChunk
from anchor.llm.providers._openai_compat import parse_stream_chunks


class _MidStreamFailProvider(BaseLLMProvider):
    """Yields two chunks, then dies with a transient error — every attempt."""

    provider_name = "test"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def _resolve_api_key(self):
        return "k"

    def _do_invoke(self, messages, tools, **kwargs) -> LLMResponse:
        raise NotImplementedError

    async def _do_ainvoke(self, messages, tools, **kwargs) -> LLMResponse:
        raise NotImplementedError

    def _do_stream(self, messages, tools, **kwargs):
        self.calls += 1
        yield StreamChunk(content="hello")
        yield StreamChunk(content=" world")
        raise ServerError("500", provider="test")

    async def _do_astream(self, messages, tools, **kwargs):
        self.calls += 1
        yield StreamChunk(content="hello")
        yield StreamChunk(content=" world")
        raise ServerError("500", provider="test")


_MSGS = [Message(role=Role.USER, content="hi")]


class TestMidStreamRetryDoesNotDuplicate:
    """A transient error after chunks were yielded must propagate, not
    restart the stream — a retry would deliver the same text twice."""

    @patch("time.sleep")
    def test_stream_mid_error_propagates_without_rerun(self, mock_sleep):
        p = _MidStreamFailProvider(model="m", max_retries=2)
        received: list[str] = []
        with pytest.raises(ServerError):
            for chunk in p.stream(_MSGS):
                received.append(chunk.content)
        assert received == ["hello", " world"]  # no duplication
        assert p.calls == 1  # committed after first chunk: no re-run
        assert mock_sleep.call_count == 0

    async def test_astream_mid_error_propagates_without_rerun(self):
        p = _MidStreamFailProvider(model="m", max_retries=2)
        received: list[str] = []
        with pytest.raises(ServerError):
            async for chunk in p.astream(_MSGS):
                received.append(chunk.content)
        assert received == ["hello", " world"]
        assert p.calls == 1


def _tc_delta(index, id=None, name=None, arguments=None):
    tc = MagicMock()
    tc.index = index
    tc.id = id
    func = MagicMock()
    func.name = name
    func.arguments = arguments
    tc.function = func
    return tc


def _chunk(content=None, tool_calls=None, finish_reason=None):
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


class TestParseStreamChunksParallelDeltas:
    def test_all_parallel_tool_call_deltas_emitted(self):
        chunk = _chunk(tool_calls=[
            _tc_delta(0, id="c0", name="alpha", arguments=""),
            _tc_delta(1, id="c1", name="beta", arguments=""),
        ])
        chunks = parse_stream_chunks(chunk)
        indices = [c.tool_call_delta.index for c in chunks]
        names = [c.tool_call_delta.name for c in chunks]
        assert indices == [0, 1]
        assert names == ["alpha", "beta"]

    def test_deltas_on_finish_chunk_not_lost(self):
        chunk = _chunk(
            tool_calls=[_tc_delta(0, arguments='{"q": 1}')],
            finish_reason="tool_calls",
        )
        chunks = parse_stream_chunks(chunk)
        assert chunks[0].tool_call_delta is not None
        assert chunks[0].tool_call_delta.arguments_fragment == '{"q": 1}'
        assert chunks[-1].stop_reason == StopReason.TOOL_USE
