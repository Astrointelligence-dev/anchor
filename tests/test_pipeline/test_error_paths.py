"""Tests for pipeline error paths: exception wrapping, type checks, and diagnostics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from anchor.exceptions import AstroContextError, FormatterError, PipelineExecutionError
from anchor.models.context import ContextItem
from anchor.models.query import QueryBundle
from anchor.pipeline.pipeline import ContextPipeline
from anchor.pipeline.step import PipelineStep
from tests.conftest import FakeTokenizer
from tests.test_pipeline.conftest import make_pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _failing_step(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """Step that raises a generic Exception (not AstroContextError)."""
    raise RuntimeError("something went wrong")


def _bad_return_step(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """Step that returns a string instead of a list."""
    return "not a list"  # type: ignore[return-value]


async def _async_failing_step(
    items: list[ContextItem], query: QueryBundle
) -> list[ContextItem]:
    """Async step that raises a generic Exception."""
    raise RuntimeError("async boom")


async def _async_bad_return_step(
    items: list[ContextItem], query: QueryBundle
) -> list[ContextItem]:
    """Async step that returns a string instead of a list."""
    return "not a list"  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 1. Pipeline step exception wrapping (sync build)
# ---------------------------------------------------------------------------


class TestSyncStepExceptionWrapping:
    """A step that raises a generic Exception should be wrapped in AstroContextError."""

    def test_generic_exception_wrapped_in_anchor_error(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_step(PipelineStep(name="boom-step", fn=_failing_step))

        with pytest.raises(AstroContextError, match="boom-step"):
            pipeline.build(QueryBundle(query_str="test"))

    def test_wrapped_exception_chains_original_cause(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_step(PipelineStep(name="boom-step", fn=_failing_step))

        with pytest.raises(AstroContextError) as exc_info:
            pipeline.build(QueryBundle(query_str="test"))

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "something went wrong" in str(exc_info.value.__cause__)


# ---------------------------------------------------------------------------
# 2. Pipeline step TypeError for wrong return type
# ---------------------------------------------------------------------------


class TestStepReturnTypeValidation:
    """A step that returns a non-list should raise TypeError."""

    def test_sync_step_bad_return_raises_type_error(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_step(PipelineStep(name="bad-return", fn=_bad_return_step))

        with pytest.raises(TypeError, match="must return a list"):
            pipeline.build(QueryBundle(query_str="test"))

    @pytest.mark.asyncio
    async def test_async_step_bad_return_raises_type_error(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_step(
            PipelineStep(name="async-bad-return", fn=_async_bad_return_step, is_async=True)
        )

        with pytest.raises(TypeError, match="must return a list"):
            await pipeline.abuild(QueryBundle(query_str="test"))


# ---------------------------------------------------------------------------
# 3. Async step in sync pipeline
# ---------------------------------------------------------------------------


class TestAsyncStepInSyncPipeline:
    """Calling build() with an async step should raise TypeError."""

    def test_async_step_raises_type_error_in_build(self) -> None:
        async def async_noop(
            items: list[ContextItem], query: QueryBundle
        ) -> list[ContextItem]:
            return items

        pipeline = make_pipeline()
        pipeline.add_step(PipelineStep(name="async-noop", fn=async_noop, is_async=True))

        with pytest.raises(TypeError, match=r"async.*cannot be called synchronously"):
            pipeline.build(QueryBundle(query_str="test"))


# ---------------------------------------------------------------------------
# 4. Formatter error wrapping
# ---------------------------------------------------------------------------


class TestFormatterErrorWrapping:
    """A formatter that raises should be wrapped in FormatterError."""

    def test_formatter_exception_wrapped_in_formatter_error(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_system_prompt("System message")

        # Create a mock formatter whose format() raises
        mock_formatter = MagicMock()
        mock_formatter.format_type = "broken"
        mock_formatter.format.side_effect = ValueError("bad format")
        pipeline.with_formatter(mock_formatter)

        with pytest.raises(FormatterError, match="Formatter 'broken' failed"):
            pipeline.build(QueryBundle(query_str="test"))

    def test_formatter_error_chains_original_cause(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_system_prompt("System message")

        mock_formatter = MagicMock()
        mock_formatter.format_type = "broken"
        mock_formatter.format.side_effect = ValueError("bad format")
        pipeline.with_formatter(mock_formatter)

        with pytest.raises(FormatterError) as exc_info:
            pipeline.build(QueryBundle(query_str="test"))

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)

    @pytest.mark.asyncio
    async def test_formatter_error_in_abuild(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_system_prompt("System message")

        mock_formatter = MagicMock()
        mock_formatter.format_type = "broken-async"
        mock_formatter.format.side_effect = RuntimeError("async format fail")
        pipeline.with_formatter(mock_formatter)

        with pytest.raises(FormatterError, match="Formatter 'broken-async' failed"):
            await pipeline.abuild(QueryBundle(query_str="test"))


# ---------------------------------------------------------------------------
# 5. build() with plain string query
# ---------------------------------------------------------------------------


class TestBuildWithStringQuery:
    """build('my query') should work the same as build(QueryBundle(query_str='my query'))."""

    def test_build_string_query_returns_context_result(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_system_prompt("Hello")

        result = pipeline.build("my query")

        assert len(result.window.items) == 1
        assert result.window.items[0].content == "Hello"

    def test_build_string_matches_query_bundle(self) -> None:
        """String and QueryBundle produce the same result."""
        pipeline = make_pipeline()
        pipeline.add_system_prompt("Hello")

        result_str = pipeline.build("my query")
        result_bundle = pipeline.build(QueryBundle(query_str="my query"))

        assert len(result_str.window.items) == len(result_bundle.window.items)
        assert result_str.formatted_output == result_bundle.formatted_output
        assert result_str.format_type == result_bundle.format_type


# ---------------------------------------------------------------------------
# 6. abuild() with plain string query
# ---------------------------------------------------------------------------


class TestAbuildWithStringQuery:
    """abuild('my query') should work the same as abuild(QueryBundle(...))."""

    @pytest.mark.asyncio
    async def test_abuild_string_query_returns_context_result(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_system_prompt("Hello")

        result = await pipeline.abuild("my query")

        assert len(result.window.items) == 1
        assert result.window.items[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_abuild_string_matches_query_bundle(self) -> None:
        """String and QueryBundle produce the same result via abuild."""
        pipeline = make_pipeline()
        pipeline.add_system_prompt("Hello")

        result_str = await pipeline.abuild("my query")
        result_bundle = await pipeline.abuild(QueryBundle(query_str="my query"))

        assert len(result_str.window.items) == len(result_bundle.window.items)
        assert result_str.formatted_output == result_bundle.formatted_output
        assert result_str.format_type == result_bundle.format_type


# ---------------------------------------------------------------------------
# 7. @pipeline.step rejecting async function
# ---------------------------------------------------------------------------


class TestStepDecoratorRejectsAsync:
    """The @pipeline.step decorator should raise TypeError when given an async function."""

    def test_step_decorator_rejects_async_bare(self) -> None:
        pipeline = make_pipeline()

        with pytest.raises(TypeError, match=r"async.*use @pipeline.async_step"):

            @pipeline.step
            async def bad_step(  # type: ignore[arg-type]
                items: list[ContextItem], query: QueryBundle
            ) -> list[ContextItem]:
                return items

    def test_step_decorator_rejects_async_with_name(self) -> None:
        pipeline = make_pipeline()

        with pytest.raises(TypeError, match=r"async.*use @pipeline.async_step"):

            @pipeline.step(name="bad-async")
            async def bad_step(  # type: ignore[arg-type]
                items: list[ContextItem], query: QueryBundle
            ) -> list[ContextItem]:
                return items


# ---------------------------------------------------------------------------
# 8. Failed step diagnostics
# ---------------------------------------------------------------------------


class TestFailedStepDiagnostics:
    """When a step fails, diagnostics['failed_step'] should be set to the step name."""

    def test_failed_step_recorded_in_diagnostics(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_step(PipelineStep(name="good-step", fn=lambda items, q: items))
        pipeline.add_step(PipelineStep(name="crashing-step", fn=_failing_step))

        with pytest.raises(AstroContextError) as exc_info:
            pipeline.build(QueryBundle(query_str="test"))

        # The exception message should reference the failed step name
        assert "crashing-step" in str(exc_info.value)

    def test_failed_step_name_in_diagnostics_build(self) -> None:
        """Verify the diagnostics dict captures the failed step name.

        The diagnostics dict is local to build() so we cannot directly inspect it
        after an exception. Instead, we verify the step name appears in the
        AstroContextError message (which is set from diagnostics['failed_step']).
        """
        pipeline = make_pipeline()
        pipeline.add_step(PipelineStep(name="step-a", fn=lambda items, q: items))
        pipeline.add_step(PipelineStep(name="step-b-will-fail", fn=_failing_step))

        with pytest.raises(AstroContextError, match="step-b-will-fail"):
            pipeline.build(QueryBundle(query_str="test"))

    @pytest.mark.asyncio
    async def test_failed_step_name_in_diagnostics_abuild(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_step(PipelineStep(name="ok-step", fn=lambda items, q: items))
        pipeline.add_step(
            PipelineStep(name="async-crasher", fn=_async_failing_step, is_async=True)
        )

        with pytest.raises(AstroContextError, match="async-crasher"):
            await pipeline.abuild(QueryBundle(query_str="test"))


# ---------------------------------------------------------------------------
# 9. abuild() step exception wrapping
# ---------------------------------------------------------------------------


class TestAsyncStepExceptionWrapping:
    """Same as #1 but for the async path (abuild)."""

    @pytest.mark.asyncio
    async def test_generic_exception_wrapped_in_anchor_error(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_step(
            PipelineStep(name="async-boom", fn=_async_failing_step, is_async=True)
        )

        with pytest.raises(AstroContextError, match="async-boom"):
            await pipeline.abuild(QueryBundle(query_str="test"))

    @pytest.mark.asyncio
    async def test_wrapped_exception_chains_original_cause(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_step(
            PipelineStep(name="async-boom", fn=_async_failing_step, is_async=True)
        )

        with pytest.raises(AstroContextError) as exc_info:
            await pipeline.abuild(QueryBundle(query_str="test"))

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "async boom" in str(exc_info.value.__cause__)

    @pytest.mark.asyncio
    async def test_sync_step_exception_wrapped_in_abuild(self) -> None:
        """A sync step that fails inside abuild() is also wrapped."""
        pipeline = make_pipeline()
        pipeline.add_step(PipelineStep(name="sync-in-async", fn=_failing_step))

        with pytest.raises(AstroContextError, match="sync-in-async"):
            await pipeline.abuild(QueryBundle(query_str="test"))


# ---------------------------------------------------------------------------
# 10. Pipeline max_tokens validation
# ---------------------------------------------------------------------------


class TestMaxTokensValidation:
    """ContextPipeline(max_tokens=0) and max_tokens=-1 should raise ValueError."""

    def test_max_tokens_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            ContextPipeline(max_tokens=0)

    def test_max_tokens_negative_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            ContextPipeline(max_tokens=-1)

    def test_max_tokens_positive_is_valid(self) -> None:
        pipeline = ContextPipeline(max_tokens=1, tokenizer=FakeTokenizer())
        assert pipeline.max_tokens == 1

    def test_max_tokens_large_is_valid(self) -> None:
        pipeline = ContextPipeline(max_tokens=1_000_000, tokenizer=FakeTokenizer())
        assert pipeline.max_tokens == 1_000_000


# ---------------------------------------------------------------------------
# 11. PipelineExecutionError wrapping with diagnostics
# ---------------------------------------------------------------------------


class TestPipelineExecutionErrorWrapping:
    """When an unknown exception type escapes to the pipeline loop,
    it is wrapped in PipelineExecutionError with diagnostics attached."""

    def test_unknown_exception_wrapped_in_pipeline_execution_error(self) -> None:
        """Force a bare RuntimeError past the step wrapper by patching execute.

        PipelineStep uses ``@dataclass(slots=True)`` so we patch the
        ``execute`` method on the *class* to raise a bare RuntimeError.
        This simulates an unexpected exception type that bypasses the
        step's internal AstroContextError wrapping and triggers the
        outer ``except Exception`` handler in ``build()``, which wraps
        the error in ``PipelineExecutionError`` with diagnostics attached.
        """
        pipeline = make_pipeline()

        step = PipelineStep(name="raw-error-step", fn=_failing_step, on_error="raise")
        pipeline.add_step(step)

        def patched_execute(
            self: PipelineStep,
            items: list[ContextItem],
            query: QueryBundle,
        ) -> list[ContextItem]:
            raise RuntimeError("unexpected")

        with (
            patch.object(PipelineStep, "execute", patched_execute),
            pytest.raises(PipelineExecutionError) as exc_info,
        ):
            pipeline.build(QueryBundle(query_str="test"))

        assert exc_info.value.diagnostics["failed_step"] == "raw-error-step"
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "unexpected" in str(exc_info.value.__cause__)

    def test_pipeline_execution_error_is_anchor_error(self) -> None:
        """PipelineExecutionError should be a subclass of AstroContextError."""
        assert issubclass(PipelineExecutionError, AstroContextError)

    def test_pipeline_execution_error_diagnostics_default_empty(self) -> None:
        """PipelineExecutionError with no diagnostics defaults to empty dict."""
        err = PipelineExecutionError("test message")
        assert err.diagnostics == {}

    def test_pipeline_execution_error_preserves_diagnostics(self) -> None:
        """PipelineExecutionError stores the diagnostics dict."""
        diag = {"failed_step": "my-step", "steps": []}
        err = PipelineExecutionError("test message", diagnostics=diag)
        assert err.diagnostics["failed_step"] == "my-step"
