"""Composable pipeline steps with sync and async support."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from anchor.exceptions import AstroContextError, RetrieverError
from anchor.models.context import ContextItem
from anchor.models.query import QueryBundle
from anchor.protocols.postprocessor import AsyncPostProcessor, PostProcessor
from anchor.protocols.query_transform import QueryTransformer
from anchor.protocols.reranker import AsyncReranker, Reranker
from anchor.protocols.retriever import AsyncRetriever, Retriever
from anchor.retrieval._rrf import rrf_fuse

SyncStepFn = Callable[[list[ContextItem], QueryBundle], list[ContextItem]]
AsyncStepFn = Callable[[list[ContextItem], QueryBundle], Awaitable[list[ContextItem]]]
StepFn = SyncStepFn | AsyncStepFn


@dataclass(slots=True)
class PipelineStep:
    """A single step in the context pipeline.

    Steps are composable and execute in order. Each step receives the
    current list of context items and the query, and returns a
    (potentially modified) list of context items.

    Supports both sync and async execution functions. If an async fn
    is provided, the step can only be executed via ``aexecute()``.
    """

    name: str
    fn: StepFn
    is_async: bool = False
    on_error: Literal["raise", "skip"] = "raise"
    metadata: dict[str, Any] = field(default_factory=dict)

    def _validate_result(self, result: Any) -> list[ContextItem]:
        """Validate that a step returned a list and return it typed."""
        if not isinstance(result, list):
            msg = f"Step '{self.name}' must return a list of ContextItem"
            raise TypeError(msg)
        return cast(list[ContextItem], result)

    def execute(self, items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        """Execute this step synchronously."""
        if self.is_async:
            msg = (
                f"Step '{self.name}' is async and cannot be called synchronously. "
                "Use pipeline.abuild() or step.aexecute() instead."
            )
            raise TypeError(msg)
        try:
            result = self.fn(items, query)
        except AstroContextError:
            raise
        except Exception as e:
            msg = f"Step '{self.name}' failed during execution"
            raise AstroContextError(msg) from e
        return self._validate_result(result)

    async def aexecute(self, items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        """Execute this step asynchronously.

        Works for both sync and async step functions -- sync functions
        are called directly (they are typically fast in-memory operations).
        """
        try:
            result = self.fn(items, query)
            if inspect.isawaitable(result):
                result = await result
        except AstroContextError:
            raise
        except Exception as e:
            msg = f"Step '{self.name}' failed during execution"
            raise AstroContextError(msg) from e
        return self._validate_result(result)


def retriever_step(name: str, retriever: Retriever, top_k: int = 10) -> PipelineStep:
    """Create a pipeline step from a Retriever protocol implementation."""

    def _retrieve(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        retrieved = retriever.retrieve(query, top_k=top_k)
        return items + retrieved

    return PipelineStep(name=name, fn=_retrieve)


def async_retriever_step(name: str, retriever: AsyncRetriever, top_k: int = 10) -> PipelineStep:
    """Create an async pipeline step from an AsyncRetriever implementation."""

    async def _aretrieve(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        retrieved = await retriever.aretrieve(query, top_k=top_k)
        return items + retrieved

    return PipelineStep(name=name, fn=_aretrieve, is_async=True)


def postprocessor_step(name: str, processor: PostProcessor) -> PipelineStep:
    """Create a pipeline step from a PostProcessor protocol implementation."""

    def _process(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        return processor.process(items, query)

    return PipelineStep(name=name, fn=_process)


def async_postprocessor_step(name: str, processor: AsyncPostProcessor) -> PipelineStep:
    """Create an async pipeline step from an AsyncPostProcessor implementation."""

    async def _aprocess(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        return await processor.aprocess(items, query)

    return PipelineStep(name=name, fn=_aprocess, is_async=True)


def reranker_step(
    name: str, reranker: Reranker, top_k: int | None = None
) -> PipelineStep:
    """Create a pipeline step from a Reranker protocol implementation.

    The reranker receives the current items and query, scores them,
    and returns the top-k most relevant items.

    Parameters:
        name: Human-readable name for the step.
        reranker: Any object implementing the Reranker protocol.
        top_k: Maximum number of items the reranker should return.
            ``None`` defers to the reranker's own configured ``top_k``.

    Returns:
        A ``PipelineStep`` that applies the reranker to the pipeline items.
    """

    def _rerank(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        return reranker.rerank(query, items, top_k=top_k)

    return PipelineStep(name=name, fn=_rerank)


def async_reranker_step(
    name: str, reranker: AsyncReranker, top_k: int | None = None
) -> PipelineStep:
    """Create an async pipeline step from an AsyncReranker implementation.

    Parameters:
        name: Human-readable name for the step.
        reranker: Any object implementing the AsyncReranker protocol.
        top_k: Maximum number of items the reranker should return.
            ``None`` defers to the reranker's own configured ``top_k``.

    Returns:
        An async ``PipelineStep`` that applies the reranker.
    """

    async def _arerank(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        return await reranker.arerank(query, items, top_k=top_k)

    return PipelineStep(name=name, fn=_arerank, is_async=True)


def filter_step(name: str, predicate: Callable[[ContextItem], bool]) -> PipelineStep:
    """Create a pipeline step that filters items by a predicate."""

    def _filter(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        return [item for item in items if predicate(item)]

    return PipelineStep(name=name, fn=_filter)


def query_transform_step(
    name: str,
    transformer: QueryTransformer,
    retriever: Retriever,
    top_k: int = 10,
) -> PipelineStep:
    """Create a pipeline step that transforms the query, retrieves for each variant, and merges.

    The transformer expands a single query into multiple queries (e.g. via
    HyDE, multi-query, decomposition).  Each expanded query is passed to the
    retriever, and the results are merged (deduplicated by item ID) into the
    existing items list.

    Parameters:
        name: A descriptive name for this pipeline step.
        transformer: A ``QueryTransformer`` to expand the query.
        retriever: A ``Retriever`` to run against each expanded query.
        top_k: Maximum items to retrieve per query variant.

    Returns:
        A ``PipelineStep`` that performs multi-query retrieval.
    """

    def _transform_and_retrieve(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        queries = transformer.transform(query)
        ranked_lists: list[list[ContextItem]] = []
        for q in queries:
            ranked_lists.append(retriever.retrieve(q, top_k=top_k))
        fused = rrf_fuse(ranked_lists, top_k=top_k)
        existing_ids: set[str] = {item.id for item in items}
        new_items = [item for item in fused if item.id not in existing_ids]
        return items + new_items

    return PipelineStep(name=name, fn=_transform_and_retrieve)


def classified_retriever_step(
    name: str,
    classifier: Any,  # QueryClassifier protocol
    retrievers: dict[str, Retriever],
    default: str | None = None,
    top_k: int = 10,
) -> PipelineStep:
    """Create a pipeline step that classifies the query and routes to a retriever.

    The classifier assigns a label to the query, then the corresponding
    retriever is looked up from the ``retrievers`` dict.  If the label
    is not found, the ``default`` key is used as a fallback.

    Parameters:
        name: Human-readable name for this pipeline step.
        classifier: Any object implementing the ``QueryClassifier`` protocol
            (must have a ``classify(QueryBundle) -> str`` method).
        retrievers: A mapping from class label to ``Retriever``.
        default: Optional fallback key to use when the classified label
            is not in ``retrievers``.  If ``None`` and the label is
            missing, a ``RetrieverError`` is raised.
        top_k: Maximum number of items to retrieve.

    Returns:
        A ``PipelineStep`` that classifies and retrieves.

    Raises:
        RetrieverError: If the classified label has no matching retriever
            and no default is configured.
    """

    def _classify_and_retrieve(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        label = classifier.classify(query)
        retriever = retrievers.get(label)
        if retriever is None and default is not None:
            retriever = retrievers.get(default)
        if retriever is None:
            msg = (
                f"No retriever found for class label {label!r} "
                f"and no default configured in step {name!r}"
            )
            raise RetrieverError(msg)
        retrieved = retriever.retrieve(query, top_k=top_k)
        return items + retrieved

    return PipelineStep(name=name, fn=_classify_and_retrieve)
