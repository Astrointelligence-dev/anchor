"""Tests for A/B testing framework."""

from __future__ import annotations

import pytest

from anchor.evaluation.ab_testing import (
    ABTestResult,
    ABTestRunner,
    AggregatedMetrics,
    EvaluationDataset,
    EvaluationSample,
    _normal_cdf,
    _t_test_paired,
)
from anchor.evaluation.evaluator import PipelineEvaluator
from anchor.models.context import ContextItem, SourceType
from anchor.models.query import QueryBundle


def _item(doc_id: str) -> ContextItem:
    """Create a minimal ContextItem with a given ID."""
    return ContextItem(id=doc_id, content=f"doc {doc_id}", source=SourceType.RETRIEVAL)


class _FakeRetriever:
    """A deterministic retriever for testing."""

    def __init__(self, items: list[ContextItem]) -> None:
        self._items = items

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        return self._items[:top_k]


class _QueryAwareRetriever:
    """A retriever that returns different items depending on the query."""

    def __init__(self, mapping: dict[str, list[ContextItem]]) -> None:
        self._mapping = mapping

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        return self._mapping.get(query.query_str, [])[:top_k]


class TestABTestResult:
    """Tests for ABTestResult model."""

    def test_ab_test_result_frozen(self) -> None:
        agg = AggregatedMetrics()
        result = ABTestResult(
            metrics_a=agg,
            metrics_b=agg,
            winner="tie",
            p_value=1.0,
            is_significant=False,
        )
        with pytest.raises(Exception):  # noqa: B017
            result.winner = "a"  # type: ignore[misc]


class TestABTestRunner:
    """Tests for ABTestRunner execution."""

    def _make_dataset(self, queries_and_relevant: list[tuple[str, list[str]]]) -> EvaluationDataset:
        samples = [EvaluationSample(query=q, relevant_ids=r) for q, r in queries_and_relevant]
        return EvaluationDataset(samples=samples)

    def test_ab_runner_a_wins(self) -> None:
        """Retriever A returns all relevant items; B returns none."""
        dataset = self._make_dataset(
            [
                ("q1", ["a", "b"]),
                ("q2", ["c", "d"]),
                ("q3", ["e", "f"]),
                ("q4", ["g", "h"]),
                ("q5", ["i", "j"]),
            ]
        )
        # A returns perfect results
        retriever_a = _FakeRetriever(
            [
                _item("a"),
                _item("b"),
                _item("c"),
                _item("d"),
                _item("e"),
                _item("f"),
                _item("g"),
                _item("h"),
                _item("i"),
                _item("j"),
            ]
        )
        # B returns irrelevant items
        retriever_b = _FakeRetriever([_item("x"), _item("y"), _item("z")])

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=10)

        assert result.winner == "a"
        assert result.is_significant
        assert result.metrics_a.mean_precision > result.metrics_b.mean_precision

    def test_ab_runner_b_wins(self) -> None:
        """Retriever B returns all relevant items; A returns none."""
        dataset = self._make_dataset(
            [
                ("q1", ["a", "b"]),
                ("q2", ["c", "d"]),
                ("q3", ["e", "f"]),
                ("q4", ["g", "h"]),
                ("q5", ["i", "j"]),
            ]
        )
        retriever_a = _FakeRetriever([_item("x"), _item("y")])
        retriever_b = _FakeRetriever(
            [
                _item("a"),
                _item("b"),
                _item("c"),
                _item("d"),
                _item("e"),
                _item("f"),
                _item("g"),
                _item("h"),
                _item("i"),
                _item("j"),
            ]
        )

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=10)

        assert result.winner == "b"
        assert result.is_significant

    def test_ab_runner_tie(self) -> None:
        """Both retrievers return the same items -> tie."""
        dataset = self._make_dataset(
            [
                ("q1", ["a"]),
                ("q2", ["b"]),
            ]
        )
        same_items = [_item("a"), _item("b")]
        retriever_a = _FakeRetriever(same_items)
        retriever_b = _FakeRetriever(same_items)

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=10)

        assert result.winner == "tie"
        assert not result.is_significant

    def test_statistical_significance(self) -> None:
        """Clear winner should have p_value < 0.05."""
        dataset = self._make_dataset([(f"q{i}", [f"doc{i}"]) for i in range(20)])
        # A always retrieves the relevant doc
        good_items = [_item(f"doc{i}") for i in range(20)]
        retriever_a = _FakeRetriever(good_items)
        # B never retrieves relevant docs
        retriever_b = _FakeRetriever([_item("irrelevant")])

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=10)

        assert result.p_value < 0.05
        assert result.is_significant

    def test_not_significant(self) -> None:
        """Similar performance should yield p > 0.05."""
        dataset = self._make_dataset(
            [
                ("q1", ["a"]),
                ("q2", ["b"]),
            ]
        )
        # Both retrievers return the same results
        retriever_a = _FakeRetriever([_item("a"), _item("b")])
        retriever_b = _FakeRetriever([_item("a"), _item("b")])

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=10)

        assert result.p_value >= 0.05 or result.p_value == 1.0
        assert not result.is_significant

    def test_per_metric_comparison(self) -> None:
        """per_metric_comparison dict should have expected keys."""
        dataset = self._make_dataset([("q1", ["a"])])
        retriever_a = _FakeRetriever([_item("a")])
        retriever_b = _FakeRetriever([_item("x")])

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=10)

        expected_keys = {"precision", "recall", "f1", "mrr", "ndcg"}
        assert set(result.per_metric_comparison.keys()) == expected_keys
        for key in expected_keys:
            assert "a" in result.per_metric_comparison[key]
            assert "b" in result.per_metric_comparison[key]
            assert "delta" in result.per_metric_comparison[key]

    def test_single_sample(self) -> None:
        """Works with a single sample; p_value should be 1.0."""
        dataset = self._make_dataset([("q1", ["a"])])
        retriever_a = _FakeRetriever([_item("a")])
        retriever_b = _FakeRetriever([_item("x")])

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=10)

        # With a single sample, the t-test returns 1.0
        assert result.p_value == 1.0
        assert result.winner == "tie"

    def test_empty_dataset(self) -> None:
        """Empty dataset should produce a tie with p=1.0."""
        dataset = EvaluationDataset(samples=[])
        retriever_a = _FakeRetriever([])
        retriever_b = _FakeRetriever([])

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=10)

        assert result.winner == "tie"
        assert result.p_value == 1.0
        assert not result.is_significant
        assert result.metrics_a.num_samples == 0
        assert result.metrics_b.num_samples == 0

    def test_repr(self) -> None:
        dataset = self._make_dataset([("q1", ["a"]), ("q2", ["b"])])
        runner = ABTestRunner(PipelineEvaluator(), dataset)
        r = repr(runner)
        assert "ABTestRunner" in r
        assert "dataset_size=2" in r

    # ------------------------------------------------------------------
    # Additional targeted tests (Batch E verification)
    # ------------------------------------------------------------------

    def test_identical_retrievers_p_value_one(self) -> None:
        """Two identical retrievers must produce p_value=1.0 (no difference)."""
        dataset = self._make_dataset([(f"q{i}", [f"doc{i}"]) for i in range(10)])
        items = [_item(f"doc{i}") for i in range(10)]
        retriever_a = _FakeRetriever(items)
        retriever_b = _FakeRetriever(items)

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=10)

        assert result.p_value == 1.0
        assert result.winner == "tie"
        assert not result.is_significant

    def test_large_sample_significance_detection(self) -> None:
        """With 50+ samples and clear difference, significance must be detected."""
        n = 60
        dataset = self._make_dataset([(f"q{i}", [f"doc{i}"]) for i in range(n)])
        # A returns correct doc for each query
        mapping_a = {f"q{i}": [_item(f"doc{i}")] for i in range(n)}
        retriever_a = _QueryAwareRetriever(mapping_a)
        # B always returns wrong doc
        retriever_b = _FakeRetriever([_item("wrong")])

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=10)

        assert result.p_value < 0.001  # Very significant with 60 samples
        assert result.is_significant
        assert result.winner == "a"
        assert result.metrics_a.num_samples == n
        assert result.metrics_b.num_samples == n

    def test_both_retrievers_return_nothing(self) -> None:
        """Both retrievers return empty results -> tie with all-zero metrics."""
        dataset = self._make_dataset([("q1", ["a"]), ("q2", ["b"]), ("q3", ["c"])])
        retriever_a = _FakeRetriever([])
        retriever_b = _FakeRetriever([])

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=10)

        assert result.winner == "tie"
        assert result.metrics_a.mean_precision == 0.0
        assert result.metrics_b.mean_precision == 0.0
        assert result.metrics_a.mean_recall == 0.0
        assert result.metrics_b.mean_recall == 0.0

    def test_asymmetric_performance(self) -> None:
        """A gets 100% precision; B gets 0% -- winner must be A."""
        n = 10
        dataset = self._make_dataset([(f"q{i}", [f"doc{i}"]) for i in range(n)])
        # A returns exactly the relevant doc per query
        mapping_a = {f"q{i}": [_item(f"doc{i}")] for i in range(n)}
        retriever_a = _QueryAwareRetriever(mapping_a)
        # B returns only irrelevant
        retriever_b = _FakeRetriever([_item("irrelevant_x")])

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=1)

        assert result.metrics_a.mean_precision == pytest.approx(1.0)
        assert result.metrics_b.mean_precision == pytest.approx(0.0)
        assert result.winner == "a"

    def test_per_metric_comparison_delta_values(self) -> None:
        """Verify delta = a - b for every metric in per_metric_comparison."""
        dataset = self._make_dataset([("q1", ["a"]), ("q2", ["b"])])
        retriever_a = _FakeRetriever([_item("a"), _item("b")])
        retriever_b = _FakeRetriever([_item("x")])

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=10)

        for metric_name, values in result.per_metric_comparison.items():
            assert {"a", "b", "delta"} == set(values.keys()), f"Missing keys in {metric_name}"
            assert values["delta"] == pytest.approx(values["a"] - values["b"]), (
                f"Delta mismatch for {metric_name}"
            )

    def test_ab_test_result_metadata_passthrough(self) -> None:
        """Metadata on ABTestResult should be stored and accessible."""
        agg = AggregatedMetrics()
        meta = {"experiment": "v2", "run_id": 42}
        result = ABTestResult(
            metrics_a=agg,
            metrics_b=agg,
            winner="tie",
            p_value=1.0,
            is_significant=False,
            metadata=meta,
        )
        assert result.metadata == meta
        assert result.metadata["experiment"] == "v2"
        assert result.metadata["run_id"] == 42

    def test_evaluation_sample_metadata(self) -> None:
        """EvaluationSample should carry metadata through."""
        sample = EvaluationSample(
            query="q1",
            relevant_ids=["a"],
            metadata={"source": "manual"},
        )
        assert sample.metadata["source"] == "manual"

    def test_evaluation_dataset_name_and_metadata(self) -> None:
        """EvaluationDataset should carry name and metadata."""
        ds = EvaluationDataset(
            samples=[EvaluationSample(query="q1", relevant_ids=["a"])],
            name="test-set-v1",
            metadata={"version": 1},
        )
        assert ds.name == "test-set-v1"
        assert ds.metadata["version"] == 1


class TestStatisticalHelpers:
    """Tests for _t_test_paired and _normal_cdf edge cases."""

    def test_normal_cdf_zero(self) -> None:
        """CDF at 0 should be 0.5."""
        assert _normal_cdf(0.0) == pytest.approx(0.5)

    def test_normal_cdf_large_positive(self) -> None:
        """CDF at large positive z should approach 1.0."""
        assert _normal_cdf(6.0) == pytest.approx(1.0, abs=1e-6)

    def test_normal_cdf_large_negative(self) -> None:
        """CDF at large negative z should approach 0.0."""
        assert _normal_cdf(-6.0) == pytest.approx(0.0, abs=1e-6)

    def test_t_test_identical_values(self) -> None:
        """Identical value lists -> p=1.0 (no difference)."""
        vals = [0.5, 0.6, 0.7, 0.8, 0.9]
        assert _t_test_paired(vals, vals) == 1.0

    def test_t_test_single_sample(self) -> None:
        """Single sample -> p=1.0 (insufficient data)."""
        assert _t_test_paired([1.0], [0.0]) == 1.0

    def test_t_test_constant_difference(self) -> None:
        """Constant non-zero difference (std=0) -> p=0.0."""
        a = [1.0, 1.0, 1.0, 1.0, 1.0]
        b = [0.0, 0.0, 0.0, 0.0, 0.0]
        assert _t_test_paired(a, b) == 0.0

    def test_t_test_symmetric(self) -> None:
        """t_test_paired(a, b) should equal t_test_paired(b, a) (two-sided)."""
        a = [0.8, 0.7, 0.9, 0.6, 0.85]
        b = [0.3, 0.4, 0.2, 0.5, 0.35]
        assert _t_test_paired(a, b) == pytest.approx(_t_test_paired(b, a))


class TestABTestRunnerIntegration:
    """Integration tests: ABTestResult compatibility with evaluation models."""

    def test_aggregated_metrics_fields_match_retrieval_metrics(self) -> None:
        """AggregatedMetrics field names should correspond to RetrievalMetrics means."""
        agg = AggregatedMetrics(
            mean_precision=0.8,
            mean_recall=0.7,
            mean_f1=0.75,
            mean_mrr=0.9,
            mean_ndcg=0.85,
            num_samples=10,
        )
        # Verify all fields are accessible and frozen
        assert agg.mean_precision == 0.8
        assert agg.num_samples == 10
        with pytest.raises(Exception):  # noqa: B017
            agg.mean_precision = 0.5  # type: ignore[misc]

    def test_human_to_dataset_through_ab_runner(self) -> None:
        """Human judgments -> EvaluationDataset -> ABTestRunner end-to-end."""
        from anchor.evaluation.human import HumanEvaluationCollector, HumanJudgment

        collector = HumanEvaluationCollector()
        # Create judgments for 5 queries, each with one relevant doc
        for i in range(5):
            collector.add_judgment(
                HumanJudgment(
                    query=f"q{i}",
                    item_id=f"doc{i}",
                    relevance=3,
                    annotator="alice",
                )
            )
            # Add some irrelevant docs too
            collector.add_judgment(
                HumanJudgment(
                    query=f"q{i}",
                    item_id=f"noise{i}",
                    relevance=0,
                    annotator="alice",
                )
            )

        dataset = collector.to_dataset(threshold=2)
        assert len(dataset.samples) == 5

        # A retrieves relevant docs, B retrieves noise
        mapping_a = {f"q{i}": [_item(f"doc{i}")] for i in range(5)}
        retriever_a = _QueryAwareRetriever(mapping_a)
        retriever_b = _FakeRetriever([_item("wrong")])

        runner = ABTestRunner(PipelineEvaluator(), dataset)
        result = runner.run(retriever_a, retriever_b, k=5)

        assert result.winner == "a"
        assert result.metrics_a.mean_precision > 0.0
        assert result.metrics_b.mean_precision == 0.0


class TestABScopeForwarding:
    def test_scope_reaches_both_retrievers(self) -> None:
        from anchor.evaluation import (
            ABTestRunner,
            EvaluationDataset,
            EvaluationSample,
            PipelineEvaluator,
        )
        from anchor.models.scope import RetrievalScope

        seen: list[object] = []

        class Spy:
            def retrieve(self, query, top_k=10, **kwargs):
                seen.append(kwargs.get("scope", "absent"))
                return []

        dataset = EvaluationDataset(samples=[EvaluationSample(query="q", relevant_ids=["a"])])
        runner = ABTestRunner(PipelineEvaluator(), dataset)
        runner.run(Spy(), Spy(), k=3)
        scope = RetrievalScope(include=("/y",))
        runner.run(Spy(), Spy(), k=3, scope=scope)
        assert seen == ["absent", "absent", scope, scope]
