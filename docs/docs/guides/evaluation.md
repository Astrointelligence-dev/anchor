# Evaluation

Measuring the quality of your RAG pipeline is essential for iterating with
confidence. anchor ships an evaluation framework that covers three
layers:

1. **Retrieval metrics** -- precision, recall, MRR, NDCG, and more.
2. **RAG quality** -- LLM-judged faithfulness, relevancy, and context quality.
3. **Orchestration** -- combine both into a single `PipelineEvaluator` result.

All evaluators work offline (no network calls) except `LLMRAGEvaluator`, which
delegates to user-supplied callback functions.

---

## Retrieval Metrics

`RetrievalMetricsCalculator` computes standard IR metrics given a ranked list
of retrieved items and a set of ground-truth relevant IDs.

```python
from anchor.evaluation import RetrievalMetricsCalculator
from anchor.models.context import ContextItem, SourceType

# Create some retrieved items (ranked order matters)
retrieved = [
    ContextItem(id="doc-1", content="Python lists", source=SourceType.RETRIEVAL),
    ContextItem(id="doc-3", content="Sort keys", source=SourceType.RETRIEVAL),
    ContextItem(id="doc-7", content="Lambda usage", source=SourceType.RETRIEVAL),
]

# Ground-truth relevant IDs
relevant = ["doc-1", "doc-3", "doc-5"]

calc = RetrievalMetricsCalculator(k=10)
metrics = calc.evaluate(retrieved, relevant, k=3)

print(f"Precision@3: {metrics.precision_at_k:.2f}")  # 0.67
print(f"Recall@3:    {metrics.recall_at_k:.2f}")     # 0.67
print(f"F1@3:        {metrics.f1_at_k:.2f}")         # 0.67
print(f"MRR:         {metrics.mrr:.2f}")             # 1.00
print(f"NDCG:        {metrics.ndcg:.2f}")            # 0.77
print(f"Hit rate:    {metrics.hit_rate:.0f}")         # 1
```

### Available metrics

| Metric | Description |
|--------|-------------|
| `precision_at_k` | Fraction of retrieved items that are relevant |
| `recall_at_k` | Fraction of relevant items that were retrieved |
| `f1_at_k` | Harmonic mean of precision and recall |
| `mrr` | Reciprocal of the rank of the first relevant item |
| `ndcg` | Normalized Discounted Cumulative Gain (binary relevance) |
| `hit_rate` | 1.0 if at least one relevant item was retrieved |

!!! tip
    Set `k` at instantiation for a project-wide default, then override per-call
    when needed: `calc.evaluate(retrieved, relevant, k=5)`.

---

## RAG Quality with LLMRAGEvaluator

`LLMRAGEvaluator` provides RAGAS-style evaluation driven entirely by callback
functions. Each callback computes a single metric dimension and returns a float
in `[0.0, 1.0]`. This design keeps the evaluation logic free of any specific
LLM SDK.

```python
from anchor.evaluation import LLMRAGEvaluator

# Define scoring callbacks (replace with real LLM-based logic)
def faithfulness_fn(answer: str, contexts: list[str]) -> float:
    """Check whether the answer overlaps with the contexts."""
    # Simplified check: do any context words appear in the answer?
    ctx_words = {w.lower() for c in contexts for w in c.split()}
    answer_words = {w.lower() for w in answer.split()}
    overlap = len(ctx_words & answer_words) / max(len(ctx_words), 1)
    return min(overlap, 1.0)

def relevancy_fn(query: str, answer: str) -> float:
    """Check whether the answer addresses the query."""
    return 0.85

evaluator = LLMRAGEvaluator(
    faithfulness_fn=faithfulness_fn,
    relevancy_fn=relevancy_fn,
)

rag_metrics = evaluator.evaluate(
    query="How do I sort a list?",
    answer="Use sorted() or list.sort() in Python.",
    contexts=["Python lists support .sort() and sorted()."],
)

print(f"Faithfulness:      {rag_metrics.faithfulness}")
print(f"Answer relevancy:  {rag_metrics.answer_relevancy}")
print(f"Context precision: {rag_metrics.context_precision}")
print(f"Context recall:    {rag_metrics.context_recall}")
```

### Callback signatures

| Callback | Signature | Description |
|----------|-----------|-------------|
| `faithfulness_fn` | `(answer, contexts) -> float` | Grounding in context |
| `relevancy_fn` | `(query, answer) -> float` | Relevance to the query |
| `precision_fn` | `(query, contexts) -> float` | Context precision |
| `recall_fn` | `(query, contexts, ground_truth) -> float` | Context recall |

!!! note
    Dimensions without a registered callback default to `0.0`.

---

## PipelineEvaluator

`PipelineEvaluator` orchestrates retrieval **and** RAG evaluation into a
single `EvaluationResult`.

```python
from anchor.evaluation import (
    PipelineEvaluator,
    RetrievalMetricsCalculator,
    LLMRAGEvaluator,
)
from anchor.models.context import ContextItem, SourceType

calc = RetrievalMetricsCalculator(k=10)
rag_eval = LLMRAGEvaluator(
    faithfulness_fn=lambda ans, ctx: 0.9,
    relevancy_fn=lambda q, a: 0.85,
)

evaluator = PipelineEvaluator(
    retrieval_calculator=calc,
    rag_evaluator=rag_eval,
)

retrieved = [
    ContextItem(id="d1", content="sorting", source=SourceType.RETRIEVAL),
]
result = evaluator.evaluate(
    query="How to sort?",
    answer="Use sorted().",
    retrieved=retrieved,
    relevant=["d1", "d2"],
    contexts=["sorting"],
    k=5,
)

print(f"Precision@5: {result.retrieval_metrics.precision_at_k:.2f}")
print(f"Faithfulness: {result.rag_metrics.faithfulness:.2f}")
```

You can also call `evaluate_retrieval()` and `evaluate_rag()` independently:

```python
# Retrieval only
r_metrics = evaluator.evaluate_retrieval(retrieved, relevant=["d1"], k=3)

# RAG only (requires rag_evaluator)
rag_metrics = evaluator.evaluate_rag(
    query="How to sort?",
    answer="Use sorted().",
    contexts=["sorting"],
)
```

!!! warning
    Calling `evaluate_rag()` raises `ValueError` if no `rag_evaluator` was
    configured.

---

## A/B Testing

`ABTestRunner` compares two retrievers on a shared evaluation dataset using a
paired t-test on precision@k.

### Building a dataset

```python
from anchor.evaluation import EvaluationSample, EvaluationDataset

samples = [
    EvaluationSample(query="sort a list", relevant_ids=["d1", "d2"]),
    EvaluationSample(query="read a file", relevant_ids=["d3"]),
    EvaluationSample(query="parse JSON", relevant_ids=["d4", "d5"]),
]

dataset = EvaluationDataset(samples=samples, name="search-quality-v1")
```

### Running the test

```python
from anchor.evaluation import ABTestRunner, PipelineEvaluator

evaluator = PipelineEvaluator()
runner = ABTestRunner(evaluator=evaluator, dataset=dataset)

result = runner.run(
    retriever_a=my_dense_retriever,
    retriever_b=my_hybrid_retriever,
    k=10,
    significance_level=0.05,
)

print(f"Winner: {result.winner}")              # "a", "b", or "tie"
print(f"p-value: {result.p_value:.4f}")
print(f"Significant: {result.is_significant}")
print(f"A precision: {result.metrics_a.mean_precision:.3f}")
print(f"B precision: {result.metrics_b.mean_precision:.3f}")

# Per-metric comparison
for metric, data in result.per_metric_comparison.items():
    print(f"  {metric}: A={data['a']:.3f} B={data['b']:.3f} delta={data['delta']:+.3f}")
```

!!! tip
    The t-test uses a normal-distribution approximation (no scipy dependency).
    For small sample sizes (< 30), results are conservative.

---

## Batch Evaluation

`BatchEvaluator` runs evaluation over an entire dataset with a real retriever,
producing aggregated statistics including percentiles and per-sample results.

```python
from anchor.evaluation.batch import (
    BatchEvaluator,
    EvaluationDataset,
    EvaluationSample,
)
from anchor.evaluation import PipelineEvaluator

dataset = EvaluationDataset(
    name="regression-suite",
    samples=[
        EvaluationSample(query="sort a list", expected_ids=["d1"]),
        EvaluationSample(query="read CSV", expected_ids=["d2", "d3"]),
    ],
)

batch = BatchEvaluator(
    evaluator=PipelineEvaluator(),
    retriever=my_retriever,
    top_k=10,
)

agg = batch.evaluate(dataset, k=10)

print(f"Samples:        {agg.count}")
print(f"Mean precision: {agg.mean_precision:.3f}")
print(f"Mean recall:    {agg.mean_recall:.3f}")
print(f"Mean MRR:       {agg.mean_mrr:.3f}")
print(f"Mean NDCG:      {agg.mean_ndcg:.3f}")
print(f"P95 precision:  {agg.p95_precision:.3f}")
print(f"Min recall:     {agg.min_recall:.3f}")
```

!!! note
    `BatchEvaluator` uses `EvaluationSample` from `anchor.evaluation.batch`,
    which includes `expected_ids`, `ground_truth_answer`, and `contexts` fields.
    The A/B testing module uses its own `EvaluationSample` with `relevant_ids`.

---

## Human Evaluation

`HumanEvaluationCollector` gathers human relevance judgments and computes
inter-annotator agreement via Cohen's kappa.

### Collecting judgments

```python
from anchor.evaluation import HumanJudgment, HumanEvaluationCollector

collector = HumanEvaluationCollector()

# Annotator 1 rates query-document pairs
collector.add_judgments([
    HumanJudgment(query="sort list", item_id="d1", relevance=3, annotator="alice"),
    HumanJudgment(query="sort list", item_id="d2", relevance=1, annotator="alice"),
    HumanJudgment(query="read file", item_id="d3", relevance=2, annotator="alice"),
])

# Annotator 2 rates the same pairs
collector.add_judgments([
    HumanJudgment(query="sort list", item_id="d1", relevance=3, annotator="bob"),
    HumanJudgment(query="sort list", item_id="d2", relevance=0, annotator="bob"),
    HumanJudgment(query="read file", item_id="d3", relevance=2, annotator="bob"),
])
```

### Computing agreement

```python
kappa = collector.compute_agreement()
print(f"Cohen's kappa: {kappa:.2f}")
```

### Converting to a dataset

Judgments can be converted into an `EvaluationDataset` for use with
`ABTestRunner`. Items with a mean relevance rating at or above the threshold
are considered relevant.

```python
dataset = collector.to_dataset(threshold=2)
print(f"Queries: {len(dataset.samples)}")
for sample in dataset.samples:
    print(f"  {sample.query}: {sample.relevant_ids}")
```

### Summary metrics

```python
metrics = collector.compute_metrics()
print(f"Mean relevance:  {metrics['mean_relevance']:.2f}")
print(f"Agreement:       {metrics['agreement']:.2f}")
print(f"Num judgments:   {int(metrics['num_judgments'])}")
print(f"Num annotators:  {int(metrics['num_annotators'])}")
print(f"Num queries:     {int(metrics['num_queries'])}")
```

!!! tip
    Use `HumanEvaluationCollector` to bootstrap gold-standard evaluation sets,
    then run automated evaluation with `ABTestRunner` or `BatchEvaluator`.

---

## Consolidation Golden Set

Retrieval golden sets ask "did the right chunk come back?". The consolidation
golden set asks "after the conversation, is the memory store right?" — the
question behind `LLMExtractor` + `LLMConsolidator` (see the
[memory guide](memory.md)). Each case seeds a store, replays a conversation
through `MemoryManager.remember()`, and checks the result with substrings: what
must stay live, what must not, how many entries remain, and what a probe
search must or must not return.

```python
from anchor.evaluation import evaluate_consolidator, load_consolidation_set, assert_metric_floor
from anchor import LLMConsolidator, LLMExtractor

cases = load_consolidation_set("tests/fixtures/consolidation_golden.jsonl")
report = evaluate_consolidator(LLMExtractor(llm), LLMConsolidator(llm), cases)
print(report.summary())     # {"passed": ..., "state_ok": ..., "size_ok": ..., "probes_ok": ...}
print(report.failures())    # case names to look at
assert_metric_floor(report, "passed", 0.8)
```

A case in JSONL (`turns` may also be a list of lists — steps remembered one
at a time, for "moved to Rio" then "moved back"):

```json
{"name": "move_city", "existing": ["User lives in São Paulo"],
 "turns": [{"role": "user", "content": "Me mudei pro Rio de Janeiro mês passado."}],
 "live_contains": ["Rio"], "live_not_contains": ["São Paulo"], "expected_live": 1,
 "probes": [{"query": "São Paulo", "must_not_hit": ["São Paulo"]}]}
```

The shipped set has 48 cases in 10 scenarios. `SimilarityConsolidator` and
the "add everything" baseline pass ~27–35% of it — by design, since neither
can see a contradiction — which is the floor the LLM pair has to beat.

## Putting It All Together

A typical evaluation workflow combines human-labeled data, batch evaluation,
and A/B testing:

```python
from anchor.evaluation import (
    ABTestRunner,
    HumanEvaluationCollector,
    HumanJudgment,
    PipelineEvaluator,
)
from anchor.evaluation.batch import BatchEvaluator

# 1. Collect human judgments
collector = HumanEvaluationCollector()
collector.add_judgments([
    HumanJudgment(query="sort", item_id="d1", relevance=3, annotator="alice"),
    HumanJudgment(query="sort", item_id="d1", relevance=3, annotator="bob"),
])

# 2. Convert to evaluation dataset
dataset = collector.to_dataset(threshold=2)

# 3. A/B test two retrievers
evaluator = PipelineEvaluator()
runner = ABTestRunner(evaluator=evaluator, dataset=dataset)
result = runner.run(retriever_a, retriever_b, k=10)
print(f"Winner: {result.winner} (p={result.p_value:.4f})")
```

---

## Next Steps

- [Observability Guide](../guides/observability.md) -- trace and monitor pipeline execution
- [Evaluation API Reference](../api/evaluation.md) -- full class and method signatures
- [Pipeline Guide](../guides/pipeline.md) -- build and configure context pipelines
