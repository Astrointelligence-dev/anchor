# Exceptions Reference

All exception classes in anchor inherit from `AstroContextError`. Import
them directly from the top-level package:

```python
from anchor import (
    AstroContextError,
    PipelineExecutionError,
    RetrieverError,
    StorageError,
    FormatterError,
    IngestionError,
)
```

## Hierarchy

```
Exception
 └── AstroContextError
      ├── PipelineExecutionError
      ├── RetrieverError
      ├── StorageError
      ├── FormatterError
      └── IngestionError
```

## `AstroContextError`

Base exception for all anchor errors. Catch this to handle any
library-level error in a single `except` clause.

## `PipelineExecutionError`

Raised when the pipeline fails at a step with `on_error="raise"`. Carries
partial diagnostics so you can inspect what happened before the failure.

```python
class PipelineExecutionError(AstroContextError):
    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None) -> None:
        ...
    diagnostics: dict[str, Any]
```

| Attribute | Type | Description |
|---|---|---|
| `diagnostics` | `dict[str, Any]` | Partial pipeline diagnostics collected before the error. |

### Example

```python
from anchor import ContextPipeline, PipelineExecutionError

try:
    result = pipeline.build("test")
except PipelineExecutionError as e:
    print(f"Failed at step: {e.diagnostics.get('failed_step')}")
    print(f"Steps completed: {e.diagnostics.get('steps', [])}")
```

## `RetrieverError`

Raised when a retriever encounters an error. For example,
`classified_retriever_step` raises this when the classified label has no
matching retriever and no default is configured.

## `StorageError`

Raised when a storage backend (context store, document store, vector store)
encounters an error.

## `FormatterError`

Raised when formatting context fails. The pipeline wraps formatter exceptions
in this class.

## `IngestionError`

Raised when document ingestion (parsing or chunking) fails.

---

## See Also

- [Pipeline Guide](../guides/pipeline.md) -- error handling patterns
- [Pipeline API Reference](pipeline.md) -- `on_error` parameter on steps
