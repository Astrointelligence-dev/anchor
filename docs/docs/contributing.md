# Contributing to anchor

We welcome contributions to anchor! This guide explains how to set up your development environment, follow our code conventions, and submit changes.

## Development Setup

### Prerequisites
- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) package manager

### Getting Started

```bash
git clone https://github.com/Astrointelligence-dev/anchor.git
cd anchor
uv sync
```

This installs all dependencies, including development tools.

### Running Tests

```bash
uv run pytest
```

This runs the full test suite (1,988 tests with ~94% coverage). To run specific tests:

```bash
uv run pytest tests/test_retrieval/  # Run tests in a module
uv run pytest tests/test_retrieval/test_rerankers.py::test_name  # Run specific test
```

### Code Quality

Check your code with Ruff:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/  # Auto-format code
```

## Code Conventions

### Language and Tools
- **Python**: 3.11+ with full type hints
- **Models**: Pydantic v2 (all models use `frozen=True` by default)
- **Extension Points**: PEP 544 Protocols for all interfaces
- **Linting**: Ruff with line-length of 100 characters
- **Type Checking**: mypy with strict mode enabled

### Requirements for Every File

Add this import at the top of every Python file:

```python
from __future__ import annotations
```

Always include complete type hints:

```python
def process_query(query: str, limit: int = 10) -> list[ContextItem]:
    """Process a query and return ranked context items."""
    ...
```

### Pydantic Models

```python
from pydantic import BaseModel, Field

class MyModel(BaseModel, frozen=True):  # frozen=True is the default
    name: str = Field(..., description="The name")
    score: float = Field(default=0.0, ge=0.0, le=1.0)
```

### Protocols for Extension Points

Define extension points using Protocol:

```python
from typing import Protocol

class MyCustomInterface(Protocol):
    """Protocol for implementing custom behavior."""

    def process(self, data: str) -> str:
        """Process data and return a result."""
        ...
```

## Testing

### Coverage Requirements
- Minimum 80% code coverage (94% for new features)
- Every new class must have tests
- Never make real API calls in tests (mock external services)

### Test File Organization

```
tests/
├── test_{module}/
│   ├── conftest.py          # Shared fixtures
│   ├── test_{file}.py       # Test one module
│   └── test_{file}_async.py # Async tests
```

### Example Test

```python
from unittest.mock import Mock, patch
import pytest

def test_retriever_returns_items():
    """Test that retriever returns expected items."""
    retriever = DenseRetriever(...)
    items = retriever.retrieve("query")
    assert len(items) > 0
    assert all(isinstance(item, ContextItem) for item in items)
```

## Adding a New Module

Follow this checklist:

1. **Define Protocol** (if it's an extension point)
   - Create `src/anchor/protocols/{name}.py`
   - Define the protocol interface

2. **Implement Core**
   - Create implementation in appropriate subpackage
   - Use type hints everywhere
   - Add docstrings

3. **Export Properly**
   - Add to module `__init__.py`
   - Add to main `src/anchor/__init__.py`
   - Update `__all__`

4. **Add Tests**
   - Create `tests/test_{module}/` directory
   - Achieve 94% coverage for new code

5. **Document**
   - Add guide: `docs/docs/guides/{name}.md`
   - Add API reference: `docs/docs/api/{name}.md`
   - Update main navigation in `docs/mkdocs.yml`

## Pull Request Guidelines

### Before Submitting
- [ ] Tests pass: `uv run pytest`
- [ ] Linting passes: `uv run ruff check src/ tests/`
- [ ] Type checking passes: `uv run mypy src/`
- [ ] Coverage is 94%+ for new code
- [ ] One feature per PR (keep PRs focused)
- [ ] Documentation updated for public APIs

### PR Title and Description
- Title: Start with verb (Add, Fix, Update, Refactor)
- Description: Explain why the change is needed
- Link related issues: "Closes #123"

### Example PR Template

```markdown
## Summary
Brief description of what this PR does.

## Changes
- Item 1
- Item 2

## Testing
- [ ] Unit tests added
- [ ] Integration tests passing
- [ ] Manual testing done

Closes #123
```

## Architecture Overview

anchor is organized into 14 core modules with 217+ exports:

- **Protocols**: Extension points for custom implementations
- **Pipeline**: Step-based execution framework
- **Retrieval**: Dense, sparse, and hybrid search
- **Ingestion**: Document parsing and chunking
- **Memory**: Conversation and context management
- **Query**: Transformation and classification
- **Evaluation**: RAG and retrieval metrics
- **Observability**: Tracing and cost tracking
- **Multimodal**: Image and table handling

See [Architecture](concepts/architecture.md) for details.

## Getting Help

- **Questions**: Check [Getting Started](getting-started/index.md)
- **API Reference**: See [API Docs](api/pipeline.md)
- **Guides**: Browse [Guides](guides/pipeline.md)
- **Issues**: [GitHub Issues](https://github.com/Astrointelligence-dev/anchor/issues)

## Code Review Expectations

We follow these principles:

- **Simplicity First**: Make changes as minimal as possible
- **No Laziness**: Find root causes, not quick fixes
- **Minimal Impact**: Only touch necessary code
- **Senior Standards**: Code should pass staff engineer review

Thank you for contributing to anchor!
