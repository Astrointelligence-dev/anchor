# Tokens API Reference

The `anchor.tokens` module provides token counting utilities used by the
context pipeline for budget management.

All classes are importable from `anchor.tokens`:

```python
from anchor.tokens import TiktokenCounter, get_default_counter
```

---

## TiktokenCounter

Token counter using OpenAI's tiktoken library. Default encoding is
`cl100k_base` (used by GPT-4; Claude tokenizers are similar enough for
budget estimation purposes). Implements the `Tokenizer` protocol via
structural subtyping.

The tiktoken import is deferred to `__init__` so that importing this module
does not trigger BPE data loading when callers supply their own `Tokenizer`
implementation.

### Constructor

```python
class TiktokenCounter:
    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        max_cache_size: int = 10_000,
    ) -> None
```

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `encoding_name` | `str` | `"cl100k_base"` | Tiktoken encoding name |
| `max_cache_size` | `int` | `10_000` | Maximum entries in the token count cache |

Raises `ImportError` if tiktoken is not installed.

### Methods

#### count_tokens

```python
def count_tokens(self, text: str) -> int
```

Count the number of tokens in a text string. Results for strings under
10,000 characters are cached for performance.

#### truncate_to_tokens

```python
def truncate_to_tokens(self, text: str, max_tokens: int) -> str
```

Truncate text to fit within a token limit. Returns the original string
if it is already within the limit.

### Example

```python
from anchor.tokens import TiktokenCounter

counter = TiktokenCounter()
count = counter.count_tokens("Hello, world!")
truncated = counter.truncate_to_tokens("A very long text...", max_tokens=10)
```

---

## get_default_counter

Singleton factory for the default `TiktokenCounter`.

### Signature

```python
@functools.cache
def get_default_counter() -> TiktokenCounter
```

Returns a cached `TiktokenCounter` instance using the default `cl100k_base`
encoding. Thread-safe via `functools.cache`.

Call `get_default_counter.cache_clear()` to reset the singleton (useful in
tests).

Raises `ImportError` if tiktoken is not installed.

### Example

```python
from anchor.tokens import get_default_counter

counter = get_default_counter()
print(counter.count_tokens("Hello!"))
```

---

## See Also

- [Token Budgets Concept](../concepts/token-budgets.md) -- how token budgets work
- [Pipeline API Reference](pipeline.md) -- the pipeline that uses token counting
- [Protocols Reference](protocols.md) -- the Tokenizer protocol
