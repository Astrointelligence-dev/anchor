---
icon: material/download
---

# Installation

## Requirements

- **Python 3.11+** is required.
- No system-level dependencies are needed for the core package.

## Install the package

=== "pip"

    ```bash
    pip install astro-anchor
    ```

=== "uv"

    ```bash
    uv add astro-anchor
    ```

## Optional extras

anchor ships with several optional dependency groups. Install only
what you need, or grab everything at once.

=== "pip"

    ```bash
    pip install astro-anchor[bm25]       # BM25 sparse retrieval
    pip install astro-anchor[cli]        # CLI tools (typer + rich)
    pip install astro-anchor[flashrank]  # FlashRank reranker
    pip install astro-anchor[anthropic]  # Anthropic token counting
    pip install astro-anchor[otlp]       # OpenTelemetry export
    pip install astro-anchor[all]        # Everything above
    ```

=== "uv"

    ```bash
    uv add astro-anchor[bm25]
    uv add astro-anchor[cli]
    uv add astro-anchor[flashrank]
    uv add astro-anchor[anthropic]
    uv add astro-anchor[otlp]
    uv add astro-anchor[all]
    ```

| Extra | What it adds | When you need it |
|-------|-------------|-----------------|
| `bm25` | `rank-bm25` | Sparse / keyword retrieval with `SparseRetriever` |
| `cli` | `typer`, `rich` | Using the `anchor` CLI |
| `flashrank` | `FlashRank` | Client-side reranking without an API call |
| `anthropic` | `anthropic` | Accurate token counting for Claude models |
| `otlp` | `opentelemetry-*` | Exporting traces and metrics via OTLP |
| `all` | All of the above | Kitchen-sink install for development |

## Verifying the installation

After installing, confirm the package is available:

```bash
python -c "import anchor; print(anchor.__version__)"
```

If you installed the `cli` extra you can also run:

```bash
anchor --version
```

## Development setup

To work on anchor itself, clone the repository and install in
editable mode with all extras:

=== "pip"

    ```bash
    git clone https://github.com/Astrointelligence-dev/anchor.git
    cd anchor
    pip install -e ".[all,dev]"
    ```

=== "uv"

    ```bash
    git clone https://github.com/Astrointelligence-dev/anchor.git
    cd anchor
    uv sync --all-extras
    ```

Run the test suite to make sure everything is working:

```bash
pytest
```

!!! tip "Next step"
    Head over to the [Quickstart](quickstart.md) to build your first pipeline
    in under 30 seconds.
