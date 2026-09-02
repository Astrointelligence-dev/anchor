"""CLI interface for anchor.

Requires the 'cli' extra: pip install astro-anchor[cli]

``anchor index`` ingests documents into a single SQLite database (chunks +
optional dense vectors); ``anchor query`` runs hybrid retrieval over it.
Works fully offline with ``--embeddings none`` (BM25 only) or
``--embeddings local`` (sentence-transformers).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

try:
    import typer
    from rich.console import Console
    from rich.table import Table
except ImportError:
    print(
        "CLI dependencies not installed. Install with: pip install astro-anchor[cli]",
        file=sys.stderr,
    )
    sys.exit(1)

from anchor import __version__

app = typer.Typer(
    name="anchor",
    help="Context engineering toolkit for AI applications.",
    add_completion=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"anchor {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Anchor CLI."""


@app.command()
def info() -> None:
    """Show information about the anchor installation."""
    table = Table(title="anchor info")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Version", __version__)
    table.add_row("Python", sys.version.split()[0])

    for dep_name in ["bm25s", "sqlite_vec", "tiktoken", "pydantic"]:
        try:
            mod = importlib.import_module(dep_name)
            ver = getattr(mod, "__version__", "installed")
            table.add_row(dep_name, str(ver))
        except ImportError:
            table.add_row(dep_name, "[red]not installed[/red]")

    console.print(table)


def _make_embeddings(spec: str) -> Any:
    """Build an EmbeddingProvider from a CLI spec.

    Specs: ``none`` | ``openai[:model]`` | ``local[:model]`` | ``voyage[:model]``.
    """
    if spec == "none":
        return None
    provider, _, model = spec.partition(":")
    if provider == "openai":
        from anchor.embeddings import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(model=model or "text-embedding-3-small")
    if provider == "local":
        from anchor.embeddings import SentenceTransformerEmbeddingProvider

        return SentenceTransformerEmbeddingProvider(model=model or "BAAI/bge-m3")
    if provider == "voyage":
        from anchor.embeddings import VoyageEmbeddingProvider

        return VoyageEmbeddingProvider(model=model or "voyage-3.5")
    console.print(
        f"[red]Unknown embeddings spec '{spec}'. "
        "Use none | openai[:model] | local[:model] | voyage[:model][/red]"
    )
    raise typer.Exit(code=1)


def _open_context_store(db_path: Path, vault: str = "__default__") -> Any:
    from anchor.storage.sqlite import (
        SqliteConnectionManager,
        SqliteContextStore,
        ensure_tables,
    )

    manager = SqliteConnectionManager(db_path)
    ensure_tables(manager.get_connection())
    try:
        return SqliteContextStore(manager, vault=vault)
    except ValueError as e:
        console.print(f"[red]--vault: {e}[/red]")
        raise typer.Exit(code=1) from None


def _open_vector_store(
    db_path: Path, dimensions: int, vault: str = "__default__",
) -> Any:
    """Prefer sqlite-vec (real KNN); fall back to the brute-force store."""
    try:
        from anchor.storage.sqlite import SqliteVecVectorStore

        return SqliteVecVectorStore(db_path, dimensions=dimensions, vault=vault)
    except ValueError as e:
        # Declared dimension disagrees with the index on disk: a different
        # embedding provider than the one used at index time.
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None
    except ImportError:
        from anchor.storage.sqlite import (
            SqliteConnectionManager,
            SqliteVectorStore,
            ensure_tables,
        )

        console.print(
            "[dim]sqlite-vec not installed; using brute-force vector store. "
            "pip install astro-anchor[sqlite-vec] for real KNN.[/dim]"
        )
        manager = SqliteConnectionManager(db_path)
        ensure_tables(manager.get_connection())
        return SqliteVectorStore(manager, vault=vault)


@app.command()
def index(
    path: Path = typer.Argument(..., help="Path to file or directory to index"),  # noqa: B008 -- typer.Argument() must be called in default
    db: Path = typer.Option(  # noqa: B008 -- typer.Option() must be called in default
        Path("anchor.db"), "--db", help="SQLite database file for the index"
    ),
    embeddings: str = typer.Option(
        "none",
        "--embeddings",
        "-e",
        help="Embedding provider: none | openai[:model] | local[:model] | voyage[:model]",
    ),
    chunk_size: int = typer.Option(384, "--chunk-size", "-c", help="Chunk size in tokens"),
    vault: str = typer.Option(
        "__default__", "--vault", help="Vault (hard isolation mount) to index into"
    ),
    namespace: str = typer.Option(
        "/", "--namespace", "-n",
        help="Namespace path to stamp on the ingested chunks (e.g. /contratos/2026)",
    ),
) -> None:
    """Ingest documents into a local index (chunks + optional dense vectors)."""
    path = path.resolve()
    if not path.exists():
        console.print(f"[red]Error: {path} does not exist[/red]")
        raise typer.Exit(code=1)

    from anchor.ingestion import DocumentIngester, RecursiveCharacterChunker

    ingester = DocumentIngester(chunker=RecursiveCharacterChunker(chunk_size=chunk_size))
    with console.status("Ingesting..."):
        items = (
            ingester.ingest_file(path)
            if path.is_file()
            else ingester.ingest_directory(path)
        )
    if not items:
        console.print("[yellow]No content ingested.[/yellow]")
        raise typer.Exit(code=1)

    from anchor.models.scope import normalize_namespace

    try:
        ns = normalize_namespace(namespace)
    except ValueError as e:
        console.print(f"[red]--namespace: {e}[/red]")
        raise typer.Exit(code=1) from None
    items = [item.model_copy(update={"namespace": ns}) for item in items]

    context_store = _open_context_store(db, vault)
    for item in items:
        context_store.add(item)

    provider = _make_embeddings(embeddings)
    if provider is not None:
        with console.status(f"Embedding {len(items)} chunks..."):
            vectors = provider.embed_documents([item.content for item in items])
            vector_store = _open_vector_store(db, len(vectors[0]), vault)
            for item, vector in zip(items, vectors, strict=True):
                vector_store.add_embedding(
                    item.id, vector, item.metadata, namespace=ns,
                )

    table = Table(title=f"Indexed into {db}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Chunks", str(len(items)))
    table.add_row("Total tokens", str(sum(item.token_count for item in items)))
    table.add_row("Dense vectors", str(len(items)) if provider else "no (BM25 only)")
    table.add_row("Vault", vault)
    table.add_row("Namespace", ns)
    console.print(table)


@app.command()
def migrate(
    db: Path = typer.Option(  # noqa: B008 -- typer.Option() must be called in default
        Path("anchor.db"), "--db", help="SQLite database file to migrate"
    ),
) -> None:
    """Migrate a pre-vault index in place (adds vault/namespace scoping).

    Idempotent: tables gain the scope columns with the ``__default__``
    sentinel, and an old sqlite-vec index is rebuilt by copying the
    embedding blobs (no re-embedding). ``anchor index`` runs the same
    migration automatically; this command exists for explicit runs.
    """
    if not db.exists():
        console.print(f"[red]No database at {db}.[/red]")
        raise typer.Exit(code=1)

    from anchor.storage.sqlite import SqliteConnectionManager, ensure_tables

    conn = SqliteConnectionManager(db).get_connection()
    ensure_tables(conn)

    # A sqlite-vec index migrates on open; its dimension is read from the
    # index itself, so nothing about the embedding provider is needed.
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'vec_index'"
    ).fetchone():
        from anchor.storage.sqlite import SqliteVecVectorStore

        try:
            SqliteVecVectorStore(db).close()
        except ImportError as e:
            console.print(f"[red]{db} has a sqlite-vec index but: {e}[/red]")
            raise typer.Exit(code=1) from None

    console.print(f"[green]{db} migrated (vault/namespace ready).[/green]")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query text"),
    db: Path = typer.Option(  # noqa: B008 -- typer.Option() must be called in default
        Path("anchor.db"), "--db", help="SQLite database file for the index"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    embeddings: str = typer.Option(
        "none",
        "--embeddings",
        "-e",
        help="Embedding provider (must match the one used at index time)",
    ),
    language: str = typer.Option(
        "english", "--language", "-l", help="Snowball language for BM25 stemming"
    ),
    vault: str = typer.Option(
        "__default__", "--vault", help="Vault (hard isolation mount) to search"
    ),
    include: list[str] = typer.Option(  # noqa: B008 -- typer.Option() must be called in default
        [], "--include", help="Namespace prefix(es) to include (repeatable)"
    ),
    exclude: list[str] = typer.Option(  # noqa: B008 -- typer.Option() must be called in default
        [], "--exclude", help="Namespace prefix(es) to exclude (repeatable; wins)"
    ),
) -> None:
    """Run hybrid retrieval (BM25 + optional dense, fused with RRF)."""
    if not db.exists():
        console.print(f"[red]No index at {db}. Run 'anchor index' first.[/red]")
        raise typer.Exit(code=1)

    from anchor.models.query import QueryBundle
    from anchor.models.scope import RetrievalScope, scope_kwargs
    from anchor.retrieval import HybridRetriever, SparseRetriever

    try:
        scope = (
            RetrievalScope(include=tuple(include), exclude=tuple(exclude))
            if include or exclude
            else None
        )
    except ValueError as e:
        console.print(f"[red]--include/--exclude: {e}[/red]")
        raise typer.Exit(code=1) from None

    context_store = _open_context_store(db, vault)
    items = context_store.get_all()
    if not items:
        console.print("[yellow]Index is empty (for this vault).[/yellow]")
        raise typer.Exit(code=1)

    sparse = SparseRetriever(language=language)
    sparse.index(items)
    retrievers: list[Any] = [sparse]

    provider = _make_embeddings(embeddings)
    if provider is not None:
        from anchor.retrieval import DenseRetriever

        vector_store = _open_vector_store(db, provider.dimensions, vault)
        retrievers.append(
            DenseRetriever(vector_store, context_store, embeddings=provider)
        )

    retriever: Any = (
        HybridRetriever(retrievers) if len(retrievers) > 1 else retrievers[0]
    )
    results = retriever.retrieve(
        QueryBundle(query_str=query_text), top_k=top_k, **scope_kwargs(scope),
    )

    if not results:
        console.print("[yellow]No results.[/yellow]")
        raise typer.Exit()

    table = Table(title=f"Top {len(results)} for: {query_text}")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", style="cyan", width=7)
    table.add_column("Source", style="magenta")
    table.add_column("Content", style="green")
    for rank, item in enumerate(results, start=1):
        source = str(item.metadata.get("doc_filename", item.metadata.get("parent_doc_id", "")))
        page = item.metadata.get("doc_page")
        if page:
            source = f"{source} p.{page}"
        snippet = item.content[:160].replace("\n", " ")
        table.add_row(str(rank), f"{item.score:.3f}", source, snippet)
    console.print(table)


if __name__ == "__main__":
    app()
