"""memory_20250818-compatible memory tool over FileMemoryBackend.

Spec-shaped commands, reference result strings, and — above all — the
path guards: nothing escapes the backend's base_path.
"""

from __future__ import annotations

from pathlib import Path

from anchor.agent import Agent, FileMemoryBackend, memory_tool
from tests.test_agent.test_agent import (
    FakeLLMProvider,
    _text_response,
    _Tok,
    _tool_use_response,
)
from tests.test_agent.test_phase4_loop import _tool_results_of


def _backend(tmp_path: Path) -> FileMemoryBackend:
    return FileMemoryBackend(tmp_path / "mem")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def test_create_and_view_file(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    assert backend.create("/memories/notes.md", "alpha\nbeta") == (
        "File created successfully at: /memories/notes.md"
    )
    out = backend.view("/memories/notes.md")
    assert "Here's the content of /memories/notes.md with line numbers:" in out
    assert "     1\talpha" in out
    assert "     2\tbeta" in out


def test_view_range_and_directory_listing(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create("/memories/a.md", "1\n2\n3\n4")
    backend.create("/memories/sub/b.md", "x")

    ranged = backend.view("/memories/a.md", [2, 3])
    assert "     2\t2" in ranged
    assert "     1\t1" not in ranged
    open_ended = backend.view("/memories/a.md", [3, -1])
    assert "     4\t4" in open_ended

    listing = backend.view("/memories")
    assert "up to 2 levels deep in /memories" in listing
    assert "a.md" in listing
    assert "sub/b.md" in listing


def test_str_replace_unique_multiple_and_missing(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create("/memories/n.md", "keep\nchange me\nkeep")

    assert backend.str_replace("/memories/n.md", "change me", "changed") == (
        "The memory file has been edited."
    )
    assert "changed" in backend.view("/memories/n.md")
    assert "did not appear verbatim" in backend.str_replace(
        "/memories/n.md", "ghost", "x",
    )
    multi = backend.str_replace("/memories/n.md", "keep", "x")
    assert "Multiple occurrences" in multi
    assert "1, 3" in multi


def test_insert_and_bounds(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create("/memories/n.md", "one\nthree")
    assert backend.insert("/memories/n.md", 1, "two") == (
        "The file /memories/n.md has been edited."
    )
    out = backend.view("/memories/n.md")
    assert "     2\ttwo" in out
    assert "Invalid `insert_line`" in backend.insert("/memories/n.md", 99, "x")


def test_delete_and_rename(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create("/memories/old.md", "data")
    assert backend.rename("/memories/old.md", "/memories/new.md") == (
        "Successfully renamed /memories/old.md to /memories/new.md"
    )
    backend.create("/memories/other.md", "x")
    assert "already exists" in backend.rename(
        "/memories/other.md", "/memories/new.md",
    )
    assert backend.delete("/memories/new.md") == (
        "Successfully deleted /memories/new.md"
    )
    assert "does not exist" in backend.view("/memories/new.md")


def test_root_is_protected(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    assert "cannot delete" in backend.delete("/memories")
    assert "cannot rename" in backend.rename("/memories", "/memories/x")


# ---------------------------------------------------------------------------
# Path guards
# ---------------------------------------------------------------------------


def test_traversal_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "secrets.env").write_text("KEY=1")
    backend = _backend(tmp_path)

    for attempt in (
        "/memories/../secrets.env",
        "/memories/../../etc/passwd",
        "/memories/%2e%2e/secrets.env",
        "/memories/%2E%2E%2Fsecrets.env",
        "/etc/passwd",
        "relative/path.md",
    ):
        out = backend.view(attempt)
        assert out.startswith("Error:"), attempt
        assert "KEY=1" not in out

    assert backend.create("/memories/../evil.md", "x").startswith("Error:")
    assert not (tmp_path / "evil.md").exists()


# ---------------------------------------------------------------------------
# Loop round-trip + agent wiring
# ---------------------------------------------------------------------------


def test_agent_round_trip_via_loop(tmp_path: Path) -> None:
    provider = FakeLLMProvider([
        _tool_use_response(
            "tu_1", "memory",
            {
                "command": "create",
                "path": "/memories/progress.md",
                "file_text": "step 1 done",
            },
        ),
        _text_response("saved"),
    ])
    agent = Agent(llm=provider, tokenizer=_Tok())
    agent.with_system_prompt("You are helpful.")
    agent.with_memory_tool(tmp_path / "mem")

    assert "".join(agent.chat("Work")) == "saved"
    (result,) = _tool_results_of(provider, 1)
    assert result.is_error is False
    assert "File created successfully" in result.content
    assert (tmp_path / "mem" / "progress.md").read_text() == "step 1 done"
    # The memory protocol went into the system prompt (multi-provider:
    # no automatic API injection).
    system = str(provider.seen_messages[0][0].content)
    assert "/memories" in system


def test_memory_tool_unknown_command(tmp_path: Path) -> None:
    tool = memory_tool(_backend(tmp_path))
    assert "unknown memory command" in tool.fn(command="explode")
