"""Client-side memory tool compatible with Anthropic's ``memory_20250818``.

One ``memory`` tool with the spec's six commands (view / create /
str_replace / insert / delete / rename) over a file-based backend, as a
regular :class:`AgentTool` — it works on every provider, not only
Anthropic. Result and error strings follow the Anthropic reference
implementation so models keep their trained behavior.

The model addresses paths under the virtual ``/memories`` root; the
backend maps them into ``base_path`` with strict containment (path
traversal — including URL-encoded variants — is rejected, per the
spec's security guidance).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from anchor.agent.models import AgentTool

_ROOT = "/memories"
_MAX_VIEW_CHARS = 16_000
_MAX_LINES = 999_999

MEMORY_INSTRUCTIONS = (
    "You have a memory tool rooted at /memories. Check it (view) before "
    "starting work, record important context and progress as you go, and "
    "assume the conversation may be interrupted at any time — anything "
    "worth keeping belongs in memory files."
)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "enum": ["view", "create", "str_replace", "insert", "delete", "rename"],
            "description": "The memory operation to run.",
        },
        "path": {"type": "string", "description": "Path under /memories."},
        "view_range": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "[start_line, end_line], 1-indexed; -1 = end.",
        },
        "file_text": {"type": "string"},
        "old_str": {"type": "string"},
        "new_str": {"type": "string"},
        "insert_line": {"type": "integer"},
        "insert_text": {"type": "string"},
        "old_path": {"type": "string"},
        "new_path": {"type": "string"},
    },
    "required": ["command"],
}


def _human_size(size: float) -> str:
    for unit in ("B", "K", "M"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}G"


class FileMemoryBackend:
    """File-based storage for the memory tool.

    Maps the virtual ``/memories`` root into *base_path* with strict
    containment. Not safe for concurrent multi-process access.
    """

    __slots__ = ("_base",)

    def __init__(self, base_path: str | Path) -> None:
        self._base = Path(base_path).resolve()
        self._base.mkdir(parents=True, exist_ok=True)

    # -- path guard --

    def _resolve(self, path: str | None) -> tuple[Path | None, str | None]:
        """Map a ``/memories`` path into the backend. (resolved, error)."""
        if not path:
            return None, "Error: a 'path' under /memories is required."
        lowered = path.lower()
        if "%2e" in lowered or "%2f" in lowered or "%5c" in lowered:
            return None, f"Error: invalid path '{path}'."
        if path != _ROOT and not path.startswith(_ROOT + "/"):
            return None, (
                f"Error: path '{path}' must be inside the {_ROOT} directory."
            )
        relative = path[len(_ROOT) :].lstrip("/")
        resolved = (self._base / relative).resolve() if relative else self._base
        if resolved != self._base and self._base not in resolved.parents:
            return None, f"Error: path '{path}' escapes the memory directory."
        return resolved, None

    @staticmethod
    def _missing(path: str) -> str:
        return f"The path {path} does not exist. Please provide a valid path."

    # -- commands --

    def view(self, path: str, view_range: list[int] | None = None) -> str:
        resolved, err = self._resolve(path)
        if err is not None:
            return err
        if resolved.is_dir():
            return self._view_directory(path, resolved)
        if not resolved.is_file():
            return self._missing(path)
        return self._view_file(path, resolved, view_range)

    def _view_directory(self, path: str, resolved: Path) -> str:
        lines: list[str] = []
        for child in sorted(resolved.rglob("*")):
            relative = child.relative_to(resolved)
            if any(
                part.startswith(".") or part == "node_modules"
                for part in relative.parts
            ):
                continue
            if len(relative.parts) > 2:  # up to 2 levels deep
                continue
            size = child.stat().st_size if child.is_file() else 0
            lines.append(f"{_human_size(size)}\t{path.rstrip('/')}/{relative}")
        listing = "\n".join(lines)
        return (
            f"Here're the files and directories up to 2 levels deep in "
            f"{path}, excluding hidden items and node_modules:\n{listing}"
        )

    def _view_file(
        self, path: str, resolved: Path, view_range: list[int] | None,
    ) -> str:
        text = resolved.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) > _MAX_LINES:
            return f"File {path} exceeds maximum line limit of 999,999 lines."
        start, end = 1, len(lines)
        if view_range:
            start = max(1, view_range[0])
            end = len(lines) if view_range[1] == -1 else min(
                len(lines), view_range[1],
            )
        numbered = "\n".join(
            f"{i:>6}\t{line}"
            for i, line in enumerate(lines[start - 1 : end], start=start)
        )
        if len(numbered) > _MAX_VIEW_CHARS and not view_range:
            numbered = numbered[:_MAX_VIEW_CHARS] + (
                "\n... (truncated — use view_range to read further)"
            )
        return f"Here's the content of {path} with line numbers:\n{numbered}"

    def create(self, path: str, file_text: str) -> str:
        resolved, err = self._resolve(path)
        if err is not None:
            return err
        if resolved == self._base:
            return f"Error: cannot create the {_ROOT} root."
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(file_text, encoding="utf-8")
        return f"File created successfully at: {path}"

    def str_replace(self, path: str, old_str: str, new_str: str = "") -> str:
        resolved, err = self._resolve(path)
        if err is not None:
            return err
        if not resolved.is_file():
            return self._missing(path)
        text = resolved.read_text(encoding="utf-8")
        count = text.count(old_str)
        if count == 0:
            return (
                f"No replacement was performed, old_str `{old_str}` did "
                f"not appear verbatim in {path}."
            )
        if count > 1:
            occurrences = [
                str(i)
                for i, line in enumerate(text.splitlines(), start=1)
                if old_str in line
            ]
            return (
                f"No replacement was performed. Multiple occurrences of "
                f"old_str `{old_str}` in lines: {', '.join(occurrences)}. "
                "Please ensure it is unique"
            )
        resolved.write_text(text.replace(old_str, new_str), encoding="utf-8")
        return "The memory file has been edited."

    def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        resolved, err = self._resolve(path)
        if err is not None:
            return err
        if not resolved.is_file():
            return self._missing(path)
        lines = resolved.read_text(encoding="utf-8").splitlines()
        if not 0 <= insert_line <= len(lines):
            return (
                f"Error: Invalid `insert_line` parameter: {insert_line}. It "
                "should be within the range of lines of the file: "
                f"[0, {len(lines)}]"
            )
        lines.insert(insert_line, insert_text)
        resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"The file {path} has been edited."

    def delete(self, path: str) -> str:
        resolved, err = self._resolve(path)
        if err is not None:
            return err
        if resolved == self._base:
            return f"Error: cannot delete the {_ROOT} root."
        if resolved.is_dir():
            shutil.rmtree(resolved)
            return f"Successfully deleted {path}"
        if resolved.is_file():
            resolved.unlink()
            return f"Successfully deleted {path}"
        return self._missing(path)

    def rename(self, old_path: str, new_path: str) -> str:
        source, err = self._resolve(old_path)
        if err is not None:
            return err
        target, err = self._resolve(new_path)
        if err is not None:
            return err
        if source == self._base:
            return f"Error: cannot rename the {_ROOT} root."
        if not source.exists():
            return self._missing(old_path)
        if target.exists():
            return f"Error: {new_path} already exists."
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        return f"Successfully renamed {old_path} to {new_path}"


def memory_tool(backend: FileMemoryBackend) -> AgentTool:
    """Build the ``memory`` tool over *backend*."""

    def memory(command: str, **kwargs: Any) -> str:
        if command == "view":
            return backend.view(kwargs.get("path", ""), kwargs.get("view_range"))
        if command == "create":
            return backend.create(
                kwargs.get("path", ""), kwargs.get("file_text", ""),
            )
        if command == "str_replace":
            return backend.str_replace(
                kwargs.get("path", ""),
                kwargs.get("old_str", ""),
                kwargs.get("new_str", ""),
            )
        if command == "insert":
            return backend.insert(
                kwargs.get("path", ""),
                kwargs.get("insert_line", -1),
                kwargs.get("insert_text", ""),
            )
        if command == "delete":
            return backend.delete(kwargs.get("path", ""))
        if command == "rename":
            return backend.rename(
                kwargs.get("old_path", ""), kwargs.get("new_path", ""),
            )
        return f"Error: unknown memory command '{command}'."

    return AgentTool(
        name="memory",
        description=(
            "Persistent memory under /memories. Commands: view (path, "
            "optional view_range), create (path, file_text), str_replace "
            "(path, old_str, new_str), insert (path, insert_line, "
            "insert_text), delete (path), rename (old_path, new_path)."
        ),
        input_schema=_SCHEMA,
        fn=memory,
    )
