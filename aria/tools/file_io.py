"""File I/O tools — read_file, write_file, list_dir."""
from __future__ import annotations

import os
from pathlib import Path

READ_FILE_DEF = {
    "name": "read_file",
    "description": "Read the contents of a file from disk.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative file path"},
            "encoding": {
                "type": "string",
                "description": "Text encoding (default: utf-8)",
                "default": "utf-8",
            },
        },
        "required": ["path"],
    },
}

WRITE_FILE_DEF = {
    "name": "write_file",
    "description": "Write text content to a file, creating parent directories as needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative file path"},
            "content": {"type": "string", "description": "Text content to write"},
            "append": {
                "type": "boolean",
                "description": "Append to file instead of overwriting (default: false)",
                "default": False,
            },
        },
        "required": ["path", "content"],
    },
}

LIST_DIR_DEF = {
    "name": "list_dir",
    "description": "List files and subdirectories inside a directory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path (default: current working directory)",
                "default": ".",
            },
        },
        "required": [],
    },
}

_MAX_READ_BYTES = 512 * 1024  # 512 KB safety limit


def read_file(path: str, encoding: str = "utf-8") -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Error: file not found: {p}"
    if not p.is_file():
        return f"Error: path is not a file: {p}"
    size = p.stat().st_size
    if size > _MAX_READ_BYTES:
        return (
            f"Error: file too large ({size:,} bytes). "
            f"Max allowed: {_MAX_READ_BYTES:,} bytes."
        )
    try:
        return p.read_text(encoding=encoding)
    except Exception as exc:
        return f"Error reading file: {exc}"


def write_file(path: str, content: str, append: bool = False) -> str:
    p = Path(path).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(p, mode, encoding="utf-8") as fh:
            fh.write(content)
        action = "Appended to" if append else "Wrote"
        return f"{action} {p} ({len(content):,} chars)"
    except Exception as exc:
        return f"Error writing file: {exc}"


def list_dir(path: str = ".") -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"Error: directory not found: {p}"
    if not p.is_dir():
        return f"Error: path is not a directory: {p}"
    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        lines = []
        for entry in entries:
            if entry.is_dir():
                lines.append(f"[DIR]  {entry.name}/")
            else:
                size = entry.stat().st_size
                lines.append(f"[FILE] {entry.name}  ({size:,} bytes)")
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as exc:
        return f"Error listing directory: {exc}"
