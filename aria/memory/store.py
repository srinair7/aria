"""Async SQLite memory store (conversations + facts)."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiosqlite

from aria.config import get_config

_SCHEMA = Path(__file__).parent / "schema.sql"


class MemoryStore:
    def __init__(self, db_path: Path | None = None) -> None:
        cfg = get_config()
        self.db_path = db_path or cfg.db_path
        self.memory_window = cfg.memory_window
        self._db: aiosqlite.Connection | None = None

    async def __aenter__(self) -> "MemoryStore":
        await self.open()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        schema = _SCHEMA.read_text()
        await self._db.executescript(schema)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # ── Conversations ────────────────────────────────────────────────────────

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        await self._db.execute(
            "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        await self._db.commit()

    async def get_history(self, session_id: str) -> list[dict]:
        """Return last `memory_window` messages for a session as Claude message dicts."""
        async with self._db.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content, created_at
                FROM conversations
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (session_id, self.memory_window),
        ) as cur:
            rows = await cur.fetchall()
        return [{"role": r, "content": c} for r, c in rows]

    async def clear_history(self, session_id: str) -> None:
        await self._db.execute(
            "DELETE FROM conversations WHERE session_id = ?", (session_id,)
        )
        await self._db.commit()

    # ── Facts ────────────────────────────────────────────────────────────────

    async def set_fact(self, session_id: str, key: str, value: str) -> None:
        await self._db.execute(
            """
            INSERT INTO facts (session_id, key, value, updated_at)
            VALUES (?, ?, ?, unixepoch('now', 'subsec'))
            ON CONFLICT(session_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (session_id, key, value),
        )
        await self._db.commit()

    async def get_facts(self, session_id: str) -> dict[str, str]:
        async with self._db.execute(
            "SELECT key, value FROM facts WHERE session_id = ? ORDER BY key",
            (session_id,),
        ) as cur:
            rows = await cur.fetchall()
        return {k: v for k, v in rows}

    async def delete_fact(self, session_id: str, key: str) -> None:
        await self._db.execute(
            "DELETE FROM facts WHERE session_id = ? AND key = ?", (session_id, key)
        )
        await self._db.commit()

    # ── Global facts (session_id = '__global__') ─────────────────────────────

    async def set_global_fact(self, key: str, value: str) -> None:
        await self.set_fact("__global__", key, value)

    async def get_global_facts(self) -> dict[str, str]:
        return await self.get_facts("__global__")


# Module-level singleton (lazy)
_store: MemoryStore | None = None
_lock = asyncio.Lock()


async def get_store() -> MemoryStore:
    global _store
    async with _lock:
        if _store is None:
            _store = MemoryStore()
            await _store.open()
    return _store
