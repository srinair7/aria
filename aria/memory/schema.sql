-- ARIA SQLite schema
-- WAL mode is set at connection time in store.py

CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
    content     TEXT    NOT NULL,
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec'))
);

CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id, created_at);

CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    updated_at  REAL    NOT NULL DEFAULT (unixepoch('now', 'subsec')),
    UNIQUE(session_id, key)
);

CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id);
