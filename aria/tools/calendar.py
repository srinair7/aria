"""Calendar & daily planning tool.

Stores events and todos in the ARIA SQLite database.
Supports: add event, list events (day/week), add todo, list/complete todos,
and get a structured daily plan summary.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, date
from pathlib import Path


# ── Tool definitions ──────────────────────────────────────────────────────────

ADD_EVENT_DEF = {
    "name": "add_event",
    "description": "Add a calendar event (meeting, appointment, task with a specific time).",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Event title"},
            "start": {
                "type": "string",
                "description": "Start datetime, ISO format or natural ('today 2pm', 'tomorrow 10am')",
            },
            "end": {
                "type": "string",
                "description": "End datetime (optional). If omitted, defaults to 1 hour after start.",
            },
            "notes": {"type": "string", "description": "Optional notes or description"},
        },
        "required": ["title", "start"],
    },
}

LIST_EVENTS_DEF = {
    "name": "list_events",
    "description": "List calendar events for a given day or range.",
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "Date to list: 'today', 'tomorrow', 'this week', or ISO date 'YYYY-MM-DD'",
                "default": "today",
            },
        },
        "required": [],
    },
}

DELETE_EVENT_DEF = {
    "name": "delete_event",
    "description": "Delete a calendar event by ID.",
    "input_schema": {
        "type": "object",
        "properties": {
            "event_id": {"type": "integer", "description": "Event ID from list_events"},
        },
        "required": ["event_id"],
    },
}

ADD_TODO_DEF = {
    "name": "add_todo",
    "description": "Add a to-do item to the user's task list.",
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "Task description"},
            "due": {
                "type": "string",
                "description": "Optional due date ('today', 'tomorrow', 'YYYY-MM-DD')",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Priority level (default: medium)",
                "default": "medium",
            },
        },
        "required": ["task"],
    },
}

LIST_TODOS_DEF = {
    "name": "list_todos",
    "description": "List to-do items, optionally filtered by completion status or due date.",
    "input_schema": {
        "type": "object",
        "properties": {
            "filter": {
                "type": "string",
                "enum": ["all", "pending", "done", "today", "overdue"],
                "description": "Filter to apply (default: pending)",
                "default": "pending",
            },
        },
        "required": [],
    },
}

COMPLETE_TODO_DEF = {
    "name": "complete_todo",
    "description": "Mark a to-do item as done.",
    "input_schema": {
        "type": "object",
        "properties": {
            "todo_id": {"type": "integer", "description": "Todo ID from list_todos"},
        },
        "required": ["todo_id"],
    },
}

DAILY_PLAN_DEF = {
    "name": "daily_plan",
    "description": (
        "Get a structured overview of the day: events, todos, and reminders. "
        "Use this to help the user plan or review their day."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "Date to plan: 'today' or 'tomorrow' (default: today)",
                "default": "today",
            },
        },
        "required": [],
    },
}


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_db_path() -> Path:
    from aria.config import get_config
    return get_config().db_path


def _conn() -> sqlite3.Connection:
    db = _get_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db)


def _ensure_tables() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                title     TEXT NOT NULL,
                start_ts  REAL NOT NULL,
                end_ts    REAL NOT NULL,
                notes     TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS todos (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                task      TEXT NOT NULL,
                due_date  TEXT,
                priority  TEXT DEFAULT 'medium',
                done      INTEGER DEFAULT 0,
                created_at REAL DEFAULT (unixepoch())
            );
        """)


# ── Time parsing (reuse from reminders) ──────────────────────────────────────

def _parse_dt(s: str) -> datetime:
    from aria.tools.reminders import _parse_when
    return _parse_when(s)


def _parse_date(s: str) -> date:
    s = s.strip().lower()
    today = date.today()
    if s in ("today", ""):
        return today
    if s == "tomorrow":
        return today + timedelta(days=1)
    if s == "yesterday":
        return today - timedelta(days=1)
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return today


# ── Event tools ───────────────────────────────────────────────────────────────

def add_event(title: str, start: str, end: str | None = None, notes: str = "") -> str:
    _ensure_tables()
    try:
        start_dt = _parse_dt(start)
    except ValueError as exc:
        return f"Error parsing start time: {exc}"

    if end:
        try:
            end_dt = _parse_dt(end)
        except ValueError as exc:
            return f"Error parsing end time: {exc}"
    else:
        end_dt = start_dt + timedelta(hours=1)

    with _conn() as c:
        cur = c.execute(
            "INSERT INTO events (title, start_ts, end_ts, notes) VALUES (?,?,?,?)",
            (title, start_dt.timestamp(), end_dt.timestamp(), notes),
        )
        eid = cur.lastrowid

    return (
        f"Event #{eid} added: '{title}'\n"
        f"  Start: {start_dt.strftime('%A %b %d, %Y %H:%M')}\n"
        f"  End:   {end_dt.strftime('%H:%M')}"
    )


def list_events(date: str = "today") -> str:
    _ensure_tables()
    d = date.strip().lower()

    if d == "this week":
        today = datetime.today()
        start = datetime.combine(today.date() - timedelta(days=today.weekday()), datetime.min.time())
        end = start + timedelta(days=7)
    else:
        day = _parse_date(d)
        start = datetime.combine(day, datetime.min.time())
        end = start + timedelta(days=1)

    with _conn() as c:
        rows = c.execute(
            "SELECT id, title, start_ts, end_ts, notes FROM events "
            "WHERE start_ts >= ? AND start_ts < ? ORDER BY start_ts",
            (start.timestamp(), end.timestamp()),
        ).fetchall()

    if not rows:
        return f"No events for {d}."

    lines = [f"Events ({d}):"]
    for eid, title, s, e, notes in rows:
        sd = datetime.fromtimestamp(s).strftime("%H:%M")
        ed = datetime.fromtimestamp(e).strftime("%H:%M")
        line = f"  #{eid} {sd}–{ed}  {title}"
        if notes:
            line += f"\n       {notes}"
        lines.append(line)
    return "\n".join(lines)


def delete_event(event_id: int) -> str:
    _ensure_tables()
    with _conn() as c:
        cur = c.execute("DELETE FROM events WHERE id=?", (event_id,))
    if cur.rowcount:
        return f"Event #{event_id} deleted."
    return f"Event #{event_id} not found."


# ── Todo tools ────────────────────────────────────────────────────────────────

def add_todo(task: str, due: str | None = None, priority: str = "medium") -> str:
    _ensure_tables()
    due_date: str | None = None
    if due:
        due_date = _parse_date(due).isoformat()

    with _conn() as c:
        cur = c.execute(
            "INSERT INTO todos (task, due_date, priority) VALUES (?,?,?)",
            (task, due_date, priority),
        )
        tid = cur.lastrowid

    due_str = f" (due {due_date})" if due_date else ""
    return f"Todo #{tid} added: '{task}'{due_str} [{priority}]"


def list_todos(filter: str = "pending") -> str:
    _ensure_tables()
    today = date.today().isoformat()
    f = filter.strip().lower()

    if f == "all":
        where, params = "1=1", ()
    elif f == "done":
        where, params = "done=1", ()
    elif f == "today":
        where, params = "done=0 AND (due_date=? OR due_date IS NULL)", (today,)
    elif f == "overdue":
        where, params = "done=0 AND due_date < ?", (today,)
    else:  # pending
        where, params = "done=0", ()

    with _conn() as c:
        rows = c.execute(
            f"SELECT id, task, due_date, priority, done FROM todos WHERE {where} "
            f"ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, due_date",
            params,
        ).fetchall()

    if not rows:
        return f"No todos ({f})."

    icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    lines = [f"Todos ({f}):"]
    for tid, task, due_date, priority, done in rows:
        status = "✓" if done else " "
        icon = icons.get(priority, "•")
        due_str = f" [due {due_date}]" if due_date else ""
        lines.append(f"  [{status}] #{tid} {icon} {task}{due_str}")
    return "\n".join(lines)


def complete_todo(todo_id: int) -> str:
    _ensure_tables()
    with _conn() as c:
        cur = c.execute("UPDATE todos SET done=1 WHERE id=? AND done=0", (todo_id,))
    if cur.rowcount:
        return f"Todo #{todo_id} marked as done. ✓"
    return f"Todo #{todo_id} not found or already done."


# ── Daily plan ────────────────────────────────────────────────────────────────

def daily_plan(date: str = "today") -> str:
    _ensure_tables()
    from aria.tools.reminders import _list_pending
    import time as _time

    d = date.strip().lower()
    day = _parse_date(d)
    start = datetime.combine(day, datetime.min.time())
    end = start + timedelta(days=1)

    sections: list[str] = [f"── Daily Plan: {day.strftime('%A, %B %d %Y')} ──\n"]

    # Events
    ev = list_events(d)
    sections.append(ev)

    # Todos due today
    today_todos = list_todos("today" if d == "today" else "pending")
    sections.append(today_todos)

    # Overdue
    overdue = list_todos("overdue")
    if "No todos" not in overdue:
        sections.append("⚠️  " + overdue)

    # Reminders firing today
    pending_reminders = _list_pending()
    day_reminders = [
        r for r in pending_reminders
        if start.timestamp() <= r["fire_at"] < end.timestamp()
    ]
    if day_reminders:
        lines = ["Reminders today:"]
        for r in day_reminders:
            t = datetime.fromtimestamp(r["fire_at"]).strftime("%H:%M")
            lines.append(f"  🔔 {t}  {r['message']}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)
