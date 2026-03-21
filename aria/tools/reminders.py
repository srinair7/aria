"""Reminders & alarms tool.

Reminders are persisted in the ARIA database and a background thread
fires them (macOS `say` or a console bell) at the scheduled time.
The scheduler thread starts automatically when the module is imported.
"""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import re

# ── Tool definitions ──────────────────────────────────────────────────────────

SET_REMINDER_DEF = {
    "name": "set_reminder",
    "description": (
        "Set a reminder or alarm for the user. "
        "Accepts a natural-language time like '5 minutes', '2 hours', "
        "'tomorrow 9am', or an ISO datetime string. "
        "The reminder fires a spoken/visual alert at the specified time."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "What to remind the user about"},
            "when": {
                "type": "string",
                "description": (
                    "When to fire: relative ('in 10 minutes', 'in 2 hours', 'in 30 seconds') "
                    "or absolute ISO datetime ('2026-03-20T14:30:00') or natural "
                    "('tomorrow 9am', 'today 3pm')"
                ),
            },
        },
        "required": ["message", "when"],
    },
}

LIST_REMINDERS_DEF = {
    "name": "list_reminders",
    "description": "List all pending (not yet fired) reminders.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

DELETE_REMINDER_DEF = {
    "name": "delete_reminder",
    "description": "Cancel a pending reminder by its ID.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reminder_id": {"type": "integer", "description": "ID from list_reminders"}
        },
        "required": ["reminder_id"],
    },
}


# ── Time parsing ──────────────────────────────────────────────────────────────

def _parse_when(when: str) -> datetime:
    """Parse a flexible time expression into a UTC datetime."""
    now = datetime.now()
    w = when.strip().lower()

    # Relative: "in N minutes/hours/seconds/days"
    m = re.match(r"in\s+(\d+)\s+(second|minute|hour|day)s?", w)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"second": timedelta(seconds=n), "minute": timedelta(minutes=n),
                 "hour": timedelta(hours=n), "day": timedelta(days=n)}[unit]
        return now + delta

    # "tomorrow HH:MM" or "tomorrow Xam/pm"
    if w.startswith("tomorrow"):
        base = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        time_part = w.replace("tomorrow", "").strip()
        if time_part:
            return _apply_time(base, time_part)
        return base

    # "today HH:MM" or "today Xam/pm"
    if w.startswith("today"):
        base = now.replace(second=0, microsecond=0)
        time_part = w.replace("today", "").strip()
        if time_part:
            return _apply_time(base, time_part)
        return base

    # Plain time like "3pm", "14:30", "9am"
    dt = _try_parse_time_today(w, now)
    if dt:
        return dt

    # ISO datetime
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(when.strip(), fmt)
        except ValueError:
            pass

    raise ValueError(f"Could not parse time expression: '{when}'")


def _apply_time(base: datetime, time_str: str) -> datetime:
    dt = _try_parse_time_today(time_str, base)
    if dt:
        return base.replace(hour=dt.hour, minute=dt.minute, second=0, microsecond=0)
    return base


def _try_parse_time_today(s: str, base: datetime) -> datetime | None:
    s = s.strip()
    # "3pm", "3:30pm", "15:00", "9am"
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", s)
    if not m:
        return None
    hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


# ── SQLite persistence ────────────────────────────────────────────────────────

def _get_db_path() -> Path:
    from aria.config import get_config
    cfg = get_config()
    return cfg.db_path


def _ensure_table() -> None:
    import sqlite3
    db = _get_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                message   TEXT NOT NULL,
                fire_at   REAL NOT NULL,
                fired     INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()


def _add_reminder(message: str, fire_at: datetime) -> int:
    import sqlite3
    _ensure_table()
    with sqlite3.connect(_get_db_path()) as conn:
        cur = conn.execute(
            "INSERT INTO reminders (message, fire_at) VALUES (?, ?)",
            (message, fire_at.timestamp()),
        )
        conn.commit()
        return cur.lastrowid


def _list_pending() -> list[dict]:
    import sqlite3
    _ensure_table()
    with sqlite3.connect(_get_db_path()) as conn:
        rows = conn.execute(
            "SELECT id, message, fire_at FROM reminders WHERE fired=0 ORDER BY fire_at"
        ).fetchall()
    return [{"id": r[0], "message": r[1], "fire_at": r[2]} for r in rows]


def _mark_fired(reminder_id: int) -> None:
    import sqlite3
    with sqlite3.connect(_get_db_path()) as conn:
        conn.execute("UPDATE reminders SET fired=1 WHERE id=?", (reminder_id,))
        conn.commit()


def _delete_reminder(reminder_id: int) -> bool:
    import sqlite3
    _ensure_table()
    with sqlite3.connect(_get_db_path()) as conn:
        cur = conn.execute(
            "DELETE FROM reminders WHERE id=? AND fired=0", (reminder_id,)
        )
        conn.commit()
        return cur.rowcount > 0


# ── Background scheduler thread ───────────────────────────────────────────────

def _fire(message: str) -> None:
    """Fire a reminder: print to console + TTS."""
    alert = f"Reminder: {message}"
    print(f"\n\n🔔 {alert}\n")
    try:
        from aria.voice.tts import _speak_kokoro, _get_kokoro
        # Pre-load Kokoro so it's ready before speaking
        _get_kokoro()
        _speak_kokoro(alert, block=False)
    except Exception:
        try:
            from aria.voice.tts import speak
            speak(alert, block=False)
        except Exception:
            pass


def _scheduler_loop() -> None:
    while True:
        try:
            now = time.time()
            for r in _list_pending():
                if r["fire_at"] <= now:
                    _fire(r["message"])
                    _mark_fired(r["id"])
        except Exception:
            pass
        time.sleep(5)


_scheduler_started = False
_scheduler_lock = threading.Lock()


def _ensure_scheduler() -> None:
    global _scheduler_started
    with _scheduler_lock:
        if not _scheduler_started:
            t = threading.Thread(target=_scheduler_loop, daemon=True, name="aria-scheduler")
            t.start()
            _scheduler_started = True


# ── Tool functions ────────────────────────────────────────────────────────────

def set_reminder(message: str, when: str) -> str:
    _ensure_scheduler()
    try:
        fire_at = _parse_when(when)
    except ValueError as exc:
        return f"Error parsing time: {exc}"

    now = datetime.now()
    if fire_at <= now:
        return f"Error: '{when}' is in the past ({fire_at.strftime('%Y-%m-%d %H:%M:%S')})."

    rid = _add_reminder(message, fire_at)
    delta = fire_at - now
    mins = int(delta.total_seconds() / 60)
    time_str = fire_at.strftime("%Y-%m-%d %H:%M:%S")

    if mins < 60:
        human = f"in {mins} minute{'s' if mins != 1 else ''}"
    elif mins < 1440:
        hours = mins // 60
        human = f"in {hours} hour{'s' if hours != 1 else ''}"
    else:
        human = f"on {time_str}"

    return f"Reminder #{rid} set: '{message}' — fires {human} ({time_str})."


def list_reminders() -> str:
    _ensure_scheduler()
    pending = _list_pending()
    if not pending:
        return "No pending reminders."
    lines = []
    now = time.time()
    for r in pending:
        dt = datetime.fromtimestamp(r["fire_at"])
        secs = r["fire_at"] - now
        if secs < 3600:
            eta = f"{int(secs/60)}m"
        elif secs < 86400:
            eta = f"{secs/3600:.1f}h"
        else:
            eta = dt.strftime("%b %d %H:%M")
        lines.append(f"  #{r['id']} [{eta}] {r['message']}")
    return "Pending reminders:\n" + "\n".join(lines)


def delete_reminder(reminder_id: int) -> str:
    if _delete_reminder(reminder_id):
        return f"Reminder #{reminder_id} cancelled."
    return f"Reminder #{reminder_id} not found or already fired."
