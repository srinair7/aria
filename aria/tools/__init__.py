"""Tool registry — exports TOOL_DEFINITIONS list and dispatch()."""
from aria.tools.web_search import web_search, WEB_SEARCH_DEF
from aria.tools.file_io import read_file, write_file, list_dir, READ_FILE_DEF, WRITE_FILE_DEF, LIST_DIR_DEF
from aria.tools.code_exec import run_python, RUN_PYTHON_DEF
from aria.tools.reminders import (
    set_reminder, list_reminders, delete_reminder,
    SET_REMINDER_DEF, LIST_REMINDERS_DEF, DELETE_REMINDER_DEF,
)
from aria.tools.calendar import (
    add_event, list_events, delete_event,
    add_todo, list_todos, complete_todo, daily_plan,
    ADD_EVENT_DEF, LIST_EVENTS_DEF, DELETE_EVENT_DEF,
    ADD_TODO_DEF, LIST_TODOS_DEF, COMPLETE_TODO_DEF, DAILY_PLAN_DEF,
)

TOOL_DEFINITIONS = [
    WEB_SEARCH_DEF,
    READ_FILE_DEF,
    WRITE_FILE_DEF,
    LIST_DIR_DEF,
    RUN_PYTHON_DEF,
    SET_REMINDER_DEF,
    LIST_REMINDERS_DEF,
    DELETE_REMINDER_DEF,
    ADD_EVENT_DEF,
    LIST_EVENTS_DEF,
    DELETE_EVENT_DEF,
    ADD_TODO_DEF,
    LIST_TODOS_DEF,
    COMPLETE_TODO_DEF,
    DAILY_PLAN_DEF,
]

_DISPATCH = {
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "run_python": run_python,
    "set_reminder": set_reminder,
    "list_reminders": list_reminders,
    "delete_reminder": delete_reminder,
    "add_event": add_event,
    "list_events": list_events,
    "delete_event": delete_event,
    "add_todo": add_todo,
    "list_todos": list_todos,
    "complete_todo": complete_todo,
    "daily_plan": daily_plan,
}


async def dispatch(name: str, inputs: dict) -> str:
    """Call a tool by name and return its string result."""
    fn = _DISPATCH.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    import inspect
    if inspect.iscoroutinefunction(fn):
        return await fn(**inputs)
    return fn(**inputs)
