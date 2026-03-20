"""Python code execution tool — sandboxed via subprocess."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

RUN_PYTHON_DEF = {
    "name": "run_python",
    "description": (
        "Execute a Python code snippet and return stdout/stderr. "
        "Runs in an isolated subprocess with a 30-second timeout. "
        "Do not use for long-running or network-heavy operations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout": {
                "type": "integer",
                "description": "Max execution time in seconds (default 30, max 60)",
                "default": 30,
            },
        },
        "required": ["code"],
    },
}

_MAX_OUTPUT = 8192  # chars


def run_python(code: str, timeout: int = 30) -> str:
    timeout = max(1, min(60, timeout))
    code = textwrap.dedent(code)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(code)
        tf_path = tf.name

    try:
        result = subprocess.run(
            [sys.executable, tf_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: code execution timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        Path(tf_path).unlink(missing_ok=True)

    stdout = result.stdout[:_MAX_OUTPUT]
    stderr = result.stderr[:_MAX_OUTPUT]

    parts = []
    if stdout:
        parts.append(f"STDOUT:\n{stdout}")
    if stderr:
        parts.append(f"STDERR:\n{stderr}")
    if result.returncode != 0:
        parts.append(f"Return code: {result.returncode}")
    return "\n".join(parts) if parts else "(no output)"
