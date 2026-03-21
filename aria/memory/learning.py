"""mem0-backed learning memory for ARIA.

Wraps mem0 with local ChromaDB vector store and HuggingFace embeddings
so no OpenAI key is needed. Used alongside the existing SQLite store —
SQLite handles conversation history, mem0 handles semantic long-term memory.
"""
from __future__ import annotations

import logging
import os
import warnings
from typing import Any

# Suppress noisy library output before any imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*asyncio.iscoroutinefunction.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("BertModel").setLevel(logging.ERROR)

log = logging.getLogger(__name__)

_mem0_instance = None


def _get_mem0() -> Any:
    global _mem0_instance
    if _mem0_instance is not None:
        return _mem0_instance

    import io
    import sys
    from mem0 import Memory
    from aria.config import get_config

    cfg = get_config()

    config: dict = {
        "embedder": {
            "provider": "huggingface",
            "config": {"model": "multi-qa-MiniLM-L6-cos-v1"},
        },
        "llm": {
            "provider": "anthropic",
            "config": {
                "model": "claude-haiku-4-5-20251001",
                "api_key": cfg.anthropic_api_key,
                "temperature": 0.1,
                "top_p": None,  # proxy rejects temperature+top_p together; None → omitted by SDK
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "aria_memories",
                "path": str(cfg.db_path.parent / "chroma"),
            },
        },
    }

    if cfg.http_proxy:
        config["llm"]["config"]["anthropic_base_url"] = cfg.http_proxy

    # Suppress noisy stderr from HuggingFace/ChromaDB during first-time model load
    _old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        _mem0_instance = Memory.from_config(config)
    finally:
        sys.stderr = _old_stderr

    return _mem0_instance


async def add_memories(messages: list[dict], user_id: str) -> None:
    """Extract and store memories from a conversation turn."""
    import asyncio

    def _add():
        try:
            mem = _get_mem0()
            mem.add(messages, user_id=user_id)
        except Exception as e:
            log.warning("mem0 add failed: %s", e)

    await asyncio.get_event_loop().run_in_executor(None, _add)


async def search_memories(query: str, user_id: str, limit: int = 5) -> list[str]:
    """Return relevant memory strings for the given query and user."""
    import asyncio

    def _search():
        try:
            mem = _get_mem0()
            results = mem.search(query, user_id=user_id, limit=limit)
            return [r["memory"] for r in results.get("results", [])]
        except Exception as e:
            log.warning("mem0 search failed: %s", e)
            return []

    return await asyncio.get_event_loop().run_in_executor(None, _search)
