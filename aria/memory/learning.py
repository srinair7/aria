"""Lightweight semantic memory for ARIA.

Stores conversation summaries as embeddings in ChromaDB using a local
sentence-transformers model. No LLM call needed for add or search —
just embed + upsert / cosine search. Fast and fully offline.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import sys
import threading
import warnings
from typing import Any

# Suppress noisy library output before any imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*asyncio.iscoroutinefunction.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

log = logging.getLogger(__name__)

_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
_embedder = None
_chroma_client = None
_lock = threading.Lock()


def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is None:
            _old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                _embedder = SentenceTransformer(_MODEL_NAME)
            finally:
                sys.stderr = _old_stderr
    return _embedder


def _get_collection(user_id: str):
    global _chroma_client
    from aria.config import get_config
    import chromadb  # type: ignore

    if _chroma_client is None:
        with _lock:
            if _chroma_client is None:
                cfg = get_config()
                path = str(cfg.db_path.parent / "chroma")
                _old_stderr = sys.stderr
                sys.stderr = io.StringIO()
                try:
                    _chroma_client = chromadb.PersistentClient(path=path)
                finally:
                    sys.stderr = _old_stderr

    # One collection per user; safe name from user_id
    safe = "mem_" + hashlib.md5(user_id.encode()).hexdigest()[:12]
    return _chroma_client.get_or_create_collection(
        name=safe,
        metadata={"hnsw:space": "cosine"},
    )


def _summarise(messages: list[dict]) -> str:
    """Turn a list of messages into a single string to embed and store."""
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(f"{role}: {content.strip()}")
    return "\n".join(parts)


def _add_sync(messages: list[dict], user_id: str) -> None:
    try:
        text = _summarise(messages)
        if not text:
            return
        model = _get_embedder()
        embedding = model.encode(text, normalize_embeddings=True).tolist()
        doc_id = hashlib.md5(text.encode()).hexdigest()
        col = _get_collection(user_id)
        col.upsert(ids=[doc_id], embeddings=[embedding], documents=[text])
    except Exception as e:
        log.warning("memory add failed: %s", e)


def _search_sync(query: str, user_id: str, limit: int) -> list[str]:
    try:
        model = _get_embedder()
        embedding = model.encode(query, normalize_embeddings=True).tolist()
        col = _get_collection(user_id)
        results = col.query(
            query_embeddings=[embedding],
            n_results=min(limit, col.count() or 1),
            include=["documents"],
        )
        docs = results.get("documents", [[]])[0]
        # Return only the most relevant lines from each stored exchange
        snippets = []
        for doc in docs:
            # Take just the assistant's part if present, else the whole thing
            for line in doc.splitlines():
                if line.startswith("assistant:"):
                    snippets.append(line[len("assistant:"):].strip())
                    break
            else:
                snippets.append(doc[:200])
        return snippets
    except Exception as e:
        log.warning("memory search failed: %s", e)
        return []


async def add_memories(messages: list[dict], user_id: str) -> None:
    """Embed and store a conversation exchange."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _add_sync, messages, user_id)


async def search_memories(query: str, user_id: str, limit: int = 5) -> list[str]:
    """Return semantically relevant past responses for the query."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _search_sync, query, user_id, limit)
