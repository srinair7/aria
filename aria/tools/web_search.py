"""Web search tool — Google Custom Search with DuckDuckGo fallback."""
from __future__ import annotations

import urllib.parse

import httpx

from aria.config import get_config

WEB_SEARCH_DEF = {
    "name": "web_search",
    "description": (
        "Search the web for current information. "
        "Returns a list of results with title, URL, and snippet."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (1-10)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


async def web_search(query: str, num_results: int = 5) -> str:
    cfg = get_config()
    num_results = max(1, min(10, num_results))

    if cfg.google_api_key and cfg.google_cx:
        return await _google_search(query, num_results, cfg.google_api_key, cfg.google_cx)
    return await _ddg_search(query, num_results)


async def _google_search(query: str, n: int, api_key: str, cx: str) -> str:
    params = {"key": api_key, "cx": cx, "q": query, "num": n}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get("https://www.googleapis.com/customsearch/v1", params=params)
        r.raise_for_status()
        data = r.json()

    items = data.get("items", [])
    if not items:
        return "No results found."

    lines = []
    for item in items:
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "").replace("\n", " ")
        lines.append(f"**{title}**\n{link}\n{snippet}")
    return "\n\n".join(lines)


async def _ddg_search(query: str, n: int) -> str:
    """DuckDuckGo instant answer (HTML scrape — no API key needed)."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ARIA/1.0)"}

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        html = r.text

    # Very light extraction — pull result blocks
    results = _parse_ddg_html(html, n)
    if not results:
        return f"No results found for: {query}"
    return "\n\n".join(results)


def _parse_ddg_html(html: str, n: int) -> list[str]:
    """Extract result snippets from DuckDuckGo HTML without dependencies."""
    import re

    results: list[str] = []
    # Each result: <div class="result__body"> ... <a class="result__a" href="...">TITLE</a>
    # ... <a class="result__snippet">SNIPPET</a>
    block_re = re.compile(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet[^"]*"[^>]*>(.*?)</(?:a|span)>',
        re.DOTALL,
    )
    tag_re = re.compile(r"<[^>]+>")

    for m in block_re.finditer(html):
        if len(results) >= n:
            break
        url = tag_re.sub("", m.group(1)).strip()
        title = tag_re.sub("", m.group(2)).strip()
        snippet = tag_re.sub("", m.group(3)).strip()
        results.append(f"**{title}**\n{url}\n{snippet}")

    return results
