"""Web search tool — Tavily (primary) with DuckDuckGo fallback."""
from __future__ import annotations

import urllib.parse
import logging

import httpx

from aria.config import get_config

log = logging.getLogger(__name__)

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

    if cfg.tavily_api_key:
        try:
            return await _tavily_search(query, num_results, cfg.tavily_api_key)
        except Exception as e:
            log.warning("Tavily search failed (%s), falling back to DuckDuckGo", e)

    if cfg.google_api_key and cfg.google_cx:
        try:
            result = await _google_search(query, num_results, cfg.google_api_key, cfg.google_cx)
            if result and "No results" not in result:
                return result
        except Exception as e:
            log.warning("Google search failed (%s), falling back to DuckDuckGo", e)

    return await _ddg_search(query, num_results)


async def _tavily_search(query: str, n: int, api_key: str) -> str:
    """Tavily AI search — returns clean summaries designed for LLMs."""
    payload = {
        "query": query,
        "max_results": n,
        "include_answer": True,
        "include_raw_content": False,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post("https://api.tavily.com/search", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()

    lines = []

    # Tavily provides a direct answer summary — prepend it
    answer = data.get("answer", "")
    if answer:
        lines.append(f"Summary: {answer}\n")

    for item in data.get("results", []):
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "").replace("\n", " ")
        lines.append(f"**{title}**\n{url}\n{content}")

    return "\n\n".join(lines) if lines else "No results found."


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
        # Include rich snippet if available
        pagemap = item.get("pagemap", {})
        metatags = pagemap.get("metatags", [{}])[0]
        description = metatags.get("og:description", "") or metatags.get("description", "")
        extra = f"\n{description}" if description and description != snippet else ""
        lines.append(f"**{title}**\n{link}\n{snippet}{extra}")

    return "\n\n".join(lines)


async def _ddg_search(query: str, n: int) -> str:
    """DuckDuckGo search via the lite HTML endpoint."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html",
    }

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        html = r.text

    results = _parse_ddg_lite(html, n)
    if not results:
        # Last resort: try the regular HTML endpoint
        return await _ddg_html_search(query, n)

    return "\n\n".join(results)


def _decode_ddg_url(url: str) -> str:
    """Extract real URL from DDG redirect link."""
    if "uddg=" in url:
        try:
            encoded = url.split("uddg=")[1].split("&")[0]
            return urllib.parse.unquote(encoded)
        except Exception:
            pass
    return url


def _parse_ddg_lite(html: str, n: int) -> list[str]:
    """Parse DuckDuckGo lite results — more stable than full HTML."""
    import re

    results: list[str] = []
    tag_re = re.compile(r"<[^>]+>")

    # DDG lite: results are in <tr> rows with <a class="result-link"> and <td class="result-snippet">
    link_re = re.compile(r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)
    snippet_re = re.compile(r'<td[^>]+class="result-snippet"[^>]*>(.*?)</td>', re.DOTALL)

    links = link_re.findall(html)
    snippets = snippet_re.findall(html)

    for i, (url, title) in enumerate(links[:n]):
        snippet = snippets[i] if i < len(snippets) else ""
        title = tag_re.sub("", title).strip()
        snippet = tag_re.sub("", snippet).strip().replace("\n", " ")
        # Decode DDG redirect URLs to get the real URL
        url = _decode_ddg_url(url)
        if title and url:
            results.append(f"**{title}**\n{url}\n{snippet}")

    return results


async def _ddg_html_search(query: str, n: int) -> str:
    """Fallback: DuckDuckGo full HTML endpoint."""
    import re

    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ARIA/1.0)"}

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
            html = r.text
    except Exception:
        return f"Search unavailable for: {query}"

    tag_re = re.compile(r"<[^>]+>")
    block_re = re.compile(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet[^"]*"[^>]*>(.*?)</(?:a|span)>',
        re.DOTALL,
    )

    results = []
    for m in block_re.finditer(html):
        if len(results) >= n:
            break
        url_r = tag_re.sub("", m.group(1)).strip()
        title = tag_re.sub("", m.group(2)).strip()
        snippet = tag_re.sub("", m.group(3)).strip()
        results.append(f"**{title}**\n{url_r}\n{snippet}")

    return "\n\n".join(results) if results else f"No results found for: {query}"
