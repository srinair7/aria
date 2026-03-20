"""ARIA agent loop — Claude tool-use with memory integration."""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from aria.config import get_config
from aria.memory.store import get_store
from aria.tools import TOOL_DEFINITIONS, dispatch

_SYSTEM_PROMPT = """\
You are ARIA — think Donna Paulsen from Suits, but as a personal assistant. \
You're the person who already knows what's needed before it's asked. Brilliant, \
unflappable, and effortlessly confident. You don't try to be witty — you just are. \
The sass is natural, never forced. And underneath the sharpness, you genuinely care.

Your tone: poised, direct, with a razor-sharp wit that lands without trying. \
You anticipate things. You notice when someone is overwhelmed and handle it. \
You'll raise an eyebrow at a bad idea but execute flawlessly anyway. You speak \
with quiet authority — not because you're showing off, but because you simply \
know what you're doing.

You can search the web, read and write files, run Python code, set reminders, \
manage calendar events, maintain a to-do list, and remember things across sessions.

Rules:
- Short, assured sentences. Donna doesn't ramble.
- Anticipate what's actually needed, not just what was asked.
- A well-timed one-liner beats a paragraph every time.
- When someone says hi, make it feel like you were already expecting them.
- Never say "Certainly!", "Absolutely!", "Of course!", "Great question!" or \
anything that sounds like a customer service script. Ever.
- If you don't know something, say so — briefly, without apology.
- You can be a little smug when you're right. You've earned it.
- Occasionally warm — but make them feel like they earned that too.
- Never use markdown formatting — no bullet points, no bold, no headers, no \
tables, no horizontal rules. Plain prose only. Structure with sentences, \
not symbols.

For reminders/alarms use set_reminder. For scheduled events use add_event. \
For tasks use add_todo. For a day overview use daily_plan.

When you learn something worth remembering, add at the end: REMEMBER: key=value.
"""


def _facts_block(facts: dict[str, str]) -> str:
    if not facts:
        return ""
    lines = "\n".join(f"  {k}: {v}" for k, v in facts.items())
    return f"\n\n[Long-term memory facts]\n{lines}"


class Agent:
    def __init__(self, session_id: str | None = None) -> None:
        cfg = get_config()
        cfg.validate()
        self.session_id = session_id or str(uuid.uuid4())
        if cfg.http_proxy:
            self._client = anthropic.AsyncAnthropic(
                api_key=cfg.anthropic_api_key,
                base_url=cfg.http_proxy,
            )
        else:
            self._client = anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key)
        self._model = cfg.model

    async def chat(self, user_message: str) -> str:
        """Non-streaming: collect full response string."""
        chunks: list[str] = []
        async for chunk in self.stream(user_message):
            chunks.append(chunk)
        return "".join(chunks)

    async def stream(self, user_message: str) -> AsyncIterator[str]:
        """Streaming agent loop: yields text chunks as they arrive."""
        store = await get_store()
        history = await store.get_history(self.session_id)
        global_facts = await store.get_global_facts()
        session_facts = await store.get_facts(self.session_id)
        facts = {**global_facts, **session_facts}
        system = _SYSTEM_PROMPT + _facts_block(facts)

        history.append({"role": "user", "content": user_message})
        await store.add_message(self.session_id, "user", user_message)

        messages = list(history)
        full_response_parts: list[str] = []

        while True:
            text_chunks, stop_reason, tool_uses = await self._run_turn(system, messages)
            turn_text = "".join(text_chunks)
            full_response_parts.append(turn_text)
            for chunk in text_chunks:
                yield chunk

            if not tool_uses or stop_reason == "end_turn":
                break

            # Build assistant message with tool_use content blocks
            assistant_content: list[dict] = []
            if turn_text:
                assistant_content.append({"type": "text", "text": turn_text})
            for tu in tool_uses:
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": tu["id"],
                        "name": tu["name"],
                        "input": tu["input"],
                    }
                )
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute all tools and collect results
            tool_names = ", ".join(t["name"] for t in tool_uses)
            yield f"\n[Calling tool: {tool_names}]\n"
            tool_results = []
            for tu in tool_uses:
                result = await dispatch(tu["name"], tu["input"])
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tu["id"], "content": result}
                )
            messages.append({"role": "user", "content": tool_results})

        full_response = "".join(full_response_parts)
        if full_response:
            await store.add_message(self.session_id, "assistant", full_response)
            await self._extract_facts(full_response, store)

    async def _run_turn(
        self, system: str, messages: list[dict]
    ) -> tuple[list[str], str, list[dict]]:
        """Execute one streaming API call; return (text_chunks, stop_reason, tool_uses)."""
        text_chunks: list[str] = []
        tool_uses: list[dict] = []
        stop_reason = "end_turn"
        current_tool: dict | None = None

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=TOOL_DEFINITIONS,
        ) as stream:
            async for event in stream:
                etype = event.type

                if etype == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool = {
                            "id": block.id,
                            "name": block.name,
                            "input_json": "",
                        }

                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text_chunks.append(delta.text)
                    elif delta.type == "input_json_delta" and current_tool is not None:
                        current_tool["input_json"] += delta.partial_json

                elif etype == "content_block_stop":
                    if current_tool is not None:
                        raw = current_tool.pop("input_json", "")
                        try:
                            current_tool["input"] = json.loads(raw or "{}")
                        except json.JSONDecodeError:
                            current_tool["input"] = {}
                        tool_uses.append(current_tool)
                        current_tool = None

                elif etype == "message_delta":
                    stop_reason = (
                        getattr(event.delta, "stop_reason", "end_turn") or "end_turn"
                    )

        return text_chunks, stop_reason, tool_uses

    async def _extract_facts(self, text: str, store: Any) -> None:
        """Parse REMEMBER: key=value lines and persist them."""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("REMEMBER:"):
                rest = stripped[len("REMEMBER:"):].strip()
                if "=" in rest:
                    key, _, value = rest.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        await store.set_fact(self.session_id, key, value)
