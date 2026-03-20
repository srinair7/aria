"""Telegram bot integration for ARIA using python-telegram-bot."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def _allowed(chat_id: int) -> bool:
    """Return True if chat_id is in the allowlist (or allowlist is empty = open)."""
    from aria.config import get_config
    raw = get_config().telegram_allowed_ids.strip()
    if not raw:
        return True
    allowed = {s.strip() for s in raw.split(",")}
    return str(chat_id) in allowed


async def _get_agent(chat_id: int):
    from aria.agent import Agent
    return Agent(session_id=f"telegram_{chat_id}")


# ── Handlers ──────────────────────────────────────────────────────────────────

async def handle_start(update, context) -> None:
    await update.message.reply_text("Hey! What's up?")


async def handle_text(update, context) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        await update.message.reply_text("Sorry, you're not authorised to use ARIA.")
        return

    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    agent = await _get_agent(chat_id)
    chunks: list[str] = []
    try:
        async for chunk in agent.stream(user_text):
            chunks.append(chunk)
    except Exception as exc:
        await update.message.reply_text(f"Error: {exc}")
        return

    response = "".join(chunks)
    # Strip tool-call annotation lines
    response = "\n".join(
        ln for ln in response.splitlines()
        if not ln.startswith("[Calling tool:")
    ).strip()

    if not response:
        response = "Done."

    # Telegram max message length is 4096
    for part in _split(response):
        await update.message.reply_text(part, parse_mode="Markdown")


async def handle_voice(update, context) -> None:
    """Voice message → Whisper STT → agent → reply."""
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return

    try:
        from aria.voice.stt import transcribe_bytes
    except ImportError:
        await update.message.reply_text("Voice dependencies not installed (`pip install 'aria[voice]'`).")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tf:
        tmp_path = Path(tf.name)

    try:
        await file.download_to_drive(str(tmp_path))
        audio_bytes = tmp_path.read_bytes()
        user_text = transcribe_bytes(audio_bytes, mime_type="audio/ogg")
    finally:
        tmp_path.unlink(missing_ok=True)

    if not user_text:
        await update.message.reply_text("Sorry, I couldn't make out what you said.")
        return

    # Show transcript
    await update.message.reply_text(f"_{user_text}_", parse_mode="Markdown")
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    agent = await _get_agent(chat_id)
    chunks: list[str] = []
    try:
        async for chunk in agent.stream(user_text):
            chunks.append(chunk)
    except Exception as exc:
        await update.message.reply_text(f"Error: {exc}")
        return

    response = "".join(chunks)
    response = "\n".join(
        ln for ln in response.splitlines()
        if not ln.startswith("[Calling tool:")
    ).strip()

    for part in _split(response):
        await update.message.reply_text(part, parse_mode="Markdown")


async def handle_photo(update, context) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return

    caption = update.message.caption or ""
    user_text = f"[User sent a photo. Caption: '{caption}']" if caption else "[User sent a photo]"
    await _reply_agent(update, context, chat_id, user_text)


async def handle_document(update, context) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return

    doc = update.message.document
    caption = update.message.caption or ""
    user_text = f"[User sent a file: {doc.file_name}. Caption: '{caption}']"
    await _reply_agent(update, context, chat_id, user_text)


async def handle_location(update, context) -> None:
    chat_id = update.effective_chat.id
    if not _allowed(chat_id):
        return

    loc = update.message.location
    user_text = f"My current location: latitude {loc.latitude}, longitude {loc.longitude}"
    await _reply_agent(update, context, chat_id, user_text)


async def _reply_agent(update, context, chat_id: int, user_text: str) -> None:
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    agent = await _get_agent(chat_id)
    chunks: list[str] = []
    try:
        async for chunk in agent.stream(user_text):
            chunks.append(chunk)
    except Exception as exc:
        await update.message.reply_text(f"Error: {exc}")
        return
    response = "".join(chunks)
    response = "\n".join(
        ln for ln in response.splitlines()
        if not ln.startswith("[Calling tool:")
    ).strip()
    for part in _split(response or "Done."):
        await update.message.reply_text(part, parse_mode="Markdown")


def _split(text: str, max_len: int = 4000) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


# ── Bot runner ────────────────────────────────────────────────────────────────

def run_bot(token: str) -> None:
    """Start the Telegram bot (blocking, uses polling)."""
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        filters,
    )

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("help", handle_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.LOCATION, handle_location))

    log.info("ARIA Telegram bot started. Polling...")
    app.run_polling(drop_pending_updates=True)
