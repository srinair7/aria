"""Text preprocessing before TTS — strip/replace emojis and clean up formatting."""
from __future__ import annotations

import re

# Emojis that map to a spoken expression
_EXPRESSIVE: dict[str, str] = {
    "😂": "haha",
    "🤣": "haha",
    "😄": "ha",
    "😁": "ha",
    "😊": "",
    "😍": "",
    "😅": "heh",
    "😆": "haha",
    "🙈": "oh no",
    "😬": "yikes",
    "😮": "oh",
    "😲": "wow",
    "😱": "oh wow",
    "🤔": "hmm",
    "🤗": "",
    "😏": "",
    "😎": "",
    "🥲": "",
    "😢": "",
    "😭": "aw",
    "😤": "",
    "🤦": "ugh",
    "🙄": "",
    "👍": "sure",
    "👋": "hey",
    "🙏": "please",
    "❤️": "",
    "💪": "",
    "🎉": "",
    "✨": "",
    "🔥": "",
    "💡": "",
    "⚠️": "warning",
    "❌": "",
    "✅": "",
    "☑️": "",
    "🔔": "",
    "📅": "",
    "📌": "",
    "📝": "",
    "🗓️": "",
    "⭐": "",
    "🌟": "",
    "🚀": "",
    "💬": "",
    "🔴": "",
    "🟡": "",
    "🟢": "",
    "🔹": "",
    "🔸": "",
    "▪️": "",
    "•": "",
}

# Regex to match any emoji not in our expressive map (strip silently)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U00002600-\U000026FF"  # misc symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols etc
    "\U0001FA70-\U0001FAFF"  # more symbols
    "\U00002300-\U000023FF"  # technical
    "\uFE00-\uFE0F"          # variation selectors
    "\u200d"                 # zero width joiner
    "\u20E3"                 # combining enclosing keycap
    "]+",
    flags=re.UNICODE,
)

# Markdown patterns to clean up
_MD_BOLD = re.compile(r"\*\*(.*?)\*\*")
_MD_ITALIC = re.compile(r"\*(.*?)\*|_(.*?)_")
_MD_CODE = re.compile(r"`[^`]+`")
_MD_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
_MD_NUMBERED = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_MD_URL = re.compile(r"https?://\S+")


def preprocess_for_tts(text: str) -> str:
    """Clean text for natural TTS output."""

    # Strip fenced code blocks entirely
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)

    # Strip horizontal rules
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Strip entire markdown table rows (lines containing |)
    text = re.sub(r"^[^\n]*\|[^\n]*$", "", text, flags=re.MULTILINE)

    # Replace expressive emojis with spoken equivalents
    for emoji, replacement in _EXPRESSIVE.items():
        if replacement:
            text = text.replace(emoji, f" {replacement} ")
        else:
            text = text.replace(emoji, " ")

    # Strip remaining emojis
    text = _EMOJI_RE.sub(" ", text)

    # Clean markdown inline formatting
    text = _MD_BOLD.sub(r"\1", text)         # **bold** → bold
    text = _MD_ITALIC.sub(r"\1\2", text)     # *italic* → italic
    text = _MD_HEADER.sub("", text)          # ## Header → Header
    text = _MD_LINK.sub(r"\1", text)         # [text](url) → text
    text = _MD_URL.sub("", text)             # bare URLs → removed

    # Convert bullet/numbered list items: strip the marker, join with comma or period
    # First pass: collect lines, detect if they're list items
    lines = text.splitlines()
    result_lines = []
    list_items = []
    for line in lines:
        stripped = line.strip()
        is_bullet = re.match(r"^[-*•]\s+(.+)", stripped)
        is_numbered = re.match(r"^\d+\.\s+(.+)", stripped)
        if is_bullet or is_numbered:
            content = (is_bullet or is_numbered).group(1).strip()
            # Remove any inline label like "**High:** " or "🌡️ **High:**"
            content = re.sub(r"^[\W\s]*\*\*[^*]+\*\*\s*:?\s*", "", content)
            content = re.sub(r"^\w[\w\s]+:\s*", "", content)
            if content:
                list_items.append(content)
        else:
            if list_items:
                result_lines.append(". ".join(list_items) + ".")
                list_items = []
            if stripped:
                result_lines.append(stripped)
    if list_items:
        result_lines.append(". ".join(list_items) + ".")

    text = " ".join(result_lines)

    # Strip leftover markdown punctuation: standalone **, *, _, ---, ---
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"_{2,}", "", text)
    text = re.sub(r"#+", "", text)

    # Deduplicate consecutive identical words
    text = re.sub(r"\b(\w+)(\s+\1)+\b", r"\1", text, flags=re.IGNORECASE)

    # Improve prosody: add a brief pause before question marks and exclamations
    # so Kokoro has room for natural intonation
    text = re.sub(r"\s*\?", "...?", text)
    text = re.sub(r"\s*!", "...!", text)

    # Expand dashes and em-dashes into a natural pause
    text = re.sub(r"\s*—\s*", ", ", text)
    text = re.sub(r"\s*--\s*", ", ", text)

    # Collapse whitespace
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()

    return text
