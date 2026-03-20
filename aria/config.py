"""Configuration — loaded from environment variables / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Anthropic
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("ARIA_MODEL", "claude-sonnet-4-6"))

    # Search
    google_api_key: str = field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""))
    google_cx: str = field(default_factory=lambda: os.environ.get("GOOGLE_CX", ""))
    tavily_api_key: str = field(default_factory=lambda: os.environ.get("TAVILY_API_KEY", ""))

    # Database
    db_path: Path = field(
        default_factory=lambda: Path(os.environ.get("ARIA_DB_PATH", "./data/aria.db"))
    )

    # Memory
    memory_window: int = field(
        default_factory=lambda: int(os.environ.get("ARIA_MEMORY_WINDOW", "20"))
    )

    # Voice
    voice_model: str = field(
        default_factory=lambda: os.environ.get("ARIA_VOICE_MODEL", "small")
    )
    tts_backend: str = field(
        default_factory=lambda: os.environ.get("ARIA_TTS_BACKEND", "auto")
    )  # auto | kokoro | say | pyttsx3 | elevenlabs
    say_voice: str = field(
        default_factory=lambda: os.environ.get("ARIA_SAY_VOICE", "")
    )  # kokoro: af_heart, af_sky, am_puck, bf_emma etc. | macOS: Samantha, Daniel etc.
    say_rate: int = field(
        default_factory=lambda: int(os.environ.get("ARIA_SAY_RATE", "0"))
    )  # words per minute; 0 = system default (~175)

    elevenlabs_api_key: str = field(
        default_factory=lambda: os.environ.get("ELEVENLABS_API_KEY", "")
    )
    elevenlabs_voice_id: str = field(
        default_factory=lambda: os.environ.get("ELEVENLABS_VOICE_ID", "")
    )

    # Proxy
    http_proxy: str = field(default_factory=lambda: os.environ.get("ARIA_HTTP_PROXY", ""))

    # Telegram
    telegram_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_TOKEN", ""))
    telegram_allowed_ids: str = field(default_factory=lambda: os.environ.get("TELEGRAM_ALLOWED_IDS", ""))  # comma-separated chat IDs

    # Web UI
    host: str = field(default_factory=lambda: os.environ.get("ARIA_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("ARIA_PORT", "8000")))

    def validate(self) -> None:
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")


# Module-level singleton
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
