"""Text-to-speech: kokoro (default) → macOS say → pyttsx3 → ElevenLabs."""
from __future__ import annotations

import platform
import subprocess
import threading
import tempfile
from pathlib import Path

from aria.config import get_config


def speak(text: str, block: bool = True) -> None:
    """Speak text using the configured TTS backend."""
    from aria.voice.preprocess import preprocess_for_tts
    text = preprocess_for_tts(text)
    if not text:
        return

    cfg = get_config()
    backend = cfg.tts_backend

    if backend == "auto":
        backend = "kokoro"

    if backend == "kokoro":
        _speak_kokoro(text, block=block)
    elif backend == "say":
        _speak_say(text, block=block)
    elif backend == "elevenlabs":
        _speak_elevenlabs(text, block=block)
    else:
        _speak_pyttsx3(text, block=block)


# ── Kokoro (local neural TTS) ─────────────────────────────────────────────────

_kokoro_model = None
_kokoro_lock = threading.Lock()


def _get_kokoro():
    global _kokoro_model
    with _kokoro_lock:
        if _kokoro_model is None:
            from kokoro_onnx import Kokoro  # type: ignore
            from aria.config import get_config
            data_dir = get_config().db_path.parent
            onnx = str(data_dir / "kokoro-v1.0.onnx")
            voices = str(data_dir / "voices-v1.0.bin")
            _kokoro_model = Kokoro(onnx, voices)
    return _kokoro_model


def _speak_kokoro(text: str, block: bool = True) -> None:
    def _run() -> None:
        import sounddevice as sd  # type: ignore
        import numpy as np

        kokoro = _get_kokoro()
        cfg = get_config()
        # Voice: af_heart is warm/natural female; am_puck is male
        voice = cfg.say_voice if cfg.say_voice else "af_heart"
        speed = (cfg.say_rate / 175.0) if cfg.say_rate else 1.0
        speed = max(0.5, min(2.0, speed))

        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
        sd.play(samples, samplerate=sample_rate)
        sd.wait()

    if block:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()


# ── macOS say ─────────────────────────────────────────────────────────────────

_KOKORO_VOICES = {"af_heart", "af_sky", "af_bella", "af_nicole", "am_puck", "am_adam", "am_echo",
                  "bf_emma", "bf_isabella", "bm_george", "bm_lewis"}


def _speak_say(text: str, block: bool = True) -> None:
    cfg = get_config()
    cmd = ["/usr/bin/say"]
    # Only pass voice if it's a real macOS voice (not a Kokoro voice name)
    if cfg.say_voice and cfg.say_voice not in _KOKORO_VOICES:
        cmd += ["-v", cfg.say_voice]
    if cfg.say_rate:
        cmd += ["-r", str(cfg.say_rate)]
    cmd.append(text)

    def _run() -> None:
        subprocess.run(cmd, check=False)

    if block:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()


# ── pyttsx3 ───────────────────────────────────────────────────────────────────

def _speak_pyttsx3(text: str, block: bool = True) -> None:
    try:
        import pyttsx3  # type: ignore
    except ImportError:
        print(f"[TTS fallback] {text}")
        return

    def _run() -> None:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    if block:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()


# ── ElevenLabs ────────────────────────────────────────────────────────────────

def _speak_elevenlabs(text: str, block: bool = True) -> None:
    cfg = get_config()
    if not cfg.elevenlabs_api_key:
        _speak_kokoro(text, block=block)
        return

    def _run() -> None:
        import httpx

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{cfg.elevenlabs_voice_id}"
        headers = {"xi-api-key": cfg.elevenlabs_api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        try:
            r = httpx.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                tf.write(r.content)
                mp3_path = Path(tf.name)
            if platform.system() == "Darwin":
                subprocess.run(["afplay", str(mp3_path)], check=False)
            else:
                subprocess.run(["mpg123", str(mp3_path)], check=False)
            mp3_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[ElevenLabs TTS error: {exc}]")

    if block:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()
