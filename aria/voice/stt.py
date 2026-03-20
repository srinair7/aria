"""Speech-to-text using OpenAI Whisper + sounddevice."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd

from aria.config import get_config

_SAMPLE_RATE = 16000
_CHANNELS = 1
_SILENCE_THRESHOLD = 0.01  # RMS threshold for end-of-speech detection
_SILENCE_DURATION = 1.5  # seconds of silence before stopping
_MAX_DURATION = 30.0  # seconds maximum recording


def transcribe_mic() -> str:
    """Record from microphone until silence, then transcribe with Whisper."""
    import whisper  # type: ignore

    cfg = get_config()
    model = _load_model(cfg.voice_model)

    audio = _record_until_silence()
    if audio is None or len(audio) == 0:
        return ""

    # Save to temp WAV for Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf_path = Path(tf.name)

    try:
        _save_wav(tf_path, audio)
        result = model.transcribe(str(tf_path), language="en", fp16=False)
        return result.get("text", "").strip()
    finally:
        tf_path.unlink(missing_ok=True)


def transcribe_bytes(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """Transcribe audio from raw bytes (used by web UI)."""
    import whisper  # type: ignore

    cfg = get_config()
    model = _load_model(cfg.voice_model)

    suffix = ".wav"
    if "webm" in mime_type:
        suffix = ".webm"
    elif "mp4" in mime_type or "m4a" in mime_type:
        suffix = ".m4a"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(audio_bytes)
        tf_path = Path(tf.name)

    try:
        result = model.transcribe(str(tf_path), fp16=False)
        return result.get("text", "").strip()
    finally:
        tf_path.unlink(missing_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

_whisper_cache: dict[str, object] = {}


def _load_model(size: str) -> object:
    import whisper  # type: ignore

    if size not in _whisper_cache:
        _whisper_cache[size] = whisper.load_model(size)
    return _whisper_cache[size]


def _record_until_silence() -> np.ndarray | None:
    """Record audio from default mic, stop on prolonged silence."""
    block_size = int(_SAMPLE_RATE * 0.1)  # 100ms chunks
    max_blocks = int(_MAX_DURATION / 0.1)
    silence_blocks = int(_SILENCE_DURATION / 0.1)

    frames: list[np.ndarray] = []
    consecutive_silence = 0
    recording_started = False

    with sd.InputStream(samplerate=_SAMPLE_RATE, channels=_CHANNELS, dtype="float32") as stream:
        for _ in range(max_blocks):
            block, _ = stream.read(block_size)
            rms = float(np.sqrt(np.mean(block**2)))

            if rms > _SILENCE_THRESHOLD:
                recording_started = True
                consecutive_silence = 0
            elif recording_started:
                consecutive_silence += 1

            frames.append(block.copy())

            if recording_started and consecutive_silence >= silence_blocks:
                break

    if not frames:
        return None
    return np.concatenate(frames, axis=0).flatten()


def _save_wav(path: Path, audio: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> None:
    import wave
    import struct

    # Convert float32 to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
