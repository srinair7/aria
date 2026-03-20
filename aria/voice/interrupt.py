"""Spacebar-interruptible TTS playback.

Usage:
    player = InterruptiblePlayer()
    interrupted = player.speak_and_listen(text)
    if interrupted:
        # user pressed space — record their next message via normal STT
"""
from __future__ import annotations

import sys
import threading


class InterruptiblePlayer:
    """Plays TTS audio; pressing Space stops playback."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()

    def speak_and_listen(self, text: str) -> bool:
        """
        Speak text. Returns True if user pressed Space to interrupt, False otherwise.
        """
        from aria.voice.tts import _get_kokoro, _speak_say
        from aria.voice.preprocess import preprocess_for_tts
        from aria.config import get_config
        import sounddevice as sd

        text = preprocess_for_tts(text)
        if not text:
            return False

        cfg = get_config()
        self._stop_event.clear()

        try:
            kokoro = _get_kokoro()
            voice = cfg.say_voice or "af_heart"
            speed = (cfg.say_rate / 175.0) if cfg.say_rate else 1.0
            speed = max(0.5, min(2.0, speed))
            samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
        except Exception:
            _speak_say(text, block=True)
            return False

        # Start spacebar listener thread
        key_thread = threading.Thread(target=self._key_listener, daemon=True)
        key_thread.start()

        # Play audio in chunks, checking stop_event
        chunk_size = int(sample_rate * 0.1)
        pos = 0
        interrupted = False

        with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
            while pos < len(samples):
                if self._stop_event.is_set():
                    interrupted = True
                    break
                chunk = samples[pos: pos + chunk_size]
                if len(chunk) == 0:
                    break
                stream.write(chunk.reshape(-1, 1))
                pos += chunk_size

        self._stop_event.set()
        key_thread.join(timeout=1)
        return interrupted

    def _key_listener(self) -> None:
        """Set stop_event when Space is pressed."""
        try:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while not self._stop_event.is_set():
                    import select
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r:
                        ch = sys.stdin.read(1)
                        if ch in (" ", "\r", "\n", "q"):
                            self._stop_event.set()
                            break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
