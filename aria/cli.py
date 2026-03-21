"""ARIA CLI — `aria chat`, `aria serve`, `aria voice`."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import uuid
import warnings

# ── Suppress noisy library warnings before any imports ────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*asyncio.iscoroutinefunction.*", category=DeprecationWarning)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

app = typer.Typer(
    name="aria",
    help="ARIA — Adaptive Reasoning & Intelligence Assistant",
    add_completion=False,
)
console = Console()

_BANNER = """[bold cyan]
   ╔═══════════════════════════════════════╗
   ║  ARIA — Adaptive Reasoning &          ║
   ║         Intelligence Assistant        ║
   ╚═══════════════════════════════════════╝
[/bold cyan]
[dim]Type your message and press Enter. Type [bold]exit[/bold] or [bold]quit[/bold] to stop.[/dim]
"""


@app.command()
def chat(
    session: str = typer.Option(
        None, "--session", "-s", help="Session ID (creates new if omitted)"
    ),
    model: str = typer.Option(None, "--model", "-m", help="Override Claude model"),
) -> None:
    """Start an interactive text chat session with ARIA."""
    from aria.config import get_config

    cfg = get_config()
    try:
        cfg.validate()
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1)

    if model:
        cfg.model = model

    session_id = session or str(uuid.uuid4())
    console.print(_BANNER)
    console.print(f"[dim]Session: {session_id}[/dim]\n")

    asyncio.run(_chat_loop(session_id))


async def _chat_loop(session_id: str) -> None:
    from aria.agent import Agent
    from aria.memory.store import get_store
    from aria.memory.learning import _get_embedder

    agent = Agent(session_id=session_id)

    # Warm up embedding model in the background so first response isn't slow
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _get_embedder)

    try:
        await _chat_loop_inner(agent)
    finally:
        store = await get_store()
        await store.close()


async def _chat_loop_inner(agent: "Agent") -> None:
    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if user_input.strip().lower() in {"exit", "quit", "bye", "goodbye"}:
            console.print("[dim]ARIA: Goodbye! Have a great day.[/dim]")
            break

        if not user_input.strip():
            continue

        # Stream and render response
        console.print("[bold cyan]ARIA:[/bold cyan] ", end="")
        response_parts: list[str] = []
        try:
            async for chunk in agent.stream(user_input):
                console.print(chunk, end="", markup=False)
                response_parts.append(chunk)
            console.print()  # newline after response
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/dim]")
        except Exception as exc:
            console.print(f"\n[red]Error:[/red] {exc}")

        console.print()


@app.command()
def serve(
    host: str = typer.Option(None, "--host", help="Bind host"),
    port: int = typer.Option(None, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
) -> None:
    """Start the ARIA web UI server."""
    from aria.config import get_config

    cfg = get_config()
    try:
        cfg.validate()
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1)

    _host = host or cfg.host
    _port = port or cfg.port

    console.print(f"[bold cyan]Starting ARIA web UI on http://{_host}:{_port}[/bold cyan]")
    import uvicorn
    from aria.ui.app import create_app

    uvicorn.run(
        "aria.ui.app:create_app",
        host=_host,
        port=_port,
        reload=reload,
        factory=True,
        log_level="info",
    )


@app.command()
def voice(
    session: str = typer.Option(
        None, "--session", "-s", help="Session ID (creates new if omitted)"
    ),
) -> None:
    """Start a voice chat session (speak → ARIA responds aloud)."""
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        console.print(
            "[red]Voice dependencies not installed.[/red]\n"
            "Run: [bold]pip install 'aria[voice]'[/bold]"
        )
        raise typer.Exit(1)

    from aria.config import get_config

    cfg = get_config()
    try:
        cfg.validate()
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1)

    session_id = session or str(uuid.uuid4())
    console.print(_BANNER)
    console.print(f"[dim]Voice session: {session_id}[/dim]")
    console.print("[dim]Say 'goodbye' to end the session. Press [bold]Space[/bold] to interrupt ARIA while she's speaking.[/dim]\n")

    asyncio.run(_voice_loop(session_id))


async def _voice_loop(session_id: str) -> None:
    from aria.agent import Agent
    from aria.memory.store import get_store
    from aria.voice.stt import transcribe_mic
    from aria.voice.tts import speak, _get_kokoro
    from aria.voice.stt import _load_model
    from aria.config import get_config

    console.print("[dim]Loading voice models...[/dim]", end="\r")
    try:
        kokoro = _get_kokoro()
        _load_model(get_config().voice_model)
        # Warm up Kokoro JIT with a silent inference
        cfg = get_config()
        voice = cfg.say_voice or "af_heart"
        kokoro.create("Ready.", voice=voice, speed=1.0, lang="en-us")
    except Exception:
        pass
    console.print("[dim]                      [/dim]", end="\r")

    agent = Agent(session_id=session_id)

    try:
        await _voice_loop_inner(agent, transcribe_mic, speak)
    finally:
        store = await get_store()
        await store.close()


async def _voice_loop_inner(agent: "Agent", transcribe_mic, speak) -> None:
    from aria.voice.interrupt import InterruptiblePlayer
    from aria.voice.preprocess import preprocess_for_tts

    loop = asyncio.get_event_loop()

    while True:
        console.print("[bold green]Listening...[/bold green]")
        try:
            text = transcribe_mic()
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye.[/dim]")
            break
        except Exception as exc:
            console.print(f"[red]STT error:[/red] {exc}")
            continue

        if not text:
            continue

        console.print(f"[bold green]You:[/bold green] {text}")

        if text.strip().lower() in {"goodbye", "exit", "quit", "stop"}:
            response = "Goodbye! Have a great day."
            console.print(f"[bold cyan]ARIA:[/bold cyan] {response}")
            speak(response)
            break

        # Stream and speak concurrently
        response_parts: list[str] = []
        console.print("[bold cyan]ARIA:[/bold cyan] ", end="")
        try:
            interrupted = await _stream_and_speak(agent, text, response_parts, loop)
            console.print()
        except Exception as exc:
            console.print(f"\n[red]Error:[/red] {exc}")
            continue

        if interrupted:
            console.print("[dim]Interrupted — listening...[/dim]")
            try:
                interrupt_text = transcribe_mic()
            except Exception:
                interrupt_text = None
            if interrupt_text:
                console.print(f"[bold green]You:[/bold green] {interrupt_text}")
                if interrupt_text.strip().lower() in {"goodbye", "exit", "quit", "stop"}:
                    speak("Okay, goodbye!")
                    return
                follow_parts: list[str] = []
                console.print("[bold cyan]ARIA:[/bold cyan] ", end="")
                try:
                    await _stream_and_speak(agent, interrupt_text, follow_parts, loop)
                    console.print()
                except Exception as exc:
                    console.print(f"\n[red]Error:[/red] {exc}")

        console.print()


async def _stream_and_speak(agent, user_text: str, parts: list, loop) -> bool:
    """Stream agent response, pipeline TTS generation and playback to minimise latency.

    Sentence N+1 is generated while sentence N is playing, eliminating gaps.
    Returns True if user pressed Space to interrupt, False otherwise.
    """
    import threading
    import queue as qmod
    from aria.voice.preprocess import preprocess_for_tts
    from aria.voice.tts import _get_kokoro, _speak_say
    from aria.config import get_config
    import sounddevice as sd
    import sys, select, tty, termios

    cfg = get_config()
    _SENTENCE_END = re.compile(r'(?<=[.!?])\s+|(?<=[,;—])\s+')

    try:
        kokoro = _get_kokoro()
    except Exception:
        kokoro = None

    voice = cfg.say_voice or "af_heart"
    speed = (cfg.say_rate / 175.0) if cfg.say_rate else 1.0
    speed = max(0.5, min(2.0, speed))

    # audio_q holds pre-generated (samples, sr) tuples ready to play
    audio_q: qmod.Queue = qmod.Queue(maxsize=3)
    text_q: qmod.Queue = qmod.Queue()
    stop_flag = threading.Event()
    interrupted_flag = threading.Event()

    def generate_worker():
        """Generate audio for each sentence as text arrives."""
        while not stop_flag.is_set():
            try:
                sentence = text_q.get(timeout=0.2)
            except qmod.Empty:
                continue
            if sentence is None:
                audio_q.put(None)
                break
            if not kokoro:
                continue
            try:
                samples, sr = kokoro.create(sentence, voice=voice, speed=speed, lang="en-us")
                audio_q.put((samples, sr))
            except Exception:
                pass

    gen_thread = threading.Thread(target=generate_worker, daemon=True)
    gen_thread.start()

    # Set terminal raw mode for spacebar detection
    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
    except Exception:
        old_settings = None

    def play_worker():
        """Play audio chunks as they become available."""
        while not stop_flag.is_set():
            try:
                item = audio_q.get(timeout=0.2)
            except qmod.Empty:
                continue
            if item is None:
                break
            samples, sr = item
            chunk_size = int(sr * 0.1)
            pos = 0
            try:
                with sd.OutputStream(samplerate=sr, channels=1, dtype="float32") as stream:
                    while pos < len(samples) and not stop_flag.is_set():
                        try:
                            r, _, _ = select.select([sys.stdin], [], [], 0)
                            if r:
                                ch = sys.stdin.read(1)
                                if ch in (" ", "q"):
                                    interrupted_flag.set()
                                    stop_flag.set()
                                    break
                        except Exception:
                            pass
                        chunk = samples[pos:pos + chunk_size]
                        if len(chunk) == 0:
                            break
                        stream.write(chunk.reshape(-1, 1))
                        pos += chunk_size
            except Exception:
                pass

    play_thread = threading.Thread(target=play_worker, daemon=True)
    play_thread.start()

    # Stream text and feed sentences to generator
    buffer = ""
    async for chunk in agent.stream(user_text):
        if "[Calling tool:" in chunk:
            continue
        console.print(chunk, end="", markup=False)
        parts.append(chunk)
        buffer += chunk

        sentences = _SENTENCE_END.split(buffer)
        if len(sentences) > 1:
            for s in sentences[:-1]:
                clean = preprocess_for_tts(s.strip())
                if clean:
                    text_q.put(clean)
            buffer = sentences[-1]

    if buffer.strip():
        clean = preprocess_for_tts(buffer.strip())
        if clean:
            text_q.put(clean)

    text_q.put(None)  # signal end to generator

    gen_thread.join(timeout=60)
    play_thread.join(timeout=60)
    stop_flag.set()

    if old_settings is not None:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

    return interrupted_flag.is_set()


@app.command()
def voices(
    preview: bool = typer.Option(False, "--preview", "-p", help="Speak a sample with each voice"),
    set_voice: str = typer.Option(None, "--set", "-s", help="Set voice by name and save to .env"),
    rate: int = typer.Option(None, "--rate", "-r", help="Set speaking rate (words/min, e.g. 150-220)"),
) -> None:
    """List available macOS voices, preview them, or set the active voice."""
    import subprocess, re
    from pathlib import Path

    result = subprocess.run(["/usr/bin/say", "-v", "?"], capture_output=True, text=True)
    lines = [l for l in result.stdout.splitlines() if "en_" in l]

    if set_voice or rate is not None:
        # Validate voice name
        if set_voice:
            names = [re.split(r"\s{2,}", l.strip())[0].strip() for l in lines]
            match = next((n for n in names if n.lower() == set_voice.lower()), None)
            if not match:
                console.print(f"[red]Voice '{set_voice}' not found.[/red] Run [bold]aria voices[/bold] to see available voices.")
                raise typer.Exit(1)
            set_voice = match

        # Update .env
        env_path = Path(".env")
        env_text = env_path.read_text() if env_path.exists() else ""

        if set_voice:
            if "ARIA_SAY_VOICE=" in env_text:
                env_text = re.sub(r"ARIA_SAY_VOICE=.*", f"ARIA_SAY_VOICE={set_voice}", env_text)
            else:
                env_text += f"\nARIA_SAY_VOICE={set_voice}"
            console.print(f"[green]Voice set to:[/green] {set_voice}")

        if rate is not None:
            if "ARIA_SAY_RATE=" in env_text:
                env_text = re.sub(r"ARIA_SAY_RATE=.*", f"ARIA_SAY_RATE={rate}", env_text)
            else:
                env_text += f"\nARIA_SAY_RATE={rate}"
            console.print(f"[green]Rate set to:[/green] {rate} wpm")

        env_path.write_text(env_text)

        # Also update live config
        from aria.config import get_config
        cfg = get_config()
        if set_voice:
            cfg.say_voice = set_voice
        if rate is not None:
            cfg.say_rate = rate

        if set_voice:
            console.print(f"[dim]Preview:[/dim]")
            subprocess.run(["/usr/bin/say", "-v", set_voice, f"Hello! I'm ARIA, using the {set_voice} voice."], check=False)
        return

    # List voices
    from aria.config import get_config
    current = get_config().say_voice or "(system default)"
    console.print(f"\n[bold cyan]Available English voices[/bold cyan]  [dim](current: {current})[/dim]\n")

    for line in lines:
        parts = re.split(r"\s{2,}", line.strip())
        name = parts[0].strip()
        locale = parts[1].strip() if len(parts) > 1 else ""
        sample = parts[2].strip().lstrip("# ") if len(parts) > 2 else ""
        marker = " [bold green]← active[/bold green]" if name == current else ""
        console.print(f"  [bold]{name:<28}[/bold] [dim]{locale}[/dim]{marker}")

        if preview:
            subprocess.run(["/usr/bin/say", "-v", name, sample], check=False)

    console.print(f"\n[dim]To set a voice:  aria voices --set Samantha[/dim]")
    console.print(f"[dim]To set rate:     aria voices --rate 180[/dim]")
    console.print(f"[dim]To preview all:  aria voices --preview[/dim]\n")


@app.command()
def telegram() -> None:
    """Start the ARIA Telegram bot."""
    from aria.config import get_config

    cfg = get_config()
    try:
        cfg.validate()
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1)

    if not cfg.telegram_token:
        console.print("""
[bold yellow]Telegram bot token not configured.[/bold yellow]

[bold]Setup (takes 2 minutes):[/bold]

1. Open Telegram and search for [bold]@BotFather[/bold]
2. Send: /newbot
3. Follow prompts — choose a name and username for your bot
4. BotFather gives you a token like: [bold]123456:ABC-DEF...[/bold]
5. Add to your [bold].env[/bold]:

   TELEGRAM_TOKEN=your_bot_token_here

   # Optional: restrict to your account only
   # TELEGRAM_ALLOWED_IDS=your_telegram_user_id

6. Re-run [bold]aria telegram[/bold]

[dim]To find your Telegram user ID, message @userinfobot[/dim]
""")
        raise typer.Exit(0)

    console.print(f"[bold cyan]Starting ARIA Telegram bot...[/bold cyan]")
    if cfg.telegram_allowed_ids:
        console.print(f"[dim]Allowed users: {cfg.telegram_allowed_ids}[/dim]")
    else:
        console.print("[yellow]Warning: No TELEGRAM_ALLOWED_IDS set — anyone can chat with your ARIA.[/yellow]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    from aria.integrations.telegram import run_bot
    run_bot(cfg.telegram_token)


@app.command()
def whatsapp(
    host: str = typer.Option(None, "--host", help="Bind host"),
    port: int = typer.Option(None, "--port", "-p", help="Bind port"),
    ngrok: bool = typer.Option(False, "--ngrok", help="Start ngrok tunnel and print webhook URL"),
) -> None:
    """Start ARIA with WhatsApp webhook (shows setup instructions)."""
    from aria.config import get_config

    cfg = get_config()
    try:
        cfg.validate()
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(1)

    _host = host or cfg.host
    _port = port or cfg.port

    if not cfg.whatsapp_token or not cfg.whatsapp_phone_number_id:
        console.print("""
[bold yellow]WhatsApp credentials not configured.[/bold yellow]

[bold]Setup steps:[/bold]

1. Go to [link]https://developers.facebook.com[/link]
2. Create an app → Business type → Add WhatsApp product
3. In WhatsApp > API Setup, note your:
   - [bold]Temporary access token[/bold]  → WHATSAPP_TOKEN
   - [bold]Phone number ID[/bold]          → WHATSAPP_PHONE_NUMBER_ID
4. Add to your [bold].env[/bold]:

   WHATSAPP_TOKEN=your_token_here
   WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id
   WHATSAPP_VERIFY_TOKEN=aria-webhook

5. Set webhook URL in Meta dashboard:
   [bold]https://your-public-url/whatsapp[/bold]
   Verify token: [bold]aria-webhook[/bold]
   Subscribe to: [bold]messages[/bold]

6. Re-run [bold]aria whatsapp[/bold]
""")
        raise typer.Exit(0)

    webhook_url = f"http://{_host}:{_port}/whatsapp"
    console.print(f"""
[bold cyan]Starting ARIA WhatsApp server[/bold cyan]

  Local webhook:  [bold]{webhook_url}[/bold]
  Verify token:   [bold]{cfg.whatsapp_verify_token}[/bold]

[dim]Set this URL in Meta Dashboard → WhatsApp → Configuration → Webhook[/dim]
[dim]Subscribe to: messages[/dim]
""")

    if ngrok:
        import subprocess
        console.print("[dim]Starting ngrok tunnel...[/dim]")
        proc = subprocess.Popen(
            ["ngrok", "http", str(_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import time; time.sleep(2)
        try:
            import httpx
            resp = httpx.get("http://localhost:4040/api/tunnels", timeout=5)
            tunnels = resp.json().get("tunnels", [])
            public_url = next(
                (t["public_url"] for t in tunnels if t["public_url"].startswith("https")),
                None,
            )
            if public_url:
                console.print(f"[bold green]ngrok URL:[/bold green] {public_url}/whatsapp")
                console.print(f"[dim]Use this as your Meta webhook URL.[/dim]\n")
        except Exception:
            console.print("[yellow]Could not fetch ngrok URL — check http://localhost:4040[/yellow]")

    import uvicorn
    from aria.ui.app import create_app

    uvicorn.run(
        create_app(),
        host=_host,
        port=_port,
        log_level="info",
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
