# ARIA — Adaptive Reasoning & Intelligence Assistant

ARIA is a personal AI assistant powered by Claude (Anthropic). She has the personality of Donna Paulsen from Suits — sharp, confident, quietly competent, and occasionally smug in the most endearing way. She runs locally, speaks aloud, listens to your voice, and works across CLI, web UI, and Telegram.

---

## Features

- **Conversational AI** — Claude-powered with persistent memory across sessions
- **Voice I/O** — Speak to ARIA, she speaks back (Whisper STT + Kokoro neural TTS)
- **Web search** — Real-time search via DuckDuckGo / Google Custom Search
- **File I/O** — Read, write, and list files on your machine
- **Code execution** — Run Python snippets in a subprocess sandbox
- **Reminders** — Natural language ("remind me in 10 minutes") with background scheduler
- **Calendar & todos** — Add events, manage tasks, get a daily plan
- **Long-term memory** — Remembers facts across sessions via SQLite
- **Telegram bot** — Chat with ARIA from your phone
- **Web UI** — Browser-based streaming chat with voice input
- **Docker** — One-command containerised deployment

---

## Architecture

```
aria/
├── agent.py              # Core agent loop (Claude streaming tool-use)
├── cli.py                # Typer CLI: chat, serve, voice, telegram, voices
├── config.py             # Config dataclass — env vars via .env
├── tools/
│   ├── __init__.py       # Tool registry + dispatcher
│   ├── web_search.py     # web_search tool
│   ├── file_io.py        # read_file, write_file, list_dir
│   ├── code_exec.py      # run_python (subprocess sandbox)
│   ├── reminders.py      # set_reminder, list_reminders, delete_reminder
│   └── calendar.py       # add_event, list_events, add_todo, list_todos, daily_plan
├── memory/
│   └── store.py          # aiosqlite: conversations + facts + reminders + events + todos
├── voice/
│   ├── stt.py            # Whisper mic recording + transcription
│   ├── tts.py            # Kokoro ONNX neural TTS (falls back to macOS say)
│   ├── preprocess.py     # Text cleaning before TTS (markdown, emoji, prosody hints)
│   └── interrupt.py      # Spacebar-interruptible TTS playback
├── integrations/
│   └── telegram.py       # python-telegram-bot: text, voice, photo, location handlers
└── ui/
    ├── app.py            # FastAPI: SSE /chat, /voice endpoints
    └── templates/
        ├── base.html
        └── chat.html     # Vanilla JS streaming chat + MediaRecorder voice input
```

---

## How It Works

### Agent Loop (`agent.py`)

ARIA uses Claude's streaming Messages API with tool use. The loop:

1. Injects long-term memory facts and conversation history into the system prompt
2. Streams a Claude response, collecting text chunks and tool-use blocks
3. Dispatches tool calls in parallel, collects results
4. Feeds results back to Claude for the next turn
5. Repeats until `stop_reason == "end_turn"`
6. Parses `REMEMBER: key=value` lines from the response and stores them as facts

### Memory (`memory/store.py`)

SQLite with WAL mode via `aiosqlite`. Three layers:

- **Conversation history** — rolling window of last N turns per session
- **Session facts** — key/value pairs scoped to a session
- **Global facts** — key/value pairs shared across all sessions

Facts are injected into the system prompt on every turn so ARIA remembers things you've told her.

### Voice Pipeline (`voice/`)

**STT**: OpenAI Whisper (local, `small` model by default). Records from mic via `sounddevice`, transcribes to text.

**TTS**: Kokoro ONNX neural TTS (`af_heart` voice). Falls back to macOS `say`.

**Pipeline**: Text streams from Claude sentence by sentence → each sentence queued for Kokoro generation → audio played back continuously. Generation of sentence N+1 happens while sentence N is playing, so there are no gaps.

**Interrupt**: Press **Space** during playback to stop ARIA mid-sentence. She immediately starts listening for your next message.

**Preprocessing** (`preprocess.py`): Strips markdown, emojis, tables, code blocks. Adds prosody hints (`...?` before questions, `, ` for em-dashes) so Kokoro produces natural intonation.

### Web UI (`ui/app.py`)

FastAPI with Server-Sent Events. The browser streams tokens as they arrive — no page reload. Voice input uses the browser `MediaRecorder` API, sends WebM audio to `/voice`, which transcribes via Whisper and streams the response back.

### Telegram Bot (`integrations/telegram.py`)

Uses `python-telegram-bot` v21 with polling. Handles:
- Text messages → agent response
- Voice notes → Whisper transcription → agent response
- Photos, documents, locations → described to agent

Each Telegram chat gets its own session ID (`telegram_{chat_id}`), so ARIA remembers per user.

---

## Setup

### Requirements

- Python 3.11+
- macOS (for `say` TTS fallback) or Linux
- [ffmpeg](https://ffmpeg.org/) (for voice: `brew install ffmpeg`)
- Kokoro model files (see below)

### Install

```bash
git clone https://github.com/srinair7/aria
cd aria

python3 -m venv .venv && source .venv/bin/activate

pip install -e .           # core (chat + web UI + Telegram)
pip install -e ".[voice]"  # + voice I/O (Whisper + sounddevice)
```

### Kokoro TTS Models

Download and place in `data/`:

```bash
mkdir -p data
# kokoro-v1.0.onnx (~27MB) and voices-v1.0.bin (~310MB)
# from https://github.com/thewh1teagle/kokoro-onnx/releases
```

### Environment Variables

```bash
cp .env.example .env
# Edit .env and set at minimum:
# ANTHROPIC_API_KEY=your_key_here
```

All variables:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `ARIA_MODEL` | `claude-sonnet-4-6` | Claude model |
| `ARIA_HTTP_PROXY` | — | Base URL proxy (e.g. corporate gateway) |
| `ARIA_DB_PATH` | `./data/aria.db` | SQLite database path |
| `ARIA_MEMORY_WINDOW` | `20` | Conversation turns to keep in context |
| `ARIA_VOICE_MODEL` | `small` | Whisper model size |
| `ARIA_TTS_BACKEND` | `kokoro` | TTS backend: `kokoro`, `say`, or `auto` |
| `ARIA_SAY_VOICE` | `af_heart` | Kokoro voice name |
| `ARIA_SAY_RATE` | `160` | Speaking rate (words/min) |
| `TELEGRAM_TOKEN` | — | Telegram bot token (from @BotFather) |
| `TELEGRAM_ALLOWED_IDS` | — | Comma-separated Telegram user IDs (leave empty for open) |
| `GOOGLE_API_KEY` | — | Google Custom Search API key |
| `GOOGLE_CX` | — | Google Custom Search engine ID |
| `ARIA_HOST` | `0.0.0.0` | Web UI bind host |
| `ARIA_PORT` | `8000` | Web UI bind port |

---

## Usage

### Text chat (CLI)

```bash
aria chat
```

### Voice chat

```bash
aria voice
# Speak naturally. Press Space to interrupt ARIA mid-sentence.
# Say "goodbye" to end the session.
```

### Web UI

```bash
aria serve
# Open http://localhost:8000
```

### Telegram bot

```bash
aria telegram
```

### List / set voices

```bash
aria voices               # list available voices
aria voices --set af_sky  # set voice
aria voices --rate 170    # set speaking rate
```

### Docker

```bash
docker compose up
# Web UI on http://localhost:8000
# Mount ./data for SQLite persistence
```

---

## Tools Available to ARIA

| Tool | Description |
|---|---|
| `web_search` | Search the web (DuckDuckGo or Google) |
| `read_file` | Read a file from disk |
| `write_file` | Write content to a file |
| `list_dir` | List directory contents |
| `run_python` | Execute Python code in a subprocess |
| `set_reminder` | Set a reminder with natural time parsing |
| `list_reminders` | List pending reminders |
| `delete_reminder` | Delete a reminder by ID |
| `add_event` | Add a calendar event |
| `list_events` | List upcoming events |
| `delete_event` | Delete an event |
| `add_todo` | Add a to-do item |
| `list_todos` | List to-do items |
| `complete_todo` | Mark a to-do as complete |
| `daily_plan` | Combined view: events + todos + reminders for today |

---

## Personality

ARIA is modelled after Donna Paulsen from *Suits* — brilliant, unflappable, effortlessly confident. She anticipates what's needed before it's asked, delivers lines with quiet authority, and will raise an eyebrow at a bad idea before executing it flawlessly anyway.

She doesn't say "Certainly!" or "Great question!" She just gets things done.

---

## License

MIT
