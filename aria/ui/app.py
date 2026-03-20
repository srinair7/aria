"""FastAPI web UI with SSE streaming and voice endpoint."""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app() -> FastAPI:
    app = FastAPI(title="ARIA Web UI", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    # ── Routes ───────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        session_id = request.cookies.get("aria_session") or str(uuid.uuid4())
        response = templates.TemplateResponse(
            "chat.html", {"request": request, "session_id": session_id}
        )
        response.set_cookie("aria_session", session_id, max_age=86400 * 30)
        return response

    @app.post("/chat")
    async def chat_stream(
        request: Request,
        message: str = Form(...),
        session_id: str = Form(...),
    ) -> StreamingResponse:
        """SSE endpoint — streams assistant tokens."""
        from aria.config import get_config
        from aria.agent import Agent

        try:
            get_config().validate()
        except ValueError as exc:
            async def _err():
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return StreamingResponse(_err(), media_type="text/event-stream")

        agent = Agent(session_id=session_id)

        async def _gen():
            try:
                async for chunk in agent.stream(message):
                    # SSE format
                    payload = json.dumps({"token": chunk})
                    yield f"data: {payload}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/voice")
    async def voice_endpoint(
        audio: UploadFile = File(...),
        session_id: str = Form(...),
    ) -> StreamingResponse:
        """Receive audio blob → STT → agent stream → SSE."""
        try:
            from aria.voice.stt import transcribe_bytes
        except ImportError:
            async def _err():
                yield f"data: {json.dumps({'error': 'Voice dependencies not installed'})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_err(), media_type="text/event-stream")

        from aria.agent import Agent

        audio_bytes = await audio.read()
        mime_type = audio.content_type or "audio/wav"

        try:
            text = transcribe_bytes(audio_bytes, mime_type)
        except Exception as exc:
            async def _stt_err():
                yield f"data: {json.dumps({'error': f'STT error: {exc}'})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stt_err(), media_type="text/event-stream")

        if not text:
            async def _empty():
                yield f"data: {json.dumps({'transcript': '', 'token': '(no speech detected)'})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_empty(), media_type="text/event-stream")

        agent = Agent(session_id=session_id)

        async def _gen():
            # First emit the transcript
            yield f"data: {json.dumps({'transcript': text})}\n\n"
            async for chunk in agent.stream(text):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app
