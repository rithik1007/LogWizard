"""
FastAPI server — REST + streaming endpoint for LogWizard.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from logwizard.agent import LogWizardAgent

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="LogWizard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: LogWizardAgent | None = None


def _get_agent() -> LogWizardAgent:
    global _agent
    if _agent is None:
        _agent = LogWizardAgent()
    return _agent


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health():
    agent = _get_agent()
    return {
        "status": "ok",
        "log_source_healthy": agent.log_source.health_check(),
        "knowledge_base": agent.knowledge_base.get_stats(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    agent = _get_agent()
    result = agent.chat(req.message)
    return ChatResponse(response=result)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    agent = _get_agent()

    def generate():
        for chunk in agent.stream_chat(req.message):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/kb/stats")
def kb_stats():
    return _get_agent().knowledge_base.get_stats()


# ── UI ────────────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


def run_server():
    """Entry point used by the CLI."""
    import uvicorn
    from logwizard.config import settings

    uvicorn.run(
        "logwizard.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
