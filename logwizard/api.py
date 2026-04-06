"""
FastAPI server — REST + streaming endpoint for LogWizard.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from logwizard.agent import LogWizardAgent

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="LogWizard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Allow embedding in Teams iframes
class TeamsHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Allow Teams to embed the UI in an iframe
        response.headers["Content-Security-Policy"] = (
            "frame-ancestors teams.microsoft.com *.teams.microsoft.com "
            "*.skype.com *.microsoft.com https://localhost:* http://localhost:*"
        )
        response.headers.pop("X-Frame-Options", None)
        return response


app.add_middleware(TeamsHeadersMiddleware)

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


# ── Teams Bot Endpoint ────────────────────────────────────────
@app.post("/api/teams/messages")
async def teams_messages(request: Request):
    """Receives messages from the Azure Bot Service (Teams bot).

    Requires: pip install botbuilder-core botbuilder-schema
    Env vars: TEAMS_BOT_ID, TEAMS_BOT_PASSWORD
    """
    try:
        from botbuilder.core import (
            BotFrameworkAdapter,
            BotFrameworkAdapterSettings,
            TurnContext,
        )
        from botbuilder.schema import Activity
    except ImportError:
        return {"error": "botbuilder-core not installed. Run: pip install botbuilder-core"}

    from logwizard.config import settings

    bot_settings = BotFrameworkAdapterSettings(
        app_id=getattr(settings, "teams_bot_id", ""),
        app_password=getattr(settings, "teams_bot_password", ""),
    )
    adapter = BotFrameworkAdapter(bot_settings)

    async def on_turn(turn_context: TurnContext):
        user_text = turn_context.activity.text or ""
        if not user_text.strip():
            return

        # Send typing indicator
        typing_activity = Activity(type="typing")
        await turn_context.send_activity(typing_activity)

        # Get agent response
        agent = _get_agent()
        response = agent.chat(user_text)
        await turn_context.send_activity(response)

    body = await request.json()
    activity = Activity().deserialize(body)
    auth_header = request.headers.get("Authorization", "")

    await adapter.process_activity(activity, auth_header, on_turn)
    return {}


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
