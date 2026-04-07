"""
FastAPI server — REST + streaming endpoint for Echelon AI.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from starlette.middleware.base import BaseHTTPMiddleware

from echelon.agent import EchelonAgent

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Echelon AI API", version="0.1.0")

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
        if "X-Frame-Options" in response.headers:
            del response.headers["X-Frame-Options"]
        return response


app.add_middleware(TeamsHeadersMiddleware)

_agent: EchelonAgent | None = None


def _get_agent() -> EchelonAgent:
    global _agent
    if _agent is None:
        _agent = EchelonAgent()
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


# ── Daily Digest Subscription ─────────────────────────────────────


class SubscribeRequest(BaseModel):
    email: EmailStr


@app.post("/digest/subscribe")
def digest_subscribe(req: SubscribeRequest):
    from echelon.digest import subscribe
    added = subscribe(req.email)
    if added:
        return {"status": "subscribed", "email": req.email}
    return {"status": "already_subscribed", "email": req.email}


@app.post("/digest/unsubscribe")
def digest_unsubscribe(req: SubscribeRequest):
    from echelon.digest import unsubscribe
    removed = unsubscribe(req.email)
    if removed:
        return {"status": "unsubscribed", "email": req.email}
    return {"status": "not_found", "email": req.email}


@app.get("/digest/subscribers")
def digest_subscribers():
    from echelon.digest import list_subscribers
    subs = list_subscribers()
    return {"count": len(subs), "subscribers": subs}


@app.post("/digest/send-now")
def digest_send_now():
    """Manually trigger a digest email for yesterday's logs."""
    from echelon.digest import list_subscribers, summarise_day, send_digest_email
    agent = _get_agent()
    subscribers = list_subscribers()
    if not subscribers:
        return {"status": "no_subscribers"}
    summary = summarise_day(agent.log_source)
    failed = send_digest_email(subscribers, summary)
    return {
        "status": "sent",
        "date": summary["date"],
        "total_logs": summary["total"],
        "errors_found": summary["by_level"].get("ERROR", 0) + summary["by_level"].get("FATAL", 0),
        "sent_to": len(subscribers) - len(failed),
        "failed": failed,
    }


@app.get("/digest/preview")
def digest_preview():
    """Preview the digest HTML without sending emails."""
    from echelon.digest import summarise_day, _build_html
    from fastapi.responses import HTMLResponse
    agent = _get_agent()
    summary = summarise_day(agent.log_source)
    html = _build_html(summary)
    return HTMLResponse(content=html)


# ── UI ────────────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(_STATIC_DIR / "dashboard.html")


# ── Dashboard Data API ────────────────────────────────────────

@app.get("/api/dashboard")
def dashboard_data(minutes: int = 1440):
    """Return structured data for the dashboard charts.

    Args:
        minutes: Look-back window (default 1440 = 24 hours).
    """
    from collections import Counter
    from datetime import datetime, timedelta

    from echelon.config import APP_REGISTRY

    agent = _get_agent()
    source = agent.log_source

    end = datetime.now()
    start = end - timedelta(minutes=minutes)
    entries = source.query(start, end, max_results=5000)

    # Build source-to-app mapping
    source_to_app: dict[str, str] = {}
    for app_name, info in APP_REGISTRY.items():
        if info.get("index"):
            source_to_app[info["index"]] = app_name
        if info.get("sourcetype"):
            source_to_app[info["sourcetype"]] = app_name
        for src in info.get("sources", []):
            source_to_app[src.lower()] = app_name

    # Per-level counts
    level_counts: Counter = Counter()
    # Per-app counts
    app_totals: Counter = Counter()
    app_errors: Counter = Counter()
    app_warnings: Counter = Counter()
    # Timeline buckets (hourly)
    timeline: dict[str, dict[str, int]] = {}
    # Error clusters
    error_messages: list[dict] = []
    # Top sources
    source_counts: Counter = Counter()

    for e in entries:
        level_counts[e.level] += 1
        source_counts[e.source] += 1

        # Map to app
        matched_app = source_to_app.get(e.source.lower(), "other")
        if matched_app == "other":
            for an, info in APP_REGISTRY.items():
                for src in info.get("sources", []):
                    if src.lower() in e.source.lower():
                        matched_app = an
                        break
                if matched_app != "other":
                    break

        app_totals[matched_app] += 1
        if e.level in ("ERROR", "FATAL"):
            app_errors[matched_app] += 1
            error_messages.append({
                "time": e.timestamp.strftime("%H:%M:%S"),
                "source": e.source,
                "message": e.message[:200],
                "app": matched_app,
            })
        elif e.level == "WARN":
            app_warnings[matched_app] += 1

        # Timeline bucket (hourly)
        hour_key = e.timestamp.strftime("%Y-%m-%d %H:00")
        if hour_key not in timeline:
            timeline[hour_key] = {"errors": 0, "warnings": 0, "info": 0}
        if e.level in ("ERROR", "FATAL"):
            timeline[hour_key]["errors"] += 1
        elif e.level == "WARN":
            timeline[hour_key]["warnings"] += 1
        else:
            timeline[hour_key]["info"] += 1

    # Cluster errors
    error_clusters: Counter = Counter()
    for em in error_messages:
        error_clusters[em["message"][:100]] += 1
    top_clusters = [
        {"pattern": p, "count": c, "source": next(
            (em["source"] for em in error_messages if em["message"][:100] == p), "?"
        )}
        for p, c in error_clusters.most_common(10)
    ]

    # App status summary
    apps_status = []
    for app_name, info in APP_REGISTRY.items():
        ec = app_errors.get(app_name, 0)
        wc = app_warnings.get(app_name, 0)
        total = app_totals.get(app_name, 0)
        if ec > 0:
            status = "critical"
        elif wc > 0:
            status = "warning"
        elif total > 0:
            status = "healthy"
        else:
            status = "inactive"
        apps_status.append({
            "name": app_name,
            "description": info.get("description", ""),
            "status": status,
            "total": total,
            "errors": ec,
            "warnings": wc,
        })

    # Sort: critical first
    status_order = {"critical": 0, "warning": 1, "healthy": 2, "inactive": 3}
    apps_status.sort(key=lambda x: status_order.get(x["status"], 9))

    # Timeline sorted
    sorted_timeline = sorted(timeline.items())

    return {
        "time_range": {"start": start.isoformat(), "end": end.isoformat(), "minutes": minutes},
        "total_entries": len(entries),
        "level_counts": dict(level_counts),
        "apps": apps_status,
        "timeline": {
            "labels": [t[0] for t in sorted_timeline],
            "errors": [t[1]["errors"] for t in sorted_timeline],
            "warnings": [t[1]["warnings"] for t in sorted_timeline],
            "info": [t[1]["info"] for t in sorted_timeline],
        },
        "top_error_clusters": top_clusters,
        "top_sources": dict(source_counts.most_common(10)),
        "kb_stats": agent.knowledge_base.get_stats(),
    }


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

    from echelon.config import settings

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


# ── Start digest scheduler on app startup ─────────────────────────

@app.on_event("startup")
def _start_digest_scheduler():
    from echelon.config import settings
    if settings.smtp_host:
        from echelon.digest import start_scheduler
        agent = _get_agent()
        base_url = f"http://{settings.api_host}:{settings.api_port}"
        start_scheduler(agent.log_source, base_url)


@app.on_event("shutdown")
def _stop_digest_scheduler():
    from echelon.digest import stop_scheduler
    stop_scheduler()


def run_server():
    """Entry point used by the CLI."""
    import uvicorn
    from echelon.config import settings

    uvicorn.run(
        "echelon.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
