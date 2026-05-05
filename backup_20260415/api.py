"""
FastAPI server — REST + streaming endpoint for Echelon AI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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


# ── Azure DevOps / Builds API ─────────────────────────────────

@app.get("/api/builds")
def api_builds(hours: int = 24, top: int = 50, app: Optional[str] = None):
    """Get recent pipeline builds from Azure DevOps.

    Builds are scoped to the configured folder (``AZDEVOPS_FOLDER``, default
    ``\\STEP-CI``) and optionally filtered to a specific app registered in
    ``APP_REGISTRY`` (e.g. ``?app=myaccount`` or ``?app=sdp``).
    """
    from echelon.sources.azdevops_source import get_azdevops_client
    client = get_azdevops_client()
    if not client.is_configured:
        return {"connected": False, "builds": [], "message": "Azure DevOps not configured. Set AZDEVOPS_ORG, AZDEVOPS_PROJECT, AZDEVOPS_PAT in .env"}

    builds = client.get_all_recent_builds(hours=hours, top=top, app=app)
    return {
        "connected": True,
        "count": len(builds),
        "builds": [
            {
                "id": b.id,
                "name": b.name,
                "status": b.status,
                "result": b.result,
                "start_time": b.start_time.isoformat() if b.start_time else None,
                "finish_time": b.finish_time.isoformat() if b.finish_time else None,
                "branch": b.source_branch,
                "commit": b.source_version,
                "requested_by": b.requested_by,
                "pipeline": b.pipeline_name,
                "url": b.url,
            }
            for b in builds
        ],
    }


@app.get("/api/builds/pipelines")
def api_pipelines():
    """List all pipeline definitions."""
    from echelon.sources.azdevops_source import get_azdevops_client
    client = get_azdevops_client()
    if not client.is_configured:
        return {"connected": False, "pipelines": []}
    pipelines = client.list_pipelines()
    return {"connected": True, "pipelines": pipelines}


@app.get("/api/builds/health")
def api_builds_health():
    """Check Azure DevOps connection."""
    from echelon.sources.azdevops_source import get_azdevops_client
    return get_azdevops_client().health_check()


@app.get("/api/builds/{build_id}")
def api_build_detail(build_id: int):
    """Get detailed build summary with stages, jobs, and tasks."""
    from echelon.sources.azdevops_source import get_azdevops_client
    client = get_azdevops_client()
    if not client.is_configured:
        return {"connected": False, "error": "Azure DevOps not configured"}
    summary = client.get_build_summary(build_id)
    summary["connected"] = "error" not in summary
    return summary


@app.get("/api/commits")
def api_commits(repo: str = "", hours: int = 24, top: int = 20):
    """Get recent commits from Azure DevOps repos."""
    from echelon.sources.azdevops_source import get_azdevops_client
    client = get_azdevops_client()
    if not client.is_configured:
        return {"connected": False, "commits": []}
    commits = client.get_recent_commits(repo_name=repo or None, hours=hours, top=top)
    return {
        "connected": True,
        "count": len(commits),
        "commits": [
            {
                "sha": c.sha,
                "message": c.message,
                "author": c.author,
                "timestamp": c.timestamp.isoformat(),
                "url": c.url,
            }
            for c in commits
        ],
    }



# ── UI ────────────────────────────────────────────────────────
@app.get("/")
def serve_ui():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(_STATIC_DIR / "dashboard.html")


@app.get("/architecture")
def serve_architecture():
    return FileResponse(_STATIC_DIR / "architecture.html")


# ── AI Classification API ─────────────────────────────────────

@app.get("/api/classifications")
def get_classifications(application: Optional[str] = None):
    """Return all AI error classifications for the intelligence dashboard."""
    agent = _get_agent()
    items = agent.knowledge_base.list_all_classifications(application=application or "")

    # Group by classification type
    summary = {"total": len(items), "by_classification": {}, "by_severity": {}, "items": []}
    for it in items:
        meta = it["metadata"]
        cls = meta.get("classification", "unknown")
        sev = meta.get("severity", "unknown")
        summary["by_classification"][cls] = summary["by_classification"].get(cls, 0) + 1
        summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1
        summary["items"].append({
            "id": it["id"],
            "pattern": meta.get("error_pattern", "")[:200],
            "classification": cls,
            "severity": sev,
            "confidence": float(meta.get("confidence", "0")),
            "application": meta.get("application", ""),
            "environment": meta.get("environment", ""),
            "root_cause": meta.get("root_cause", ""),
            "suggested_action": meta.get("suggested_action", ""),
            "occurrence_count": int(meta.get("occurrence_count", "1")),
            "times_classified": int(meta.get("times_classified", "1")),
            "first_seen": meta.get("first_seen", ""),
            "last_seen": meta.get("last_seen", ""),
        })

    return summary


# ── Dashboard Data API ────────────────────────────────────────

@app.get("/api/dashboard")
def dashboard_data(minutes: int = 1440, app: Optional[str] = None, env: Optional[str] = None):
    """Return structured data for the dashboard charts.

    Args:
        minutes: Look-back window (default 1440 = 24 hours).
        app: Optional app name to filter results (e.g. 'myaccount', 'sdp').
        env: Optional environment filter (e.g. 'dev', 'qa', 'stg', 'prod').
    """
    from collections import Counter
    from datetime import datetime, timedelta

    from echelon.config import APP_REGISTRY
    from echelon.sources.splunk_source import SplunkSource

    agent = _get_agent()
    base_source = agent.log_source

    end = datetime.now()
    start = end - timedelta(minutes=minutes)

    # Determine which apps to query
    filter_app = app.strip().lower() if app else None
    filter_env = env.strip().lower() if env else None
    apps_to_query = {}
    for app_name, info in APP_REGISTRY.items():
        if filter_app and app_name != filter_app:
            continue
        apps_to_query[app_name] = info

    # Query each app's Splunk queries separately so we get logs from all pods/indexes
    all_entries: list = []
    entry_app_map: dict = {}  # id(entry) -> app_name
    entry_env_map: dict = {}  # id(entry) -> env
    entry_label_map: dict = {}  # id(entry) -> label

    # Collect queries, applying env filter
    filtered_queries: list = []  # (app_name, query_dict)
    for app_name, info in apps_to_query.items():
        queries = info.get("splunk_queries", [])
        if not queries:
            idx = info.get("index")
            st = info.get("sourcetype")
            if idx:
                queries = [{"index": idx, "sourcetype": st, "env": "prod", "label": "default"}]
        for q in queries:
            if filter_env and q.get("env", "").lower() != filter_env:
                continue
            filtered_queries.append((app_name, q))

    per_query_limit = max(500, 5000 // max(len(filtered_queries), 1))

    for app_name, q in filtered_queries:
        q_index = q.get("index")
        q_sourcetype = q.get("sourcetype")
        if not q_index:
            continue
        try:
            app_source = SplunkSource(index=q_index, sourcetype=q_sourcetype)
            app_entries = app_source.query(start, end, max_results=per_query_limit)
            for e in app_entries:
                entry_app_map[id(e)] = app_name
                entry_env_map[id(e)] = q.get("env", "unknown")
                entry_label_map[id(e)] = q.get("label", q_index)
            all_entries.extend(app_entries)
        except Exception:
            pass

    # If no app-specific entries (e.g. no APP_REGISTRY indexes), fall back to default
    if not all_entries and not filter_app and not filter_env:
        all_entries = base_source.query(start, end, max_results=5000)

    entries = all_entries

    # Collect available environments for the response
    available_envs = sorted({
        q.get("env", "unknown")
        for _, info in APP_REGISTRY.items()
        for q in info.get("splunk_queries", [])
        if q.get("env")
    })

    # Build source-to-app mapping (exact keys) — include all splunk_queries
    source_to_app: dict[str, str] = {}
    for app_name, info in APP_REGISTRY.items():
        if info.get("index"):
            source_to_app[info["index"].lower()] = app_name
        if info.get("sourcetype"):
            source_to_app[info["sourcetype"].lower()] = app_name
        for q in info.get("splunk_queries", []):
            if q.get("index"):
                source_to_app[q["index"].lower()] = app_name
            if q.get("sourcetype"):
                source_to_app[q["sourcetype"].lower()] = app_name
        for src in info.get("sources", []):
            source_to_app[src.lower()] = app_name

    def _match_app(entry) -> str:
        """Resolve a log entry to an app name using direct map, metadata, source path, and patterns."""
        # 0. Direct map from per-app query (most reliable)
        direct = entry_app_map.get(id(entry))
        if direct:
            return direct
        # 1. Try Splunk metadata fields (index, sourcetype)
        idx = (entry.metadata.get("index") or "").lower()
        if idx and idx in source_to_app:
            return source_to_app[idx]
        st = (entry.metadata.get("sourcetype") or "").lower()
        if st and st in source_to_app:
            return source_to_app[st]
        # 2. Try exact source match
        src_lower = entry.source.lower()
        if src_lower in source_to_app:
            return source_to_app[src_lower]
        # 3. Try pipeline/component substring match on source path
        for an, reg_info in APP_REGISTRY.items():
            for pat in reg_info.get("pipelines", []):
                if pat.lower() in src_lower:
                    return an
            for comp in reg_info.get("sources", []):
                if comp.lower() in src_lower:
                    return an
        return "other"

    # Per-level counts
    level_counts: Counter = Counter()
    # Per-app counts
    app_totals: Counter = Counter()
    app_errors: Counter = Counter()
    app_warnings: Counter = Counter()
    # Per-environment counts
    env_totals: Counter = Counter()
    env_errors: Counter = Counter()
    env_warnings: Counter = Counter()
    # Timeline buckets (hourly)
    timeline: dict[str, dict[str, int]] = {}
    # Error clusters
    error_messages: list[dict] = []
    # Top sources
    source_counts: Counter = Counter()

    for e in entries:
        matched_app = _match_app(e)

        # If filtering by app, skip entries that don't belong
        if filter_app and matched_app != filter_app:
            continue

        e_env = entry_env_map.get(id(e), "unknown")
        e_label = entry_label_map.get(id(e), "")

        level_counts[e.level] += 1
        source_counts[e.source] += 1

        app_totals[matched_app] += 1
        env_totals[e_env] += 1
        if e.level in ("ERROR", "FATAL"):
            app_errors[matched_app] += 1
            env_errors[e_env] += 1
            error_messages.append({
                "time": e.timestamp.strftime("%H:%M:%S"),
                "source": e.source,
                "message": e.message[:200],
                "app": matched_app,
                "env": e_env,
                "label": e_label,
            })
        elif e.level == "WARN":
            app_warnings[matched_app] += 1
            env_warnings[e_env] += 1

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

    # Cluster errors — include app name
    error_clusters: Counter = Counter()
    for em in error_messages:
        error_clusters[em["message"][:100]] += 1
    top_clusters = [
        {"pattern": p, "count": c, "source": next(
            (em["source"] for em in error_messages if em["message"][:100] == p), "?"
        ), "app": next(
            (em["app"] for em in error_messages if em["message"][:100] == p), "unknown"
        )}
        for p, c in error_clusters.most_common(10)
    ]

    # Enrich error clusters with AI classification intelligence
    try:
        kb = agent.knowledge_base
        for cluster in top_clusters:
            classifications = kb.search_classifications(cluster["pattern"], top_k=1)
            if classifications and classifications[0].get("distance", 99) < 1.5:
                meta = classifications[0]["metadata"]
                cluster["ai_classification"] = meta.get("classification", "")
                cluster["ai_severity"] = meta.get("severity", "")
                cluster["ai_confidence"] = float(meta.get("confidence", "0"))
                cluster["ai_root_cause"] = meta.get("root_cause", "")
                cluster["ai_action"] = meta.get("suggested_action", "")
                cluster["ai_seen_count"] = int(meta.get("occurrence_count", "1"))
                cluster["ai_times_classified"] = int(meta.get("times_classified", "1"))
            else:
                cluster["ai_classification"] = ""
                cluster["ai_severity"] = ""
                cluster["ai_confidence"] = 0
                # Also check user feedback
            feedback = kb.search_feedback(cluster["pattern"], top_k=1)
            if feedback and feedback[0].get("distance", 99) < 1.5:
                fb_meta = feedback[0]["metadata"]
                cluster["user_feedback"] = fb_meta.get("feedback_type", "")
                cluster["user_note"] = fb_meta.get("user_note", "")
            else:
                cluster["user_feedback"] = ""
                cluster["user_note"] = ""
    except Exception:
        pass  # Don't break dashboard if KB lookup fails

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

    # Filter apps list when a specific app is requested
    if filter_app:
        apps_status = [a for a in apps_status if a["name"] == filter_app]

    # Timeline sorted
    sorted_timeline = sorted(timeline.items())

    # Per-environment summary
    env_summary = []
    for ev in available_envs:
        et = env_totals.get(ev, 0)
        ee = env_errors.get(ev, 0)
        ew = env_warnings.get(ev, 0)
        if ee > 0:
            estatus = "critical"
        elif ew > 0:
            estatus = "warning"
        elif et > 0:
            estatus = "healthy"
        else:
            estatus = "inactive"
        env_summary.append({
            "env": ev,
            "status": estatus,
            "total": et,
            "errors": ee,
            "warnings": ew,
        })

    return {
        "time_range": {"start": start.isoformat(), "end": end.isoformat(), "minutes": minutes},
        "total_entries": len(entries),
        "level_counts": dict(level_counts),
        "apps": apps_status,
        "environments": env_summary,
        "available_envs": available_envs,
        "current_env": filter_env,
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
