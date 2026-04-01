"""
LangChain tools that the agent uses to interact with log sources and the knowledge base.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from langchain_core.tools import tool

from logwizard.knowledge import KnowledgeBase
from logwizard.sources.base import LogEntry, LogSource
from logwizard.sources.file_source import FileSource

# Module-level singletons — initialised by `init_tools()`
_log_source: LogSource | None = None
_kb: KnowledgeBase | None = None


def init_tools(
    log_source: LogSource | None = None,
    knowledge_base: KnowledgeBase | None = None,
):
    """Call once at startup to inject the concrete source + KB."""
    global _log_source, _kb
    _log_source = log_source or FileSource()
    _kb = knowledge_base or KnowledgeBase()


def _get_source() -> LogSource:
    if _log_source is None:
        init_tools()
    return _log_source  # type: ignore[return-value]


def _get_kb() -> KnowledgeBase:
    if _kb is None:
        init_tools()
    return _kb  # type: ignore[return-value]


def _entries_to_text(entries: list[LogEntry], limit: int = 60) -> str:
    if not entries:
        return "No log entries found for the given criteria."
    lines = []
    for e in entries[:limit]:
        lines.append(
            f"[{e.timestamp.isoformat()}] [{e.level}] ({e.source}) {e.message[:300]}"
        )
    summary = f"Showing {len(lines)} of {len(entries)} entries."
    return summary + "\n" + "\n".join(lines)


# ── Log query tools ───────────────────────────────────────────────


@tool
def query_logs(
    start_time: str,
    end_time: str,
    search_query: str = "",
    max_results: int = 200,
) -> str:
    """Query log entries from the configured log source.

    Args:
        start_time: ISO-8601 datetime string for the start of the window, e.g. "2026-03-31T14:00:00".
        end_time: ISO-8601 datetime string for the end of the window, e.g. "2026-03-31T15:00:00".
        search_query: Optional keyword / SPL filter to narrow results.
        max_results: Maximum entries to return (default 200).
    """
    try:
        st = datetime.fromisoformat(start_time)
        et = datetime.fromisoformat(end_time)
    except ValueError:
        return "Error: start_time and end_time must be valid ISO-8601 datetime strings."

    entries = _get_source().query(st, et, search_query or None, max_results)
    return _entries_to_text(entries)


@tool
def query_recent_errors(minutes: int = 60, max_results: int = 100) -> str:
    """Fetch recent ERROR / FATAL log entries from the last N minutes.

    Args:
        minutes: Look-back window in minutes (default 60).
        max_results: Maximum entries to return.
    """
    end = datetime.now()
    start = end - timedelta(minutes=minutes)
    entries = _get_source().query(start, end, max_results=max_results)
    errors = [e for e in entries if e.level in ("ERROR", "FATAL", "WARN")]
    return _entries_to_text(errors)


@tool
def get_log_statistics(start_time: str, end_time: str) -> str:
    """Get a statistical summary of logs in a time window (counts by level, top sources).

    Args:
        start_time: ISO-8601 datetime string.
        end_time: ISO-8601 datetime string.
    """
    try:
        st = datetime.fromisoformat(start_time)
        et = datetime.fromisoformat(end_time)
    except ValueError:
        return "Error: Invalid datetime format."

    entries = _get_source().query(st, et, max_results=1000)
    if not entries:
        return "No logs found in the given time range."

    level_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for e in entries:
        level_counts[e.level] = level_counts.get(e.level, 0) + 1
        source_counts[e.source] = source_counts.get(e.source, 0) + 1

    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return json.dumps(
        {
            "total_entries": len(entries),
            "by_level": level_counts,
            "top_sources": dict(top_sources),
            "time_range": {"start": st.isoformat(), "end": et.isoformat()},
        },
        indent=2,
    )


# ── Knowledge base tools ──────────────────────────────────────────


@tool
def search_known_errors(query: str) -> str:
    """Search the knowledge base for previously-seen error patterns.

    Args:
        query: Description of the error to look up.
    """
    results = _get_kb().search_error_patterns(query, top_k=5)
    if not results:
        return "No matching error patterns found in the knowledge base."
    lines = []
    for r in results:
        meta = r["metadata"]
        lines.append(
            f"- Pattern: {r['document'][:200]}\n"
            f"  Actionable: {meta.get('is_actionable', '?')} | "
            f"Category: {meta.get('category', 'N/A')} | "
            f"Notes: {meta.get('notes', '')}"
        )
    return "\n".join(lines)


@tool
def search_past_incidents(query: str) -> str:
    """Search past incident analyses for similar issues.

    Args:
        query: Description of the incident to look up.
    """
    results = _get_kb().search_incidents(query, top_k=5)
    if not results:
        return "No similar past incidents found."
    lines = []
    for r in results:
        meta = r["metadata"]
        lines.append(
            f"- Incident: {meta.get('summary', 'N/A')}\n"
            f"  Root Cause: {meta.get('root_cause', 'N/A')}\n"
            f"  Resolution: {meta.get('resolution', 'N/A')}\n"
            f"  Severity: {meta.get('severity', 'unknown')}"
        )
    return "\n".join(lines)


@tool
def store_learned_pattern(
    pattern: str,
    is_actionable: bool,
    category: str = "",
    notes: str = "",
) -> str:
    """Store a new error pattern that the agent learned from analysis.

    Call this when you identify a recurring error pattern during analysis
    to improve future investigations.

    Args:
        pattern: The error signature / message pattern.
        is_actionable: Whether this error typically requires human action.
        category: Category label (e.g. 'database', 'network', 'oom').
        notes: Any additional notes or context.
    """
    doc_id = _get_kb().store_error_pattern(pattern, is_actionable, category, notes)
    return f"Stored error pattern with id={doc_id}."


@tool
def store_incident_analysis(
    summary: str,
    root_cause: str,
    resolution: str,
    severity: str = "medium",
    tags: str = "",
) -> str:
    """Store the result of an incident analysis to the knowledge base.

    Call this after completing a root-cause analysis so the knowledge
    base can be used in future investigations.

    Args:
        summary: Brief summary of the incident.
        root_cause: Identified root cause.
        resolution: Resolution steps or recommended fix.
        severity: low / medium / high / critical.
        tags: Comma-separated tags (e.g. "database,timeout,prod").
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    doc_id = _get_kb().store_incident_analysis(
        summary, root_cause, resolution, severity, tag_list
    )
    return f"Stored incident analysis with id={doc_id}."


@tool
def knowledge_base_stats() -> str:
    """Return counts of items in each knowledge base collection."""
    stats = _get_kb().get_stats()
    return json.dumps(stats, indent=2)


# ── All tools list ────────────────────────────────────────────────

ALL_TOOLS = [
    query_logs,
    query_recent_errors,
    get_log_statistics,
    search_known_errors,
    search_past_incidents,
    store_learned_pattern,
    store_incident_analysis,
    knowledge_base_stats,
]
