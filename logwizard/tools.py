"""
LangChain tools that the agent uses to interact with log sources and the knowledge base.
"""

from __future__ import annotations

import json
from collections import Counter
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
    entries = _get_source().query(start, end, search_query="error OR fatal OR exception", max_results=max_results)
    errors = [e for e in entries if e.level in ("ERROR", "FATAL", "WARN")]
    if not errors:
        errors = entries  # return whatever we got if level filtering removed everything
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


# ── Deep analysis tools ───────────────────────────────────────────


@tool
def analyze_error_context(
    error_keyword: str,
    minutes: int = 30,
    context_window_seconds: int = 60,
    max_errors: int = 10,
) -> str:
    """Fetch errors matching a keyword and retrieve surrounding log context.

    For each error found, this tool also fetches logs from a time window
    around it (before + after) so you can see what led to the error and
    what happened afterwards. This is essential for understanding error
    cascades and root causes.

    Args:
        error_keyword: Keyword to search for in error logs (e.g. "NullPointerException", "timeout", "OOM").
        minutes: Look-back window in minutes (default 30).
        context_window_seconds: Seconds of context to fetch before and after each error (default 60).
        max_errors: Maximum number of distinct errors to analyse (default 10).
    """
    end = datetime.now()
    start = end - timedelta(minutes=minutes)

    # Fetch errors matching the keyword
    entries = _get_source().query(start, end, search_query=error_keyword, max_results=500)
    errors = [e for e in entries if e.level in ("ERROR", "FATAL", "WARN")]

    if not errors:
        # Fall back: maybe the keyword itself is the filter
        errors = entries[:max_errors] if entries else []

    if not errors:
        return f"No errors matching '{error_keyword}' found in the last {minutes} minutes."

    # Deduplicate by message (keep first occurrence of each unique message prefix)
    seen: set[str] = set()
    unique_errors: list[LogEntry] = []
    for e in errors:
        key = e.message[:120]
        if key not in seen:
            seen.add(key)
            unique_errors.append(e)
        if len(unique_errors) >= max_errors:
            break

    sections: list[str] = []
    sections.append(f"Found {len(errors)} error(s), {len(unique_errors)} unique pattern(s) in the last {minutes} min.\n")

    for i, err in enumerate(unique_errors, 1):
        # Fetch context window around this error
        ctx_start = err.timestamp - timedelta(seconds=context_window_seconds)
        ctx_end = err.timestamp + timedelta(seconds=context_window_seconds)
        context_logs = _get_source().query(ctx_start, ctx_end, max_results=50)

        sections.append(f"--- Error #{i} ---")
        sections.append(f"Timestamp: {err.timestamp.isoformat()}")
        sections.append(f"Level: {err.level}")
        sections.append(f"Source: {err.source}")
        sections.append(f"Message: {err.message[:500]}")

        if context_logs:
            sections.append(f"\nContext ({len(context_logs)} log entries around this error):")
            for ctx in context_logs:
                marker = ">>>" if ctx.timestamp == err.timestamp and ctx.message[:80] == err.message[:80] else "   "
                sections.append(
                    f"  {marker} [{ctx.timestamp.isoformat()}] [{ctx.level}] ({ctx.source}) {ctx.message[:200]}"
                )
        sections.append("")

    return "\n".join(sections)


@tool
def query_error_clusters(minutes: int = 60) -> str:
    """Group and count errors from the recent time window to identify patterns.

    Returns clusters of similar errors sorted by frequency, helping identify
    the most impactful issues vs one-off noise. Also shows the time span of
    each cluster to detect whether errors are bursty or sustained.

    Args:
        minutes: Look-back window in minutes (default 60).
    """
    end = datetime.now()
    start = end - timedelta(minutes=minutes)

    entries = _get_source().query(start, end, search_query="error", max_results=1000)
    errors = [e for e in entries if e.level in ("ERROR", "FATAL", "WARN")]

    if not errors:
        return f"No errors found in the last {minutes} minutes."

    # Cluster by normalised message prefix (first 120 chars)
    clusters: dict[str, list[LogEntry]] = {}
    for e in errors:
        key = e.message[:120].strip()
        clusters.setdefault(key, []).append(e)

    # Sort by frequency descending
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    lines: list[str] = []
    lines.append(f"Error clusters in the last {minutes} min: {len(errors)} total errors, {len(sorted_clusters)} distinct patterns.\n")

    for rank, (pattern, group) in enumerate(sorted_clusters[:15], 1):
        first = min(e.timestamp for e in group)
        last = max(e.timestamp for e in group)
        sources = list({e.source for e in group})
        lines.append(f"#{rank} — Count: {len(group)} | First: {first.isoformat()} | Last: {last.isoformat()}")
        lines.append(f"  Sources: {', '.join(sources[:5])}")
        lines.append(f"  Pattern: {pattern}")
        lines.append("")

    return "\n".join(lines)


# ── Application-aware tools ───────────────────────────────────────


@tool
def lookup_application(app_name: str) -> str:
    """Look up a registered application by name to get its Splunk index and sourcetype.

    ALWAYS call this first when a user mentions an application name (e.g. "myaccount",
    "order manager") to resolve the correct Splunk index/sourcetype before querying logs.

    Args:
        app_name: The application name or alias (e.g. "myaccount", "mya").
    """
    from logwizard.config import resolve_app, APP_REGISTRY

    entry = resolve_app(app_name)
    if entry:
        return json.dumps({
            "found": True,
            "app_name": app_name,
            "index": entry["index"],
            "sourcetype": entry["sourcetype"],
            "description": entry.get("description", ""),
        }, indent=2)

    # Not found — list available apps
    available = [
        f"  - {name}: {info['description']}" for name, info in APP_REGISTRY.items()
    ]
    return (
        f"Application '{app_name}' not found in the registry.\n"
        f"Available applications:\n" + "\n".join(available)
    )


@tool
def query_app_errors(
    app_name: str,
    minutes: int = 60,
    search_keywords: str = "",
    max_results: int = 200,
) -> str:
    """Query recent errors for a specific application by name.

    This combines application lookup + Splunk query in one step.
    Use this when the user asks about errors in a specific application.
    ALWAYS set the `minutes` parameter to match what the user asks for
    (e.g. "last 10 minutes" → minutes=10, "last hour" → minutes=60).

    Args:
        app_name: The application name (e.g. "myaccount").
        minutes: Look-back window in minutes (default 60). Set this based on what the user asks.
        search_keywords: Additional keywords to filter (e.g. "timeout", "500", "NullPointer").
        max_results: Maximum entries to return.
    """
    from logwizard.config import resolve_app
    from logwizard.sources.splunk_source import SplunkSource

    entry = resolve_app(app_name)
    if not entry:
        return f"Application '{app_name}' not found. Use lookup_application to see available apps."

    app_source = SplunkSource(index=entry["index"], sourcetype=entry["sourcetype"])

    end = datetime.now()
    start = end - timedelta(minutes=minutes)

    # Run TWO queries and merge results:
    # 1. Broad query — the original Splunk search (just "error") to catch everything
    # 2. Focused query — targets specific severity patterns for deeper analysis
    broad_search = "error"
    focused_search = "(ERROR OR FATAL OR exception OR fail OR timeout)"
    if search_keywords:
        broad_search = f"({search_keywords}) {broad_search}"
        focused_search = f"({search_keywords}) AND ({focused_search})"

    # Query 1: Broad (original style)
    broad_entries = app_source.query(start, end, search_query=broad_search, max_results=max_results)

    # Query 2: Focused (catches exceptions/failures that might not contain "error")
    focused_entries = app_source.query(start, end, search_query=focused_search, max_results=max_results)

    # Merge & deduplicate by (timestamp, message prefix)
    seen: set[str] = set()
    all_entries: list[LogEntry] = []
    for e in broad_entries + focused_entries:
        key = f"{e.timestamp.isoformat()}|{e.message[:100]}"
        if key not in seen:
            seen.add(key)
            all_entries.append(e)

    # Sort newest first
    all_entries.sort(key=lambda e: e.timestamp, reverse=True)

    errors = [e for e in all_entries if e.level in ("ERROR", "FATAL", "WARN")]
    if not errors:
        errors = all_entries

    if not errors:
        return f"No errors found for '{app_name}' ({entry['description']}) in the last {minutes} minutes. The application looks healthy in this window."

    header = (
        f"Errors for **{app_name}** ({entry['description']})\n"
        f"Index: {entry['index']} | Sourcetype: {entry['sourcetype']}\n"
        f"Time range: last {minutes} min | Found: {len(errors)} error(s)\n\n"
    )
    return header + _entries_to_text(errors)


@tool
def query_app_logs(
    app_name: str,
    start_time: str,
    end_time: str,
    search_query: str = "",
    max_results: int = 200,
) -> str:
    """Query logs for a specific application by name within a time range.

    Args:
        app_name: The application name (e.g. "myaccount").
        start_time: ISO-8601 datetime string for the start.
        end_time: ISO-8601 datetime string for the end.
        search_query: Optional SPL filter / keywords.
        max_results: Maximum entries to return.
    """
    from logwizard.config import resolve_app
    from logwizard.sources.splunk_source import SplunkSource

    entry = resolve_app(app_name)
    if not entry:
        return f"Application '{app_name}' not found. Use lookup_application to see available apps."

    try:
        st = datetime.fromisoformat(start_time)
        et = datetime.fromisoformat(end_time)
    except ValueError:
        return "Error: start_time and end_time must be valid ISO-8601 datetime strings."

    app_source = SplunkSource(index=entry["index"], sourcetype=entry["sourcetype"])
    entries = app_source.query(st, et, search_query or None, max_results)

    header = (
        f"Logs for **{app_name}** ({entry['description']})\n"
        f"Index: {entry['index']} | Sourcetype: {entry['sourcetype']}\n\n"
    )
    return header + _entries_to_text(entries)


# ── All tools list ────────────────────────────────────────────────

ALL_TOOLS = [
    lookup_application,
    query_app_errors,
    query_app_logs,
    query_logs,
    query_recent_errors,
    get_log_statistics,
    analyze_error_context,
    query_error_clusters,
    search_known_errors,
    search_past_incidents,
    store_learned_pattern,
    store_incident_analysis,
    knowledge_base_stats,
]
