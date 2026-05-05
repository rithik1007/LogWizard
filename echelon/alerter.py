"""
Proactive Error Alerter — monitors production logs in the background,
detects new errors, runs AI root cause analysis, and sends email alerts
with the RCA + suggested fix.

Works alongside the daily digest but is real-time:
  - Checks prod logs every N minutes (configurable)
  - Groups errors by pattern to avoid alert storms
  - Uses a cooldown to prevent re-alerting the same pattern repeatedly
  - Calls the AI agent to generate RCA + fix for each new error cluster
  - Sends a polished HTML email to all digest subscribers
"""

from __future__ import annotations

import logging
import smtplib
import threading
from collections import Counter
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from echelon.agent import EchelonAgent

logger = logging.getLogger(__name__)

# Track recently alerted patterns to avoid re-alerting (pattern_key -> last_alerted_time)
_alerted_patterns: dict[str, datetime] = {}
_alerted_lock = threading.Lock()

# ── Error detection ───────────────────────────────────────────────


def _check_prod_errors(agent: EchelonAgent) -> list[dict]:
    """Query all prod Splunk indexes for recent errors.

    Returns a list of error cluster dicts:
        [{"pattern": str, "count": int, "app": str, "env": str,
          "first_time": str, "last_time": str, "sample_source": str}]
    """
    from echelon.config import APP_REGISTRY, settings
    from echelon.sources.splunk_source import SplunkSource

    monitor_envs = {e.strip().lower() for e in settings.alert_environments.split(",")}
    lookback = settings.alert_lookback_minutes

    end = datetime.now()
    start = end - timedelta(minutes=lookback)

    all_errors: list[dict] = []

    for app_name, info in APP_REGISTRY.items():
        queries = info.get("splunk_queries", [])
        if not queries:
            idx = info.get("index")
            if idx:
                queries = [{"index": idx, "sourcetype": info.get("sourcetype"), "env": "prod"}]

        for q in queries:
            q_env = q.get("env", "").lower()
            if q_env not in monitor_envs:
                continue

            q_index = q.get("index")
            q_sourcetype = q.get("sourcetype")
            if not q_index:
                continue

            try:
                source = SplunkSource(index=q_index, sourcetype=q_sourcetype)
                entries = source.query(start, end, max_results=500)

                for e in entries:
                    if e.level in ("ERROR", "FATAL"):
                        all_errors.append({
                            "message": e.message[:300],
                            "timestamp": e.timestamp,
                            "source": e.source,
                            "app": app_name,
                            "env": q_env,
                            "label": q.get("label", q_index),
                        })
            except Exception:
                logger.debug("Failed to query %s/%s for alerter", q_index, q_sourcetype)

    if not all_errors:
        return []

    # Cluster by error pattern (first 120 chars)
    clusters: dict[str, list[dict]] = {}
    for err in all_errors:
        key = err["message"][:120]
        clusters.setdefault(key, []).append(err)

    # Build cluster summaries
    result = []
    for pattern, group in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
        timestamps = [e["timestamp"] for e in group]
        result.append({
            "pattern": pattern,
            "count": len(group),
            "app": group[0]["app"],
            "env": group[0]["env"],
            "label": group[0]["label"],
            "sample_source": group[0]["source"],
            "first_time": min(timestamps).strftime("%H:%M:%S"),
            "last_time": max(timestamps).strftime("%H:%M:%S"),
        })

    return result


def _is_on_cooldown(pattern_key: str, cooldown_minutes: int) -> bool:
    """Check if we've already alerted for this pattern recently."""
    with _alerted_lock:
        last = _alerted_patterns.get(pattern_key)
        if last and (datetime.now() - last).total_seconds() < cooldown_minutes * 60:
            return True
        return False


def _mark_alerted(pattern_key: str) -> None:
    """Record that we just alerted for this pattern."""
    with _alerted_lock:
        _alerted_patterns[pattern_key] = datetime.now()
        # Prune old entries (> 24h)
        cutoff = datetime.now() - timedelta(hours=24)
        stale = [k for k, v in _alerted_patterns.items() if v < cutoff]
        for k in stale:
            del _alerted_patterns[k]

def _is_known_noise(agent: EchelonAgent, pattern: str) -> bool:
    """Check if this error pattern is marked as noise/expected in the AI databank."""
    try:
        kb = agent.knowledge_base
        # Check user feedback (databank entries)
        feedback_hits = kb.search_feedback(pattern, top_k=3)
        for hit in feedback_hits:
            if hit.get("distance", 99) > 1.0:
                continue
            fb_type = hit.get("metadata", {}).get("feedback_type", "")
            if fb_type in ("noise", "expected", "known_issue", "resolved"):
                logger.info("Skipping alert — pattern matched databank noise entry: %s", fb_type)
                return True

        # Check AI classifications
        classification_hits = kb.search_classifications(pattern, top_k=3)
        for hit in classification_hits:
            if hit.get("distance", 99) > 1.0:
                continue
            cls = hit.get("metadata", {}).get("classification", "")
            if cls in ("noise", "known_issue"):
                logger.info("Skipping alert — pattern matched AI classification: %s", cls)
                return True
    except Exception:
        logger.debug("Databank check failed for: %s", pattern[:60])

    return False


# ── AI Root Cause Analysis ────────────────────────────────────────


def _generate_rca(agent: EchelonAgent, cluster: dict) -> dict:
    """Ask the AI agent to perform RCA on an error cluster.

    Returns:
        {"summary": str, "root_cause": str, "fix": str, "severity": str}
    """
    prompt = (
        f"ALERT MODE — A production error was just detected and needs immediate analysis.\n\n"
        f"Application: {cluster['app']}\n"
        f"Environment: {cluster['env']} ({cluster['label']})\n"
        f"Error count: {cluster['count']}x in the last few minutes\n"
        f"Time range: {cluster['first_time']} - {cluster['last_time']}\n"
        f"Source: {cluster['sample_source']}\n"
        f"Error pattern: {cluster['pattern']}\n\n"
        f"Please:\n"
        f"1. Identify the root cause of this error\n"
        f"2. Assess the severity (critical/high/medium/low)\n"
        f"3. Provide a specific fix or remediation steps\n"
        f"4. Check if this is a known issue in your knowledge base\n"
        f"5. Auto-classify this error for future reference\n\n"
        f"Be concise but specific. This will be sent as an alert email."
    )

    try:
        response = agent.chat(prompt)
        # Clear this from conversation history so alerts don't pollute user chat
        if len(agent._history) >= 2:
            agent._history = agent._history[:-2]
        return {
            "summary": response,
            "root_cause": _extract_section(response, "root cause"),
            "fix": _extract_section(response, "fix", "remediation", "action", "recommended"),
            "severity": _extract_severity(response),
        }
    except Exception as exc:
        logger.exception("AI RCA generation failed for: %s", cluster["pattern"][:80])
        return {
            "summary": f"AI analysis failed: {str(exc)[:200]}",
            "root_cause": "Unable to determine — AI analysis error",
            "fix": "Please investigate manually",
            "severity": "unknown",
        }


def _extract_section(text: str, *keywords: str) -> str:
    """Try to extract a section from AI response matching keywords."""
    lines = text.split("\n")
    capture = False
    result = []
    for line in lines:
        lower = line.lower().strip()
        if any(kw in lower for kw in keywords) and ("##" in line or "**" in line or ":" in line):
            capture = True
            continue
        elif capture:
            if line.strip().startswith("##") or line.strip().startswith("###"):
                break  # Next section
            if line.strip():
                result.append(line.strip())
            elif result:  # Empty line after content = end
                break
    return "\n".join(result[:10]) if result else ""


def _extract_severity(text: str) -> str:
    """Extract severity from AI response."""
    lower = text.lower()
    for sev in ["critical", "high", "medium", "low"]:
        if f"severity: **{sev}" in lower or f"severity:** {sev}" in lower or f"severity: {sev}" in lower:
            return sev
    if "critical" in lower[:500]:
        return "critical"
    if "high" in lower[:500]:
        return "high"
    return "medium"


# ── Alert email ───────────────────────────────────────────────────

_ALERT_CSS = """\
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f4f7; margin: 0; padding: 20px; color: #2d3748; }
.container { max-width: 680px; margin: 0 auto; background: #fff; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
.header { padding: 24px 32px; color: #fff; }
.header-critical { background: linear-gradient(135deg, #e53e3e, #c53030); }
.header-high { background: linear-gradient(135deg, #dd6b20, #c05621); }
.header-medium { background: linear-gradient(135deg, #d69e2e, #b7791f); }
.header-low { background: linear-gradient(135deg, #3182ce, #2b6cb0); }
.header-unknown { background: linear-gradient(135deg, #6c63ff, #a78bfa); }
.header h1 { margin: 0; font-size: 18px; font-weight: 700; }
.header p { margin: 6px 0 0; opacity: 0.9; font-size: 13px; }
.body-content { padding: 24px 32px; }
.meta-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
.meta-item { background: #f7fafc; border-radius: 8px; padding: 10px 16px; flex: 1; min-width: 120px; }
.meta-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: #a0aec0; margin-bottom: 2px; }
.meta-value { font-size: 14px; font-weight: 600; color: #2d3748; }
.error-box { background: #fff5f5; border: 1px solid #fed7d7; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; }
.error-box code { font-family: 'Consolas', monospace; font-size: 12px; color: #c53030; word-break: break-all; }
.section { margin-bottom: 20px; }
.section h3 { font-size: 14px; font-weight: 700; color: #2d3748; margin: 0 0 8px; }
.section-content { font-size: 13px; line-height: 1.7; color: #4a5568; white-space: pre-wrap; }
.fix-box { background: #f0fff4; border: 1px solid #c6f6d5; border-radius: 8px; padding: 14px 18px; }
.fix-box .section-content { color: #276749; }
.ai-badge { display: inline-block; background: linear-gradient(135deg, #6c63ff, #a78bfa); color: #fff; font-size: 10px; font-weight: 700; padding: 3px 10px; border-radius: 12px; letter-spacing: 0.3px; }
.footer { padding: 16px 32px; background: #f8f9fc; font-size: 11px; color: #a0aec0; text-align: center; }
"""


def _build_alert_html(cluster: dict, rca: dict) -> str:
    """Build the HTML alert email with error details and AI RCA."""
    severity = rca.get("severity", "unknown")
    header_class = f"header-{severity}"
    app_display = cluster["app"].upper() if len(cluster["app"]) <= 4 else cluster["app"].title()
    now_str = datetime.now().strftime("%B %d, %Y at %H:%M")

    severity_emoji = {
        "critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "unknown": "❓"
    }
    sev_icon = severity_emoji.get(severity, "❓")

    ai_summary = escape(rca.get("summary", "No analysis available."))
    # Convert markdown bold to HTML bold for readability
    import re
    ai_summary = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', ai_summary)
    ai_summary = ai_summary.replace("\n", "<br>")

    fix_html = ""
    if rca.get("fix"):
        fix_text = escape(rca["fix"])
        fix_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', fix_text)
        fix_text = fix_text.replace("\n", "<br>")
        fix_html = f"""
        <div class="section">
            <h3>✅ Suggested Fix</h3>
            <div class="fix-box">
                <div class="section-content">{fix_text}</div>
            </div>
        </div>
        """

    return f"""\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{_ALERT_CSS}</style></head>
<body>
<div class="container">
    <div class="header {header_class}">
        <h1>{sev_icon} Production Error Alert — {escape(app_display)}</h1>
        <p>{now_str} · {cluster['env'].upper()} · {escape(cluster['label'])}</p>
    </div>

    <div class="body-content">
        <div class="meta-grid">
            <div class="meta-item">
                <div class="meta-label">Application</div>
                <div class="meta-value">{escape(app_display)}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Environment</div>
                <div class="meta-value">{cluster['env'].upper()}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Occurrences</div>
                <div class="meta-value" style="color: #e53e3e;">{cluster['count']}×</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Severity</div>
                <div class="meta-value">{sev_icon} {severity.upper()}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Time Range</div>
                <div class="meta-value">{cluster['first_time']} — {cluster['last_time']}</div>
            </div>
        </div>

        <div class="section">
            <h3>🔍 Error Pattern</h3>
            <div class="error-box">
                <code>{escape(cluster['pattern'])}</code>
            </div>
            <p style="font-size: 11px; color: #a0aec0;">Source: {escape(cluster['sample_source'][:120])}</p>
        </div>

        <div class="section">
            <h3>🧠 AI Root Cause Analysis <span class="ai-badge">Echelon AI</span></h3>
            <div class="section-content">{ai_summary}</div>
        </div>

        {fix_html}
    </div>

    <div class="footer">
        Echelon AI — Proactive Error Alerting &amp; Root Cause Analysis<br>
        This alert was automatically generated. Reply with questions or investigate in the <a href="#" style="color: #6c63ff;">Echelon AI dashboard</a>.
    </div>
</div>
</body>
</html>"""


def _send_alert_email(recipients: list[str], cluster: dict, rca: dict) -> list[str]:
    """Send the alert email to all subscribers. Returns list of failed recipients."""
    from echelon.config import settings

    if not settings.smtp_host:
        logger.warning("SMTP not configured — cannot send alert email")
        return recipients

    html = _build_alert_html(cluster, rca)
    severity = rca.get("severity", "unknown")
    app_display = cluster["app"].upper() if len(cluster["app"]) <= 4 else cluster["app"].title()

    severity_prefix = {
        "critical": "🔴 CRITICAL",
        "high": "🟠 HIGH",
        "medium": "🟡 MEDIUM",
        "low": "🔵 LOW",
    }
    prefix = severity_prefix.get(severity, "⚠️ ALERT")
    subject = f"{prefix} — {app_display} Production Error ({cluster['count']}× in {cluster['env'].upper()})"

    failed: list[str] = []
    try:
        if settings.smtp_use_tls:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30)
            server.starttls()
        else:
            server = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30)

        if settings.smtp_username:
            server.login(settings.smtp_username, settings.smtp_password)

        for recipient in recipients:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = settings.digest_from_email
                msg["To"] = recipient
                msg.attach(MIMEText(html, "html", "utf-8"))
                server.sendmail(settings.digest_from_email, [recipient], msg.as_string())
            except Exception:
                logger.exception("Failed to send alert to %s", recipient)
                failed.append(recipient)

        server.quit()
    except Exception:
        logger.exception("SMTP connection failed — alert not sent")
        return recipients

    sent = len(recipients) - len(failed)
    logger.info("Alert email sent to %d/%d subscribers for %s", sent, len(recipients), cluster["app"])
    return failed


# ── Background monitor loop ───────────────────────────────────────

_monitor_thread: threading.Thread | None = None
_monitor_stop = threading.Event()


def _monitor_loop(agent: EchelonAgent) -> None:
    """Background loop: check prod logs for errors, run AI RCA, send alerts."""
    from echelon.config import settings
    from echelon.digest import list_subscribers

    logger.info(
        "Proactive alerter started — checking %s every %ds (threshold: %d errors, cooldown: %dm)",
        settings.alert_environments,
        settings.alert_check_interval,
        settings.alert_error_threshold,
        settings.alert_cooldown_minutes,
    )

    while not _monitor_stop.is_set():
        try:
            subscribers = list_subscribers()
            if not subscribers:
                _monitor_stop.wait(settings.alert_check_interval)
                continue

            # Check for new prod errors
            clusters = _check_prod_errors(agent)

            for cluster in clusters:
                # Skip if below threshold
                if cluster["count"] < settings.alert_error_threshold:
                    continue

                # Build a unique key for cooldown tracking
                pattern_key = f"{cluster['app']}:{cluster['env']}:{cluster['pattern'][:80]}"

                if _is_on_cooldown(pattern_key, settings.alert_cooldown_minutes):
                    logger.debug("Skipping (cooldown): %s", pattern_key[:60])
                    continue

                # Skip errors marked as noise/valid in the AI databank
                if _is_known_noise(agent, cluster["pattern"]):
                    logger.debug("Skipping (databank noise): %s", pattern_key[:60])
                    continue

                logger.info(
                    "New error detected: %s — %dx in %s/%s. Running AI RCA...",
                    cluster["pattern"][:60], cluster["count"], cluster["app"], cluster["env"],
                )

                # Generate AI RCA
                rca = _generate_rca(agent, cluster)

                # Send alert email
                failed = _send_alert_email(subscribers, cluster, rca)

                # Mark as alerted (even if some emails failed)
                _mark_alerted(pattern_key)

                if failed:
                    logger.warning("Alert email failed for: %s", ", ".join(failed))

        except Exception:
            logger.exception("Alerter cycle failed")

        # Wait for next check
        _monitor_stop.wait(settings.alert_check_interval)


def start_alerter(agent: EchelonAgent) -> None:
    """Start the background proactive alerter thread."""
    global _monitor_thread
    from echelon.config import settings

    if not settings.alert_enabled:
        logger.info("Proactive alerter is disabled (ALERT_ENABLED=false)")
        return

    if _monitor_thread and _monitor_thread.is_alive():
        return

    _monitor_stop.clear()
    _monitor_thread = threading.Thread(
        target=_monitor_loop,
        args=(agent,),
        daemon=True,
        name="prod-alerter",
    )
    _monitor_thread.start()


def stop_alerter() -> None:
    """Signal the alerter to stop."""
    _monitor_stop.set()
