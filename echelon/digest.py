"""
Daily Digest — Summarise logs and email subscribers.

Collects the day's log data, builds an HTML summary with errors
highlighted in red, and sends it to all subscribed email addresses.
"""

from __future__ import annotations

import json
import logging
import smtplib
import threading
from collections import Counter
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from echelon.sources.base import LogSource

logger = logging.getLogger(__name__)

# ── Subscriber storage ────────────────────────────────────────────


def _subscribers_path() -> Path:
    from echelon.config import settings
    return Path(settings.digest_subscribers_file)


def _load_subscribers() -> list[str]:
    path = _subscribers_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list({e.strip().lower() for e in data if isinstance(e, str) and e.strip()})
    except (json.JSONDecodeError, OSError):
        return []


def _save_subscribers(emails: list[str]) -> None:
    path = _subscribers_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(set(emails)), indent=2), encoding="utf-8")


def subscribe(email: str) -> bool:
    """Add an email to the digest subscriber list. Returns True if newly added."""
    email = email.strip().lower()
    subs = _load_subscribers()
    if email in subs:
        return False
    subs.append(email)
    _save_subscribers(subs)
    return True


def unsubscribe(email: str) -> bool:
    """Remove an email from the digest subscriber list. Returns True if removed."""
    email = email.strip().lower()
    subs = _load_subscribers()
    if email not in subs:
        return False
    subs.remove(email)
    _save_subscribers(subs)
    return True


def list_subscribers() -> list[str]:
    return _load_subscribers()


# ── Log summarisation ─────────────────────────────────────────────


def summarise_day(log_source: LogSource, target_date: datetime | None = None) -> dict:
    """Build a summary dict for the given day's logs, grouped by application.

    Returns:
        {
            "date": "YYYY-MM-DD",
            "total": int,
            "by_level": {"ERROR": n, ...},
            "apps": {
                "myaccount": {"total": n, "errors": [...], "warnings": [...], "status": "ok"|"warn"|"critical"},
                ...
            },
            "unmatched": {"total": n, "errors": [...], "warnings": [...]},
            "errors": [LogEntry, ...],
            "warnings": [LogEntry, ...],
        }
    """
    from echelon.config import APP_REGISTRY

    if target_date is None:
        target_date = datetime.now() - timedelta(days=1)  # yesterday

    start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    entries = log_source.query(start, end, max_results=5000)

    level_counts: Counter[str] = Counter()
    all_errors = []
    all_warnings = []

    # Build source-to-app mapping from APP_REGISTRY
    source_to_app: dict[str, str] = {}
    for app_name, info in APP_REGISTRY.items():
        # Map by Splunk index/sourcetype
        if info.get("index"):
            source_to_app[info["index"]] = app_name
        if info.get("sourcetype"):
            source_to_app[info["sourcetype"]] = app_name
        # Map by component source names (for local file logs)
        for src in info.get("sources", []):
            source_to_app[src.lower()] = app_name

    # Per-app buckets
    app_data: dict[str, dict] = {}
    for app_name in APP_REGISTRY:
        app_data[app_name] = {"total": 0, "errors": [], "warnings": [], "info": 0}
    unmatched = {"total": 0, "errors": [], "warnings": [], "info": 0}

    for e in entries:
        level_counts[e.level] += 1

        # Try to match entry to an app by source field
        matched_app = source_to_app.get(e.source.lower())
        if not matched_app:
            # Fuzzy: check if source contains app index/sourcetype/component substring
            for app_name, info in APP_REGISTRY.items():
                if info["index"] in e.source or info.get("sourcetype", "") in e.source:
                    matched_app = app_name
                    break
                # Check component sources
                for src in info.get("sources", []):
                    if src.lower() in e.source.lower():
                        matched_app = app_name
                        break
                if matched_app:
                    break

        bucket = app_data.get(matched_app) if matched_app else None
        if bucket is None:
            bucket = unmatched

        bucket["total"] += 1
        if e.level in ("ERROR", "FATAL"):
            bucket["errors"].append(e)
            all_errors.append(e)
        elif e.level == "WARN":
            bucket["warnings"].append(e)
            all_warnings.append(e)
        else:
            bucket["info"] += 1

    # Compute status per app
    apps_summary = {}
    for app_name, data in app_data.items():
        error_count = len(data["errors"])
        warn_count = len(data["warnings"])
        if error_count > 0:
            status = "critical"
        elif warn_count > 0:
            status = "warn"
        else:
            status = "ok"
        apps_summary[app_name] = {
            "total": data["total"],
            "errors": data["errors"],
            "warnings": data["warnings"],
            "error_count": error_count,
            "warn_count": warn_count,
            "info_count": data["info"],
            "status": status,
            "description": APP_REGISTRY[app_name].get("description", app_name),
        }

    return {
        "date": start.strftime("%Y-%m-%d"),
        "total": len(entries),
        "by_level": dict(level_counts),
        "apps": apps_summary,
        "unmatched": unmatched,
        "errors": all_errors,
        "warnings": all_warnings,
    }


# ── HTML email builder ────────────────────────────────────────────

_CSS = """\
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f4f7; margin: 0; padding: 20px; color: #2d3748; }
.container { max-width: 640px; margin: 0 auto; background: #fff; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
.header { background: linear-gradient(135deg, #6c63ff, #a78bfa); color: #fff; padding: 24px 32px; }
.header h1 { margin: 0; font-size: 20px; font-weight: 600; }
.header p { margin: 6px 0 0; opacity: 0.85; font-size: 13px; }
.body-section { padding: 24px 32px; }
.greeting { font-size: 15px; line-height: 1.6; margin-bottom: 20px; color: #4a5568; }

/* App status cards */
.app-card { border-radius: 8px; padding: 16px 20px; margin-bottom: 12px; }
.app-card-ok { background: #f0fff4; border-left: 4px solid #38a169; }
.app-card-warn { background: #fffff0; border-left: 4px solid #dd6b20; }
.app-card-critical { background: #fff5f5; border-left: 4px solid #e53e3e; }
.app-card-empty { background: #f7fafc; border-left: 4px solid #a0aec0; }
.app-name { font-size: 15px; font-weight: 700; margin: 0; }
.app-name-ok { color: #276749; }
.app-name-warn { color: #c05621; }
.app-name-critical { color: #c53030; }
.app-name-empty { color: #718096; }
.app-status { font-size: 13px; margin: 4px 0 0; line-height: 1.5; }
.app-status-ok { color: #38a169; }
.app-status-warn { color: #c05621; }
.app-status-critical { color: #e53e3e; font-weight: 600; }
.app-status-empty { color: #a0aec0; }
.app-desc { font-size: 11px; color: #a0aec0; margin: 2px 0 0; }

/* Error detail list inside critical cards */
.error-list { margin: 8px 0 0; padding-left: 18px; }
.error-list li { font-size: 12px; color: #c53030; margin: 3px 0; line-height: 1.4; }
.error-list li strong { color: #9b2c2c; }

/* Summary bar */
.summary-bar { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.summary-pill { border-radius: 20px; padding: 6px 14px; font-size: 12px; font-weight: 600; }
.pill-ok { background: #c6f6d5; color: #276749; }
.pill-warn { background: #fefcbf; color: #975a16; }
.pill-error { background: #fed7d7; color: #9b2c2c; }
.pill-total { background: #e2e8f0; color: #4a5568; }

.footer { padding: 16px 32px; background: #f8f9fc; font-size: 11px; color: #a0aec0; text-align: center; }
.footer a { color: #6c63ff; text-decoration: none; }
"""


def _build_html(summary: dict, base_url: str = "") -> str:
    date_str = summary["date"]
    total = summary["total"]
    by_level = summary["by_level"]
    apps = summary.get("apps", {})
    all_errors = summary["errors"]
    all_warnings = summary["warnings"]

    # App cards
    app_cards_html = ""
    for app_name, app in sorted(apps.items(), key=lambda x: {"critical": 0, "warn": 1, "ok": 2}.get(x[1]["status"], 3)):
        status = app["status"]
        ec = app["error_count"]
        wc = app["warn_count"]
        app_total = app["total"]
        desc = app.get("description", "")

        display_name = app_name.upper() if len(app_name) <= 4 else app_name.title()

        if app_total == 0:
            status_text = "No log activity yesterday."
            card_class = "app-card-empty"
            name_class = "app-name-empty"
            status_class = "app-status-empty"
            icon = "💤"
        elif status == "ok":
            status_text = f"No major issues. {app_total} entries — all clean."
            card_class = "app-card-ok"
            name_class = "app-name-ok"
            status_class = "app-status-ok"
            icon = "✅"
        elif status == "warn":
            status_text = f"{wc} warning(s) detected across {app_total} entries. No critical errors."
            card_class = "app-card-warn"
            name_class = "app-name-warn"
            status_class = "app-status-warn"
            icon = "⚠️"
        else:  # critical
            status_text = f"{ec} error(s) found across {app_total} entries — needs investigation."
            card_class = "app-card-critical"
            name_class = "app-name-critical"
            status_class = "app-status-critical"
            icon = "🔴"

        # Error details for critical apps
        error_details = ""
        if status == "critical" and app["errors"]:
            # Cluster errors
            clusters: dict[str, list] = {}
            for e in app["errors"]:
                key = e.message[:120]
                clusters.setdefault(key, []).append(e)
            sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
            items = "".join(
                f'<li><strong>({len(group)}x)</strong> '
                f'<span style="color:#718096;">[{escape(group[0].source)}]</span> '
                f'{escape(pattern[:140])}'
                f'<br><span style="font-size:11px;color:#a0aec0;">First: {group[0].timestamp.strftime("%H:%M:%S")} · Last: {group[-1].timestamp.strftime("%H:%M:%S")}</span>'
                f'</li>'
                for pattern, group in sorted_clusters[:5]
            )
            error_details = f'<ul class="error-list">{items}</ul>'

        app_cards_html += f"""
        <div class="app-card {card_class}">
            <p class="app-name {name_class}">{icon} {escape(display_name)}</p>
            <p class="app-status {status_class}">{status_text}</p>
            {f'<p class="app-desc">{escape(desc)}</p>' if desc else ''}
            {error_details}
        </div>
        """

    unsubscribe_link = ""

    unsubscribe_link = ""
    if base_url:
        unsubscribe_link = f' · <a href="{escape(base_url)}">Manage subscription</a>'

    html = f"""\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{_CSS}</style></head>
<body>
<div class="container">
    <div class="header">
        <h1>📋 Echelon AI — Daily Log Digest</h1>
        <p>{date_str}</p>
    </div>

    <div class="body-section">
        <h3 style="margin: 0 0 14px; font-size: 15px; color: #2d3748;">Application Status</h3>
        {app_cards_html}
    </div>

    <div class="footer">
        Echelon AI — Automated Log Analysis &amp; Monitoring{unsubscribe_link}
    </div>
</div>
</body>
</html>"""
    return html


# ── Email sending ─────────────────────────────────────────────────


def send_digest_email(
    recipients: list[str],
    summary: dict,
    base_url: str = "",
) -> list[str]:
    """Send the daily digest email to all recipients.

    Returns list of emails that failed to send.
    """
    from echelon.config import settings

    if not settings.smtp_host:
        logger.warning("SMTP not configured — skipping digest email send")
        return recipients  # all "failed"

    html = _build_html(summary, base_url)
    date_str = summary["date"]
    error_count = summary["by_level"].get("ERROR", 0) + summary["by_level"].get("FATAL", 0)

    subject = f"Echelon AI — Daily Log Digest ({date_str})"
    if error_count:
        subject += f" — {error_count} error(s) found"

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
                logger.exception("Failed to send digest to %s", recipient)
                failed.append(recipient)

        server.quit()
    except Exception:
        logger.exception("SMTP connection failed — digest not sent")
        return recipients

    sent_count = len(recipients) - len(failed)
    logger.info("Daily digest sent to %d/%d subscribers", sent_count, len(recipients))
    return failed


# ── Scheduler ─────────────────────────────────────────────────────

_scheduler_thread: threading.Thread | None = None
_scheduler_stop = threading.Event()


def _run_daily_digest(log_source: LogSource, base_url: str = "") -> None:
    """Execute one digest cycle: summarise yesterday's logs and email subscribers."""
    subscribers = list_subscribers()
    if not subscribers:
        logger.info("No digest subscribers — skipping")
        return

    summary = summarise_day(log_source)
    failed = send_digest_email(subscribers, summary, base_url)
    if failed:
        logger.warning("Digest failed for: %s", ", ".join(failed))


def _scheduler_loop(log_source: LogSource, base_url: str) -> None:
    """Background loop that fires the digest at the configured time each day."""
    import time
    from echelon.config import settings

    logger.info(
        "Digest scheduler started — will send at %02d:%02d daily",
        settings.digest_send_hour,
        settings.digest_send_minute,
    )

    last_sent_date: str | None = None

    while not _scheduler_stop.is_set():
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        if (
            now.hour == settings.digest_send_hour
            and now.minute == settings.digest_send_minute
            and last_sent_date != today_str
        ):
            logger.info("Triggering daily digest for %s", today_str)
            try:
                _run_daily_digest(log_source, base_url)
            except Exception:
                logger.exception("Digest send failed")
            last_sent_date = today_str

        # Sleep 30s between checks
        _scheduler_stop.wait(30)


def start_scheduler(log_source: LogSource, base_url: str = "") -> None:
    """Start the background digest scheduler thread."""
    global _scheduler_thread

    if _scheduler_thread and _scheduler_thread.is_alive():
        return

    _scheduler_stop.clear()
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        args=(log_source, base_url),
        daemon=True,
        name="digest-scheduler",
    )
    _scheduler_thread.start()


def stop_scheduler() -> None:
    """Signal the scheduler to stop."""
    _scheduler_stop.set()
