"""File-based log source — reads plain-text or JSON log files from a directory."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from echelon.config import settings
from echelon.sources.base import LogEntry, LogSource

# Common syslog / application log pattern:
#   2024-06-01 14:32:11,123 ERROR [module] Some message
_TS_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)"
)
_LEVEL_PATTERN = re.compile(
    r"\b(FATAL|ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE)\b", re.IGNORECASE
)
# Extract component name from log lines like: ... ERROR [inventory-service] ...
_SOURCE_PATTERN = re.compile(
    r"\b(?:FATAL|ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE)\b\s+\[([^\]]+)\]", re.IGNORECASE
)


class FileSource(LogSource):
    def __init__(self, log_dir: str | None = None):
        self._log_dir = Path(log_dir or settings.log_files_dir)

    def health_check(self) -> bool:
        return self._log_dir.exists()

    def query(
        self,
        start_time: datetime,
        end_time: datetime,
        search_query: str | None = None,
        max_results: int = 500,
    ) -> list[LogEntry]:
        entries: list[LogEntry] = []
        if not self._log_dir.exists():
            return entries

        for filepath in sorted(self._log_dir.rglob("*")):
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in (".log", ".txt", ".json", ".out"):
                continue

            try:
                entries.extend(
                    self._parse_file(filepath, start_time, end_time, search_query)
                )
            except Exception:
                continue

            if len(entries) >= max_results:
                break

        return entries[:max_results]

    def _parse_file(
        self,
        filepath: Path,
        start_time: datetime,
        end_time: datetime,
        search_query: str | None,
    ) -> list[LogEntry]:
        results: list[LogEntry] = []

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue

                entry = self._parse_line(line, filepath.name)
                if entry is None:
                    continue

                if not (start_time <= entry.timestamp <= end_time):
                    continue

                if search_query and search_query.lower() not in entry.raw.lower():
                    continue

                results.append(entry)

        return results

    @staticmethod
    def _parse_line(line: str, source_name: str) -> LogEntry | None:
        # Try JSON first
        if line.lstrip().startswith("{"):
            try:
                obj = json.loads(line)
                ts_str = obj.get("timestamp") or obj.get("time") or obj.get("@timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, TypeError, AttributeError):
                    ts = datetime.now()
                level = str(obj.get("level", obj.get("severity", "INFO"))).upper()
                msg = obj.get("message") or obj.get("msg") or line
                return LogEntry(
                    timestamp=ts,
                    level=level,
                    source=str(obj.get("source", source_name)),
                    message=str(msg)[:2000],
                    raw=line,
                    metadata={
                        k: v
                        for k, v in obj.items()
                        if k not in ("timestamp", "time", "@timestamp", "level", "severity", "message", "msg")
                    },
                )
            except json.JSONDecodeError:
                pass

        # Plain-text parsing
        ts_match = _TS_PATTERN.search(line)
        if ts_match:
            try:
                ts_raw = ts_match.group(1).replace(",", ".")
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                ts = datetime.now()
        else:
            return None  # Skip lines without timestamps

        level_match = _LEVEL_PATTERN.search(line)
        level = level_match.group(1).upper().replace("WARNING", "WARN") if level_match else "INFO"

        # Try to extract component name from brackets after log level
        source_match = _SOURCE_PATTERN.search(line)
        source = source_match.group(1).strip() if source_match else source_name

        return LogEntry(
            timestamp=ts,
            level=level,
            source=source,
            message=line[:2000],
            raw=line,
        )
