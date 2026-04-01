"""Splunk log source connector using the official Splunk SDK."""

from __future__ import annotations

import json
from datetime import datetime

import splunklib.client as splunk_client
import splunklib.results as splunk_results

from logwizard.config import settings
from logwizard.sources.base import LogEntry, LogSource


class SplunkSource(LogSource):
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        index: str = "main",
    ):
        self._host = host or settings.splunk_host
        self._port = port or settings.splunk_port
        self._username = username or settings.splunk_username
        self._password = password or settings.splunk_password
        self._index = index
        self._service: splunk_client.Service | None = None

    def _connect(self) -> splunk_client.Service:
        if self._service is None:
            self._service = splunk_client.connect(
                host=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
                scheme=settings.splunk_scheme,
                autologin=True,
            )
        return self._service

    def health_check(self) -> bool:
        try:
            svc = self._connect()
            svc.indexes.list()
            return True
        except Exception:
            return False

    def query(
        self,
        start_time: datetime,
        end_time: datetime,
        search_query: str | None = None,
        max_results: int = 500,
    ) -> list[LogEntry]:
        svc = self._connect()

        # Build SPL query
        base = f'search index="{self._index}"'
        if search_query:
            base += f" {search_query}"

        kwargs = {
            "earliest_time": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "latest_time": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "count": max_results,
            "output_mode": "json",
        }

        job = svc.jobs.oneshot(base, **kwargs)
        reader = splunk_results.JSONResultsReader(job)

        entries: list[LogEntry] = []
        for result in reader:
            if isinstance(result, splunk_results.Message):
                continue
            if isinstance(result, dict):
                entries.append(self._to_log_entry(result))

        return entries

    @staticmethod
    def _to_log_entry(raw_event: dict) -> LogEntry:
        raw_str = raw_event.get("_raw", json.dumps(raw_event))
        timestamp_str = raw_event.get("_time", "")
        try:
            ts = datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            ts = datetime.now()

        level = "INFO"
        for lvl in ("FATAL", "ERROR", "WARN", "WARNING", "DEBUG", "TRACE"):
            if lvl in raw_str.upper():
                level = lvl.replace("WARNING", "WARN")
                break

        return LogEntry(
            timestamp=ts,
            level=level,
            source=raw_event.get("source", raw_event.get("host", "splunk")),
            message=raw_str[:2000],
            raw=raw_str,
            metadata={
                k: v
                for k, v in raw_event.items()
                if not k.startswith("_") and k not in ("source", "host")
            },
        )
