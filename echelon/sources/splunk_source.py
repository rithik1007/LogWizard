"""Splunk log source connector — supports both REST API (token) and SDK (user/pass)."""

from __future__ import annotations

import json
from datetime import datetime

import httpx

from echelon.config import settings
from echelon.sources.base import LogEntry, LogSource


class SplunkSource(LogSource):
    """Connect to Splunk via the REST API (token auth over port 443) or the SDK."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        token: str | None = None,
        index: str | None = None,
        sourcetype: str | None = None,
    ):
        self._host = host or settings.splunk_host
        self._port = port or settings.splunk_port
        self._username = username or settings.splunk_username
        self._password = password or settings.splunk_password
        self._token = token or settings.splunk_token
        self._index = index or settings.splunk_index
        self._sourcetype = sourcetype or settings.splunk_sourcetype

    # ── REST API helpers (token auth) ─────────────────────────────

    def _base_url(self) -> str:
        scheme = settings.splunk_scheme or "https"
        if self._port in (443, 8089):
            return f"{scheme}://{self._host}"
        return f"{scheme}://{self._host}:{self._port}"

    def _http_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self._base_url(),
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            verify=False,
            timeout=httpx.Timeout(60.0, connect=15.0),
        )

    # ── SDK fallback (user/pass) ──────────────────────────────────

    def _connect_sdk(self):
        import splunklib.client as splunk_client

        return splunk_client.connect(
            host=self._host,
            port=self._port,
            username=self._username,
            password=self._password,
            scheme=settings.splunk_scheme,
            autologin=True,
        )

    # ── Public interface ──────────────────────────────────────────

    def health_check(self) -> bool:
        if self._token:
            return self._health_check_rest()
        return self._health_check_sdk()

    def _health_check_rest(self) -> bool:
        try:
            with self._http_client() as client:
                resp = client.get(
                    "/services/server/info", params={"output_mode": "json"}
                )
                return resp.status_code == 200
        except Exception:
            return False

    def _health_check_sdk(self) -> bool:
        try:
            svc = self._connect_sdk()
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
        if self._token:
            return self._query_rest(start_time, end_time, search_query, max_results)
        return self._query_sdk(start_time, end_time, search_query, max_results)

    # ── REST API query ────────────────────────────────────────────

    def _query_rest(
        self,
        start_time: datetime,
        end_time: datetime,
        search_query: str | None = None,
        max_results: int = 500,
    ) -> list[LogEntry]:
        # Build optimised SPL:
        #   - Filter by index + sourcetype first (most selective)
        #   - Append user search terms
        #   - Sort newest-first so the most recent events come back
        #   - Limit via head to avoid pulling more than needed
        spl = f'search index="{self._index}"'
        if self._sourcetype:
            spl += f' sourcetype="{self._sourcetype}"'
        if search_query:
            spl += f" {search_query}"
        spl += f" | sort - _time | head {max_results}"

        params = {
            "search": spl,
            "earliest_time": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "latest_time": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "count": str(max_results),
            "output_mode": "json",
        }

        with self._http_client() as client:
            resp = client.post(
                "/services/search/jobs/export",
                data=params,
            )
            resp.raise_for_status()

        entries: list[LogEntry] = []
        for line in resp.text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            # The export endpoint returns one JSON object per line.
            # Each object may contain "result" (single) or "results" (batch).
            if "result" in payload:
                entries.append(self._to_log_entry(payload["result"]))
            elif "results" in payload and isinstance(payload["results"], list):
                for event in payload["results"]:
                    entries.append(self._to_log_entry(event))

            if len(entries) >= max_results:
                break

        return entries[:max_results]

    # ── SDK query ─────────────────────────────────────────────────

    def _query_sdk(
        self,
        start_time: datetime,
        end_time: datetime,
        search_query: str | None = None,
        max_results: int = 500,
    ) -> list[LogEntry]:
        import splunklib.results as splunk_results

        svc = self._connect_sdk()

        spl = f'search index="{self._index}"'
        if self._sourcetype:
            spl += f' sourcetype="{self._sourcetype}"'
        if search_query:
            spl += f" {search_query}"

        kwargs = {
            "earliest_time": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "latest_time": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "count": max_results,
            "output_mode": "json",
        }

        job = svc.jobs.oneshot(spl, **kwargs)
        reader = splunk_results.JSONResultsReader(job)

        entries: list[LogEntry] = []
        for result in reader:
            if isinstance(result, splunk_results.Message):
                continue
            if isinstance(result, dict):
                entries.append(self._to_log_entry(result))

        return entries

    # ── Helpers ───────────────────────────────────────────────────

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
