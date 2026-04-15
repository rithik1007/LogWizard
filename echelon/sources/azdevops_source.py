"""
Azure DevOps integration — query pipeline runs, commits, and build status.
"""

from __future__ import annotations

import logging
from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import httpx

from echelon.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PipelineRun:
    id: int
    name: str
    status: str          # "completed", "inProgress", "cancelling", "notStarted"
    result: str          # "succeeded", "failed", "canceled", "partiallySucceeded"
    start_time: datetime | None
    finish_time: datetime | None
    source_branch: str
    source_version: str  # commit SHA
    requested_by: str
    pipeline_name: str
    url: str


@dataclass
class Commit:
    sha: str
    message: str
    author: str
    timestamp: datetime
    url: str


@dataclass
class BuildTimelineRecord:
    """A single record (stage, job, or task) from a build timeline."""
    id: str
    parent_id: str
    record_type: str   # "Stage", "Job", "Task"
    name: str
    state: str         # "completed", "inProgress", "pending"
    result: str        # "succeeded", "failed", "canceled", "skipped", etc.
    order: int
    start_time: datetime | None
    finish_time: datetime | None
    duration_ms: int
    error_count: int
    warning_count: int
    log_url: str


class AzDevOpsClient:
    """Client for Azure DevOps REST API."""

    def __init__(
        self,
        org: str | None = None,
        project: str | None = None,
        pat: str | None = None,
    ):
        self._org = org or settings.azdevops_org
        self._project = project or settings.azdevops_project
        self._pat = pat or settings.azdevops_pat
        self._base_url = f"https://dev.azure.com/{self._org}/{self._project}"
        self._api_version = "7.1"

    @property
    def is_configured(self) -> bool:
        return bool(self._org and self._project and self._pat)

    def _auth_header(self) -> dict[str, str]:
        token = b64encode(f":{self._pat}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        url = f"{self._base_url}/_apis/{path}"
        if params is None:
            params = {}
        params["api-version"] = self._api_version

        with httpx.Client(timeout=30, verify=True) as client:
            resp = client.get(url, headers=self._auth_header(), params=params)
            resp.raise_for_status()
            return resp.json()

    # ── Pipelines ──────────────────────────────────────────────

    def list_pipelines(self) -> list[dict]:
        """List all pipeline definitions."""
        if not self.is_configured:
            return []
        try:
            data = self._get("pipelines")
            return data.get("value", [])
        except Exception:
            logger.exception("Failed to list pipelines")
            return []

    def get_pipeline_runs(
        self,
        pipeline_id: int | None = None,
        pipeline_name: str | None = None,
        top: int = 20,
    ) -> list[PipelineRun]:
        """Get recent pipeline runs. Can filter by pipeline_id or pipeline_name."""
        if not self.is_configured:
            return []

        try:
            # If name given, resolve to ID
            if pipeline_name and not pipeline_id:
                pipelines = self.list_pipelines()
                for p in pipelines:
                    if pipeline_name.lower() in p.get("name", "").lower():
                        pipeline_id = p["id"]
                        break

            if pipeline_id:
                data = self._get(f"pipelines/{pipeline_id}/runs", {"$top": str(top)})
            else:
                # Fall back to build API for all pipelines
                data = self._get("build/builds", {
                    "$top": str(top),
                    "queryOrder": "finishTimeDescending",
                })
                return self._parse_builds(data.get("value", []))

            return self._parse_runs(data.get("value", []), pipeline_name or "")

        except Exception:
            logger.exception("Failed to get pipeline runs")
            return []

    def find_build_by_number(self, build_number: str, app: str | None = None) -> PipelineRun | None:
        """Look up a build by its human-readable build number (e.g. '2026.4.8.4').

        Azure DevOps API supports ``buildNumber`` as a filter parameter.
        """
        if not self.is_configured:
            return None
        try:
            params: dict[str, str] = {
                "buildNumber": build_number,
                "queryOrder": "finishTimeDescending",
                "$top": "5",
            }
            folder = settings.azdevops_folder
            if folder:
                params["path"] = folder
            data = self._get("build/builds", params)
            builds = self._parse_builds(data.get("value", []))

            # App-level filter if specified
            if app and builds:
                from echelon.config import APP_REGISTRY
                entry = APP_REGISTRY.get(app.lower(), {})
                keywords = [kw.lower() for kw in entry.get("pipelines", [])]
                if keywords:
                    builds = [
                        b for b in builds
                        if any(kw in b.pipeline_name.lower() for kw in keywords)
                    ]

            return builds[0] if builds else None
        except Exception:
            logger.exception("Failed to find build by number %s", build_number)
            return None

    def get_all_recent_builds(
        self,
        hours: int = 24,
        top: int = 50,
        folder: str | None = None,
        app: str | None = None,
    ) -> list[PipelineRun]:
        """Get recent builds, optionally filtered to a folder scope and/or app.

        Args:
            hours:  How far back to look.
            top:    Max builds to return.
            folder: Pipeline folder path (e.g. ``\\STEP-CI``).  Maps to the
                    Azure DevOps ``path`` query param so only definitions in
                    that folder (and sub-folders) are returned.
            app:    Optional app name (key in APP_REGISTRY).  When set, only
                    pipelines whose name contains one of the registered
                    ``pipelines`` substrings are returned.
        """
        if not self.is_configured:
            return []

        try:
            params: dict[str, str] = {
                "$top": str(top),
                "queryOrder": "finishTimeDescending",
            }
            # hours=0 means no time filter — just get the latest N builds
            if hours > 0:
                min_time = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
                params["minTime"] = min_time
            # Folder scope — keeps Helm / unrelated pipelines out
            effective_folder = folder or settings.azdevops_folder
            if effective_folder:
                params["path"] = effective_folder

            data = self._get("build/builds", params)
            builds = self._parse_builds(data.get("value", []))

            # App-level filter
            if app:
                from echelon.config import APP_REGISTRY
                entry = APP_REGISTRY.get(app.lower(), {})
                keywords = [kw.lower() for kw in entry.get("pipelines", [])]
                if keywords:
                    builds = [
                        b for b in builds
                        if any(kw in b.pipeline_name.lower() for kw in keywords)
                    ]

            return builds
        except Exception:
            logger.exception("Failed to get recent builds")
            return []

    # ── Build timeline / details ───────────────────────────────

    def get_build_timeline(self, build_id: int) -> list[BuildTimelineRecord]:
        """Get the timeline (stages, jobs, tasks) for a specific build.

        This is what you see when you click into a build run in Azure DevOps:
        stages → phases → jobs → individual tasks (Initialize job, Checkout, Build, etc.).

        Azure DevOps record types:
          Stage → Phase (logical job) → Job (agent job) → Task (individual step)
        """
        if not self.is_configured:
            return []

        try:
            data = self._get(f"build/builds/{build_id}/timeline")
            records = []
            for r in data.get("records", []):
                rtype = r.get("type", "")
                # Accept all meaningful record types
                if rtype not in ("Stage", "Phase", "Job", "Task", "Checkpoint"):
                    continue
                start = _parse_dt(r.get("startTime"))
                finish = _parse_dt(r.get("finishTime"))
                duration_ms = 0
                if start and finish:
                    duration_ms = int((finish - start).total_seconds() * 1000)
                records.append(BuildTimelineRecord(
                    id=r.get("id", ""),
                    parent_id=r.get("parentId", ""),
                    record_type=rtype,
                    name=r.get("name", ""),
                    state=r.get("state", "unknown"),
                    result=r.get("result", "unknown") if r.get("result") else "pending",
                    order=r.get("order", 0),
                    start_time=start,
                    finish_time=finish,
                    duration_ms=duration_ms,
                    error_count=r.get("errorCount", 0),
                    warning_count=r.get("warningCount", 0),
                    log_url=r.get("log", {}).get("url", "") if r.get("log") else "",
                ))
            # Sort by order so stages/jobs/tasks appear in execution order
            records.sort(key=lambda x: x.order)
            return records
        except Exception:
            logger.exception("Failed to get build timeline for build %s", build_id)
            return []

    def get_build_summary(self, build_id: int) -> dict:
        """Get a structured summary of a build — stages with their jobs and tasks.

        Returns a dict with build info + nested stages → jobs → tasks tree.
        """
        if not self.is_configured:
            return {"error": "Not configured"}

        try:
            # Get the build itself
            build_data = self._get(f"build/builds/{build_id}")
            build = self._parse_builds([build_data])[0] if build_data else None
            if not build:
                return {"error": f"Build {build_id} not found"}

            # Get the timeline
            records = self.get_build_timeline(build_id)

            # Build a lookup of all record IDs for ancestor traversal
            by_id = {r.id: r for r in records}

            # Categorise: Azure DevOps hierarchy is Stage → Phase → Job → Task
            stages = [r for r in records if r.record_type == "Stage"]
            phases = [r for r in records if r.record_type == "Phase"]
            jobs = [r for r in records if r.record_type == "Job"]
            tasks = [r for r in records if r.record_type == "Task"]

            def _find_ancestor_id(rec, target_type):
                """Walk up the parent chain to find the nearest ancestor of a given type."""
                visited = set()
                current = rec
                while current and current.parent_id and current.parent_id not in visited:
                    visited.add(current.parent_id)
                    parent = by_id.get(current.parent_id)
                    if not parent:
                        break
                    if parent.record_type == target_type:
                        return parent.id
                    current = parent
                return None

            stage_tree = []
            for stage in stages:
                # Find jobs under this stage (either direct children or via Phase)
                stage_jobs = [j for j in jobs if _find_ancestor_id(j, "Stage") == stage.id]
                # If no Job records, try Phase records as jobs
                if not stage_jobs:
                    stage_jobs = [p for p in phases if p.parent_id == stage.id]

                job_list = []
                for job in stage_jobs:
                    # Tasks: direct children of this job, or tasks whose Job ancestor is this
                    job_tasks = [t for t in tasks if t.parent_id == job.id]
                    if not job_tasks:
                        # Walk: find tasks whose nearest Job/Phase ancestor is this record
                        job_tasks = [t for t in tasks if _find_ancestor_id(t, job.record_type) == job.id]
                    job_list.append({
                        "name": job.name,
                        "state": job.state,
                        "result": job.result,
                        "duration_ms": job.duration_ms,
                        "error_count": job.error_count,
                        "warning_count": job.warning_count,
                        "tasks": [
                            {
                                "name": t.name,
                                "state": t.state,
                                "result": t.result,
                                "duration_ms": t.duration_ms,
                                "error_count": t.error_count,
                                "warning_count": t.warning_count,
                                "log_url": t.log_url,
                            }
                            for t in job_tasks
                        ],
                    })
                # If neither jobs nor phases found, attach tasks directly to stage
                if not job_list:
                    orphan_tasks = [t for t in tasks if _find_ancestor_id(t, "Stage") == stage.id]
                    if orphan_tasks:
                        job_list.append({
                            "name": stage.name,
                            "state": stage.state,
                            "result": stage.result,
                            "duration_ms": stage.duration_ms,
                            "error_count": stage.error_count,
                            "warning_count": stage.warning_count,
                            "tasks": [
                                {
                                    "name": t.name,
                                    "state": t.state,
                                    "result": t.result,
                                    "duration_ms": t.duration_ms,
                                    "error_count": t.error_count,
                                    "warning_count": t.warning_count,
                                    "log_url": t.log_url,
                                }
                                for t in orphan_tasks
                            ],
                        })
                stage_tree.append({
                    "name": stage.name,
                    "state": stage.state,
                    "result": stage.result,
                    "duration_ms": stage.duration_ms,
                    "jobs": job_list,
                })

            # If no stages at all, just list all tasks flat
            if not stage_tree and tasks:
                all_tasks_flat = [
                    {
                        "name": t.name,
                        "state": t.state,
                        "result": t.result,
                        "duration_ms": t.duration_ms,
                        "error_count": t.error_count,
                        "warning_count": t.warning_count,
                        "log_url": t.log_url,
                    }
                    for t in tasks
                ]
                stage_tree.append({
                    "name": "Build",
                    "state": tasks[0].state if tasks else "unknown",
                    "result": build.result,
                    "duration_ms": sum(t.duration_ms for t in tasks),
                    "jobs": [{"name": "Build", "state": build.status, "result": build.result,
                              "duration_ms": 0, "error_count": 0, "warning_count": 0,
                              "tasks": all_tasks_flat}],
                })

            # Collect failed task names for quick summary
            failed_tasks = []
            for s in stage_tree:
                for j in s.get("jobs", []):
                    for t in j.get("tasks", []):
                        if t.get("result") == "failed":
                            failed_tasks.append(t["name"])

            def _fmt_duration(ms: int) -> str:
                s = ms // 1000
                if s < 60:
                    return f"{s}s"
                return f"{s // 60}m {s % 60}s"

            return {
                "build_id": build.id,
                "name": build.name,
                "pipeline": build.pipeline_name,
                "status": build.status,
                "result": build.result,
                "branch": build.source_branch,
                "commit": build.source_version,
                "requested_by": build.requested_by,
                "start_time": build.start_time.isoformat() if build.start_time else None,
                "finish_time": build.finish_time.isoformat() if build.finish_time else None,
                "stages": stage_tree,
                "total_errors": sum(r.error_count for r in records),
                "total_warnings": sum(r.warning_count for r in records),
                "failed_tasks": failed_tasks,
                "record_type_counts": {
                    "stages": len(stages),
                    "phases": len(phases),
                    "jobs": len(jobs),
                    "tasks": len(tasks),
                    "total_records": len(records),
                },
            }
        except Exception:
            logger.exception("Failed to get build summary for %s", build_id)
            return {"error": f"Failed to fetch build {build_id}"}

    def get_build_log(self, log_url: str, tail: int = 80) -> str:
        """Fetch the raw log text for a build task.

        Args:
            log_url: The full log API URL from the timeline record.
            tail: Number of lines from the end to return (default 80).
        """
        if not log_url or not self.is_configured:
            return ""
        try:
            with httpx.Client(timeout=30, verify=True) as client:
                resp = client.get(log_url, headers=self._auth_header())
                resp.raise_for_status()
                lines = resp.text.strip().split("\n")
                # Return the last N lines (most relevant for failures)
                return "\n".join(lines[-tail:])
        except Exception:
            logger.exception("Failed to fetch build log from %s", log_url)
            return ""

    def _parse_runs(self, runs: list[dict], pipeline_name: str) -> list[PipelineRun]:
        results = []
        for r in runs:
            results.append(PipelineRun(
                id=r.get("id", 0),
                name=r.get("name", ""),
                status=r.get("state", r.get("status", "unknown")),
                result=r.get("result", "unknown"),
                start_time=_parse_dt(r.get("createdDate")),
                finish_time=_parse_dt(r.get("finishedDate")),
                source_branch=r.get("resources", {}).get("repositories", {}).get("self", {}).get("refName", ""),
                source_version=r.get("resources", {}).get("repositories", {}).get("self", {}).get("version", ""),
                requested_by=r.get("createdBy", {}).get("displayName", ""),
                pipeline_name=pipeline_name,
                url=r.get("_links", {}).get("web", {}).get("href", ""),
            ))
        return results

    def _parse_builds(self, builds: list[dict]) -> list[PipelineRun]:
        results = []
        for b in builds:
            results.append(PipelineRun(
                id=b.get("id", 0),
                name=b.get("buildNumber", ""),
                status=b.get("status", "unknown"),
                result=b.get("result", "unknown"),
                start_time=_parse_dt(b.get("startTime")),
                finish_time=_parse_dt(b.get("finishTime")),
                source_branch=b.get("sourceBranch", "").replace("refs/heads/", ""),
                source_version=b.get("sourceVersion", "")[:8],
                requested_by=b.get("requestedBy", {}).get("displayName", ""),
                pipeline_name=b.get("definition", {}).get("name", ""),
                url=b.get("_links", {}).get("web", {}).get("href", ""),
            ))
        return results

    # ── Commits ────────────────────────────────────────────────

    def get_recent_commits(
        self,
        repo_name: str | None = None,
        hours: int = 24,
        top: int = 20,
    ) -> list[Commit]:
        """Get recent commits from a repo (or first repo found)."""
        if not self.is_configured:
            return []

        try:
            # List repos
            repos_data = self._get("git/repositories")
            repos = repos_data.get("value", [])

            target_repo = None
            if repo_name:
                for r in repos:
                    if repo_name.lower() in r.get("name", "").lower():
                        target_repo = r
                        break
            if not target_repo and repos:
                target_repo = repos[0]

            if not target_repo:
                return []

            from_date = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
            repo_id = target_repo["id"]
            data = self._get(f"git/repositories/{repo_id}/commits", {
                "searchCriteria.fromDate": from_date,
                "$top": str(top),
            })

            commits = []
            for c in data.get("value", []):
                commits.append(Commit(
                    sha=c.get("commitId", "")[:8],
                    message=c.get("comment", "").split("\n")[0][:200],
                    author=c.get("author", {}).get("name", ""),
                    timestamp=_parse_dt(c.get("author", {}).get("date")) or datetime.now(),
                    url=c.get("remoteUrl", ""),
                ))
            return commits

        except Exception:
            logger.exception("Failed to get commits")
            return []

    # ── Health check ───────────────────────────────────────────

    def health_check(self) -> dict:
        if not self.is_configured:
            return {"connected": False, "reason": "Not configured — set AZDEVOPS_ORG, AZDEVOPS_PROJECT, AZDEVOPS_PAT"}
        try:
            data = self._get("build/builds", {"$top": "1"})
            return {"connected": True, "org": self._org, "project": self._project}
        except Exception as e:
            return {"connected": False, "reason": str(e)}


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


# Module-level singleton
_client: AzDevOpsClient | None = None


def get_azdevops_client() -> AzDevOpsClient:
    global _client
    if _client is None:
        _client = AzDevOpsClient()
    return _client
