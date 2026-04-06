"""Abstract base for all log sources."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LogEntry:
    """Normalised log record used across the entire system."""

    timestamp: datetime
    level: str  # e.g. INFO, WARN, ERROR, FATAL
    source: str  # application / host / index name
    message: str
    raw: str  # original unmodified line
    metadata: dict = field(default_factory=dict)


class LogSource(abc.ABC):
    """Every connector must implement these two methods."""

    @abc.abstractmethod
    def query(
        self,
        start_time: datetime,
        end_time: datetime,
        search_query: str | None = None,
        max_results: int = 500,
    ) -> list[LogEntry]:
        """Return log entries matching the time range and optional query."""

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Return True if the source is reachable."""
