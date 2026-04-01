"""
Knowledge Base — vector store backed by ChromaDB.

Stores three collections:
  1. error_patterns   — known error signatures + whether they're actionable
  2. incidents        — past analysed incidents with root cause & resolution
  3. log_embeddings   — raw log chunks for semantic retrieval
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings

from logwizard.config import settings


def _chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


class KnowledgeBase:
    ERRORS_COLLECTION = "error_patterns"
    INCIDENTS_COLLECTION = "incidents"
    LOGS_COLLECTION = "log_embeddings"

    def __init__(self):
        self._client = _chroma_client()
        self._errors = self._client.get_or_create_collection(self.ERRORS_COLLECTION)
        self._incidents = self._client.get_or_create_collection(self.INCIDENTS_COLLECTION)
        self._logs = self._client.get_or_create_collection(self.LOGS_COLLECTION)

    # ── Error Patterns ─────────────────────────────────────────────

    def store_error_pattern(
        self,
        pattern: str,
        is_actionable: bool,
        category: str = "",
        notes: str = "",
    ) -> str:
        doc_id = uuid.uuid4().hex[:12]
        self._errors.add(
            ids=[doc_id],
            documents=[pattern],
            metadatas=[
                {
                    "is_actionable": str(is_actionable),
                    "category": category,
                    "notes": notes,
                    "created_at": datetime.now().isoformat(),
                }
            ],
        )
        return doc_id

    def search_error_patterns(self, query: str, top_k: int = 5) -> list[dict]:
        results = self._errors.query(query_texts=[query], n_results=top_k)
        return self._format_results(results)

    # ── Incidents ──────────────────────────────────────────────────

    def store_incident(
        self,
        summary: str,
        root_cause: str,
        resolution: str,
        severity: str = "unknown",
        tags: list[str] | None = None,
    ) -> str:
        doc_id = uuid.uuid4().hex[:12]
        document = (
            f"Summary: {summary}\n"
            f"Root Cause: {root_cause}\n"
            f"Resolution: {resolution}"
        )
        self._incidents.add(
            ids=[doc_id],
            documents=[document],
            metadatas=[
                {
                    "summary": summary,
                    "root_cause": root_cause,
                    "resolution": resolution,
                    "severity": severity,
                    "tags": json.dumps(tags or []),
                    "created_at": datetime.now().isoformat(),
                }
            ],
        )
        return doc_id

    def search_incidents(self, query: str, top_k: int = 5) -> list[dict]:
        results = self._incidents.query(query_texts=[query], n_results=top_k)
        return self._format_results(results)

    # ── Raw log embeddings ─────────────────────────────────────────

    def store_log_chunk(self, chunk: str, source: str, timestamp: str) -> str:
        doc_id = uuid.uuid4().hex[:12]
        self._logs.add(
            ids=[doc_id],
            documents=[chunk],
            metadatas=[{"source": source, "timestamp": timestamp}],
        )
        return doc_id

    def search_logs(self, query: str, top_k: int = 10) -> list[dict]:
        results = self._logs.query(query_texts=[query], n_results=top_k)
        return self._format_results(results)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _format_results(results: dict) -> list[dict]:
        formatted = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            formatted.append(
                {
                    "id": doc_id,
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                }
            )
        return formatted

    def get_stats(self) -> dict:
        return {
            "error_patterns": self._errors.count(),
            "incidents": self._incidents.count(),
            "log_chunks": self._logs.count(),
        }
