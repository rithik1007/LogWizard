"""
Knowledge Base — vector store backed by ChromaDB.

Stores four collections:
  1. error_patterns   — known error signatures + whether they're actionable
  2. incidents        — past analysed incidents with root cause & resolution
  3. log_embeddings   — raw log chunks for semantic retrieval
  4. user_feedback    — user-reported known issues, dismissed errors, notes
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings

from echelon.config import settings


def _chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


class KnowledgeBase:
    ERRORS_COLLECTION = "error_patterns"
    INCIDENTS_COLLECTION = "incidents"
    LOGS_COLLECTION = "log_embeddings"
    FEEDBACK_COLLECTION = "user_feedback"
    CLASSIFICATIONS_COLLECTION = "error_classifications"

    def __init__(self):
        self._client = _chroma_client()
        self._errors = self._client.get_or_create_collection(self.ERRORS_COLLECTION)
        self._incidents = self._client.get_or_create_collection(self.INCIDENTS_COLLECTION)
        self._logs = self._client.get_or_create_collection(self.LOGS_COLLECTION)
        self._feedback = self._client.get_or_create_collection(self.FEEDBACK_COLLECTION)
        self._classifications = self._client.get_or_create_collection(self.CLASSIFICATIONS_COLLECTION)

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

    # ── User Feedback / Known Issues ───────────────────────────────

    def store_feedback(
        self,
        error_pattern: str,
        feedback_type: str,
        application: str = "",
        user_note: str = "",
        resolution: str = "",
    ) -> str:
        """Store user feedback about an error pattern.

        feedback_type: 'known_issue', 'expected', 'noise', 'resolved', 'critical'
        """
        doc_id = uuid.uuid4().hex[:12]
        document = (
            f"Error: {error_pattern}\n"
            f"Feedback: {feedback_type}\n"
            f"Note: {user_note}"
        )
        self._feedback.add(
            ids=[doc_id],
            documents=[document],
            metadatas=[
                {
                    "error_pattern": error_pattern,
                    "feedback_type": feedback_type,
                    "application": application,
                    "user_note": user_note,
                    "resolution": resolution,
                    "created_at": datetime.now().isoformat(),
                }
            ],
        )
        return doc_id

    def search_feedback(self, query: str, top_k: int = 10) -> list[dict]:
        results = self._feedback.query(query_texts=[query], n_results=top_k)
        return self._format_results(results)

    def list_all_feedback(self) -> list[dict]:
        """Return all feedback entries (known issues, dismissed errors, etc.)."""
        count = self._feedback.count()
        if count == 0:
            return []
        results = self._feedback.get(limit=count)
        items = []
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        for i, doc_id in enumerate(ids):
            items.append({
                "id": doc_id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
            })
        return items

    def delete_feedback(self, doc_id: str) -> bool:
        """Remove a feedback entry by ID."""
        try:
            self._feedback.delete(ids=[doc_id])
            return True
        except Exception:
            return False

    # ── Error Classifications (AI-generated) ─────────────────────────

    def store_classification(
        self,
        error_pattern: str,
        classification: str,
        severity: str,
        confidence: float,
        application: str = "",
        environment: str = "",
        root_cause: str = "",
        suggested_action: str = "",
        occurrence_count: int = 1,
    ) -> str:
        """Store an AI-generated error classification.

        classification: 'actionable', 'noise', 'transient', 'known_issue', 'configuration', 'dependency'
        severity: 'critical', 'high', 'medium', 'low', 'noise'
        confidence: 0.0-1.0 how confident the AI is in this classification
        """
        doc_id = uuid.uuid4().hex[:12]
        document = (
            f"Error: {error_pattern}\n"
            f"Classification: {classification}\n"
            f"Severity: {severity}\n"
            f"Root cause: {root_cause}"
        )
        self._classifications.upsert(
            ids=[doc_id],
            documents=[document],
            metadatas=[
                {
                    "error_pattern": error_pattern[:500],
                    "classification": classification,
                    "severity": severity,
                    "confidence": str(confidence),
                    "application": application,
                    "environment": environment,
                    "root_cause": root_cause[:500],
                    "suggested_action": suggested_action[:500],
                    "occurrence_count": str(occurrence_count),
                    "first_seen": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat(),
                    "times_classified": "1",
                }
            ],
        )
        return doc_id

    def search_classifications(self, query: str, top_k: int = 5) -> list[dict]:
        results = self._classifications.query(query_texts=[query], n_results=top_k)
        return self._format_results(results)

    def list_all_classifications(self, application: str = "") -> list[dict]:
        count = self._classifications.count()
        if count == 0:
            return []
        results = self._classifications.get(limit=count)
        items = []
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        for i, doc_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            if application and meta.get("application", "").lower() != application.lower():
                continue
            items.append({
                "id": doc_id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": meta,
            })
        return items

    def update_classification_count(self, doc_id: str, new_count: int) -> bool:
        """Increment occurrence count and update last_seen for a classification."""
        try:
            existing = self._classifications.get(ids=[doc_id])
            if existing and existing["ids"]:
                meta = existing["metadatas"][0]
                meta["occurrence_count"] = str(new_count)
                meta["last_seen"] = datetime.now().isoformat()
                meta["times_classified"] = str(int(meta.get("times_classified", "1")) + 1)
                self._classifications.update(ids=[doc_id], metadatas=[meta])
                return True
        except Exception:
            pass
        return False

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
            "user_feedback": self._feedback.count(),
            "error_classifications": self._classifications.count(),
        }
