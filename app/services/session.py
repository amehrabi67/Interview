from __future__ import annotations

import json
from dataclasses import asdict
from threading import Lock
from typing import Dict, Optional, TYPE_CHECKING

from app.models.domain import (
    CompositeReport,
    ContentEvaluation,
    ExamSession,
    QuestionEntry,
    VisualAnalysis,
)

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from redis.client import Redis


class SessionStore:
    """Thread-safe in-memory session store suitable for prototyping."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ExamSession] = {}
        self._lock = Lock()

    def add(self, session: ExamSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def get(self, session_id: str) -> ExamSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session_id {session_id}")
            return self._sessions[session_id]

    def update_status(self, session_id: str, status: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Unknown session_id {session_id}")
            session.status = status

    def attach_transcript(self, session_id: str, transcript: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Unknown session_id {session_id}")
            session.transcript = transcript

    def attach_report(self, session_id: str, report: CompositeReport) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Unknown session_id {session_id}")
            session.report = report
            session.status = "completed"

    def maybe_get(self, session_id: str) -> Optional[ExamSession]:
        with self._lock:
            return self._sessions.get(session_id)


class RedisSessionStore(SessionStore):
    """Redis-backed session store that persists state across processes."""

    def __init__(self, redis_client: "Redis", key_prefix: str = "session:") -> None:
        # Intentionally skip super().__init__ to avoid the in-memory structures.
        self._redis = redis_client
        self._key_prefix = key_prefix

    def add(self, session: ExamSession) -> None:
        mapping = {
            "question": json.dumps(asdict(session.question)),
            "status": session.status,
            "transcript": session.transcript or "",
            "report": json.dumps(asdict(session.report)) if session.report else "",
        }
        self._redis.hset(self._key(session.session_id), mapping=mapping)

    def get(self, session_id: str) -> ExamSession:
        data = self._redis.hgetall(self._key(session_id))
        if not data:
            raise KeyError(f"Unknown session_id {session_id}")
        return self._deserialize_session(session_id, data)

    def update_status(self, session_id: str, status: str) -> None:
        key = self._ensure_exists(session_id)
        self._redis.hset(key, mapping={"status": status})

    def attach_transcript(self, session_id: str, transcript: str) -> None:
        key = self._ensure_exists(session_id)
        self._redis.hset(key, mapping={"transcript": transcript})

    def attach_report(self, session_id: str, report: CompositeReport) -> None:
        key = self._ensure_exists(session_id)
        payload = json.dumps(asdict(report))
        self._redis.hset(key, mapping={"report": payload, "status": "completed"})

    def maybe_get(self, session_id: str) -> Optional[ExamSession]:
        data = self._redis.hgetall(self._key(session_id))
        if not data:
            return None
        return self._deserialize_session(session_id, data)

    def _key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    def _ensure_exists(self, session_id: str) -> str:
        key = self._key(session_id)
        if not self._redis.exists(key):
            raise KeyError(f"Unknown session_id {session_id}")
        return key

    def _deserialize_session(self, session_id: str, data: Dict[str, str]) -> ExamSession:
        question = QuestionEntry(**json.loads(data["question"]))
        report = self._deserialize_report(data.get("report"))
        transcript = data.get("transcript") or None
        status = data.get("status", "awaiting_response")
        return ExamSession(
            session_id=session_id,
            question=question,
            status=status,
            report=report,
            transcript=transcript,
        )

    def _deserialize_report(self, payload: Optional[str]) -> Optional[CompositeReport]:
        if not payload:
            return None
        data = json.loads(payload)
        content = ContentEvaluation(**data["content_analysis"])
        delivery = VisualAnalysis(**data["delivery_analysis"])
        return CompositeReport(
            question_asked=data["question_asked"],
            student_answer_transcript=data["student_answer_transcript"],
            content_analysis=content,
            delivery_analysis=delivery,
            composite_score=data["composite_score"],
        )
