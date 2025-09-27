from __future__ import annotations

from threading import Lock
from typing import Dict, Optional

from app.models.domain import CompositeReport, ExamSession


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
