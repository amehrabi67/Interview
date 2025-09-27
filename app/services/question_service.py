from __future__ import annotations

from uuid import uuid4

from app.config import Settings
from app.db.vector_store import QuestionVectorStore
from app.models.domain import ExamSession
from app.services.session import SessionStore


class QuestionService:
    """Handles retrieval of questions and creation of interview sessions."""

    def __init__(
        self,
        store: QuestionVectorStore,
        session_store: SessionStore,
        settings: Settings,
    ) -> None:
        self._store = store
        self._sessions = session_store
        self._settings = settings

    def start_new_session(self) -> ExamSession:
        question = self._store.random_question()
        session = ExamSession(session_id=uuid4().hex, question=question)
        self._sessions.add(session)
        return session

    def get_session(self, session_id: str) -> ExamSession:
        return self._sessions.get(session_id)

    @property
    def answer_duration_seconds(self) -> int:
        return self._settings.question_duration_seconds
