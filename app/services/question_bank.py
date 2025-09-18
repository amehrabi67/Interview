"""Services responsible for managing assessment questions."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List, Optional

from app.models.schema import Question
from app.services.vector_store import InMemoryVectorStore


class QuestionRepository:
    """Loads questions from disk."""

    def __init__(self, dataset_path: Path):
        self._dataset_path = dataset_path
        self._questions: List[Question] = []

    def load(self) -> List[Question]:
        if not self._questions:
            data = json.loads(self._dataset_path.read_text())
            self._questions = [
                Question(id=item["id"], prompt=item["question"], answer=item["answer"], citations=item.get("citations", []))
                for item in data
            ]
        return self._questions


class QuestionService:
    """High level API for selecting questions and initialising the vector store."""

    def __init__(self, repository: QuestionRepository, rng: Optional[random.Random] = None):
        self._repository = repository
        self._rng = rng or random.Random()
        self._vector_store: Optional[InMemoryVectorStore] = None

    @property
    def vector_store(self) -> InMemoryVectorStore:
        if self._vector_store is None:
            self._vector_store = InMemoryVectorStore(self._repository.load())
        return self._vector_store

    def get_random_question(self) -> Question:
        questions = self._repository.load()
        return self._rng.choice(questions)

    def get_question(self, question_id: Optional[str]) -> Question:
        if question_id is None:
            return self.get_random_question()
        for question in self._repository.load():
            if question.id == question_id:
                return question
        raise ValueError(f"Unknown question id: {question_id}")

    def all_questions(self) -> Iterable[Question]:
        return self._repository.load()
