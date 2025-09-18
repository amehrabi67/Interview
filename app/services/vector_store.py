"""Lightweight vector store implementation for retrieval."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Iterable, List, Sequence

from app.models.schema import ContextChunk, Question


@dataclass(slots=True)
class _Document:
    """Internal document representation with cached statistics."""

    question: Question
    token_counts: Counter[str]
    norm: float


def _tokenise(text: str) -> List[str]:
    """Tokenise text using a simple whitespace and punctuation split."""

    cleaned = [
        char.lower() if char.isalnum() else " "
        for char in text
    ]
    tokens = [token for token in "".join(cleaned).split() if token]
    return tokens


class InMemoryVectorStore:
    """Naive vector store that relies on term-frequency cosine similarity."""

    def __init__(self, documents: Iterable[Question]):
        self._documents: List[_Document] = []
        for question in documents:
            combined_text = f"{question.prompt}\n{question.answer}"
            token_counts = Counter(_tokenise(combined_text))
            norm = sqrt(sum(count * count for count in token_counts.values())) or 1.0
            self._documents.append(_Document(question=question, token_counts=token_counts, norm=norm))

    def _build_query_vector(self, query: str) -> Counter[str]:
        return Counter(_tokenise(query))

    def _cosine_similarity(self, query_vector: Counter[str], document: _Document) -> float:
        dot_product = sum(query_vector[token] * document.token_counts.get(token, 0) for token in query_vector)
        query_norm = sqrt(sum(value * value for value in query_vector.values())) or 1.0
        return dot_product / (query_norm * document.norm)

    def retrieve(self, query: str, top_k: int) -> Sequence[ContextChunk]:
        """Return the top-K most relevant contexts for the supplied query."""

        query_vector = self._build_query_vector(query)
        scored = [
            (
                self._cosine_similarity(query_vector, document),
                document.question,
            )
            for document in self._documents
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        top_items = scored[:top_k]
        return [
            ContextChunk(
                id=question.id,
                text=question.answer,
                score=score,
                metadata={"prompt": question.prompt, "citations": question.citations},
            )
            for score, question in top_items
        ]

    def all_questions(self) -> List[Question]:
        """Expose all stored questions."""

        return [document.question for document in self._documents]
