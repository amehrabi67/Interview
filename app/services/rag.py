"""RAG pipeline components."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List

from app.models.schema import ContentEvaluation, ContextChunk, Question
from app.services.vector_store import InMemoryVectorStore

COMMON_MISCONCEPTIONS = {
    "correlation implies causation": "Incorrectly equates correlation with causation.",
    "dependent variable causes": "Misstates causality direction.",
    "ignoring residuals": "Fails to consider residual analysis when evaluating model fit.",
}


@dataclass(slots=True)
class EvaluationArtifacts:
    """Intermediate artifacts produced during evaluation."""

    contexts: List[ContextChunk]
    key_points: List[str]


def _split_into_key_points(reference_answer: str) -> List[str]:
    sentences = re.split(r"[\.;]\s+", reference_answer)
    key_points = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not key_points:
        key_points = [reference_answer.strip()]
    return key_points


def _point_covered(point: str, transcript: str) -> bool:
    point_tokens = set(token for token in re.findall(r"[a-zA-Z]+", point.lower()) if len(token) > 2)
    if not point_tokens:
        return False
    transcript_tokens = set(re.findall(r"[a-zA-Z]+", transcript.lower()))
    intersection = point_tokens & transcript_tokens
    return len(intersection) / len(point_tokens) >= 0.3


class RAGEvaluator:
    """Compare a student's response against retrieved authoritative answers."""

    def __init__(self, vector_store: InMemoryVectorStore, top_k: int):
        self._vector_store = vector_store
        self._top_k = top_k

    def _retrieve_context(self, transcript: str) -> EvaluationArtifacts:
        contexts = list(self._vector_store.retrieve(transcript, self._top_k))
        reference_answer = contexts[0].text if contexts else ""
        key_points = _split_into_key_points(reference_answer)
        return EvaluationArtifacts(contexts=contexts, key_points=key_points)

    def evaluate(self, question: Question, transcript: str) -> ContentEvaluation:
        artifacts = self._retrieve_context(transcript or question.prompt)
        key_points = artifacts.key_points
        covered_points = [point for point in key_points if _point_covered(point, transcript)]
        missed_points = [point for point in key_points if point not in covered_points]

        accuracy = (len(covered_points) / len(key_points)) if key_points else 0.0
        accuracy_score = round(accuracy * 100, 2)

        errors = [
            description
            for phrase, description in COMMON_MISCONCEPTIONS.items()
            if phrase in transcript.lower()
        ]

        confidence = round(min(1.0, accuracy + 0.5 * math.exp(-len(missed_points))) * 100, 2)

        return ContentEvaluation(
            accuracy_score=accuracy_score,
            missed_concepts=missed_points,
            errors_made=errors,
            correct_points=covered_points,
            confidence=confidence,
            context_used=artifacts.contexts,
        )
