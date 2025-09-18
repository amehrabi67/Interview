"""Synthesis utilities for combining assessment results."""
from __future__ import annotations

from typing import Dict
from app.models.schema import AssessmentReport, ContentEvaluation, Question, VisualAnalysis


class ReportSynthesizer:
    """Combine content and delivery metrics into a single report."""

    def __init__(self, content_weight: float, delivery_weight: float):
        total = content_weight + delivery_weight
        if total == 0:
            raise ValueError("Weights must sum to a positive value.")
        self._weights = {
            "content": content_weight / total,
            "delivery": delivery_weight / total,
        }
        self._engagement_scores: Dict[str, float] = {
            "low": 0.25,
            "medium": 0.6,
            "high": 0.9,
        }

    def _delivery_score(self, visual_analysis: VisualAnalysis) -> float:
        return self._engagement_scores.get(visual_analysis.engagement_level, 0.4)

    def synthesize(
        self,
        question: Question,
        transcript: str,
        content_analysis: ContentEvaluation,
        delivery_analysis: VisualAnalysis,
    ) -> AssessmentReport:
        content_component = content_analysis.accuracy_score / 100
        delivery_component = self._delivery_score(delivery_analysis)
        composite = (
            content_component * self._weights["content"]
            + delivery_component * self._weights["delivery"]
        )
        composite_score = round(composite * 100, 2)
        return AssessmentReport(
            question_asked=question,
            student_answer_transcript=transcript,
            content_analysis=content_analysis,
            delivery_analysis=delivery_analysis,
            composite_score=composite_score,
        )
