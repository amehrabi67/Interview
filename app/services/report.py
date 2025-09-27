from __future__ import annotations

from app.config import Settings
from app.models.domain import CompositeReport, ContentEvaluation, VisualAnalysis


class ReportBuilder:
    """Combines content and delivery analysis into a single JSON-friendly report."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(
        self,
        question_text: str,
        transcript: str,
        content: ContentEvaluation,
        visual: VisualAnalysis,
    ) -> CompositeReport:
        delivery_score = self._delivery_score(visual)
        composite = (
            content.accuracy_score * self._settings.composite_score_content_weight
            + delivery_score * self._settings.composite_score_delivery_weight
        )
        composite = max(0.0, min(100.0, composite))
        return CompositeReport(
            question_asked=question_text,
            student_answer_transcript=transcript,
            content_analysis=content,
            delivery_analysis=visual,
            composite_score=round(composite, 2),
        )

    def _delivery_score(self, visual: VisualAnalysis) -> float:
        base = {"low": 40.0, "medium": 70.0, "high": 90.0}.get(visual.engagement_level, 50.0)
        bonus = min(10.0, 2.5 * len(visual.gestures_detected))
        if visual.confidence_estimate == "high":
            bonus += 5.0
        if "analysis_unavailable" in visual.gestures_detected:
            return 50.0
        return max(30.0, min(95.0, base + bonus))
