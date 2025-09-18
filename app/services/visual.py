"""Visual analysis utilities."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List

from app.models.schema import KeyFrame, MediaCapture, VisualAnalysis


class VisualAnalyzer:
    """Aggregate gesture metadata and produce qualitative assessments."""

    def __init__(self, engagement_thresholds: Iterable[str]):
        self._engagement_scale = list(engagement_thresholds)

    def _aggregate_gestures(self, key_frames: List[KeyFrame]) -> Counter:
        counter: Counter[str] = Counter()
        for frame in key_frames:
            counter.update(frame.precomputed_gestures)
        return counter

    def _estimate_confidence(self, gesture_counts: Counter[str]) -> str:
        if gesture_counts.get("open_hand", 0) + gesture_counts.get("counting", 0) >= 2:
            return "high"
        if gesture_counts.get("thinking", 0) >= 1 or gesture_counts.get("fidget", 0) >= 2:
            return "medium"
        return "low"

    def _estimate_engagement(self, gesture_counts: Counter[str]) -> str:
        if gesture_counts.get("counting", 0) or gesture_counts.get("diagramming", 0):
            return "high"
        if gesture_counts.get("eye_contact", 0):
            return "medium"
        return "low"

    def analyze(self, capture: MediaCapture) -> VisualAnalysis:
        gesture_counts = self._aggregate_gestures(capture.key_frames)
        gestures = sorted(gesture_counts)
        confidence = self._estimate_confidence(gesture_counts)
        engagement = self._estimate_engagement(gesture_counts)
        if self._engagement_scale and engagement not in self._engagement_scale:
            engagement = self._engagement_scale[0]
        notes = "; ".join(
            f"{gesture}: {count}"
            for gesture, count in gesture_counts.items()
        ) or None
        return VisualAnalysis(
            gestures_detected=gestures,
            confidence_estimate=confidence,
            engagement_level=engagement,
            notes=notes,
        )
