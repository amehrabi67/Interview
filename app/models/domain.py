from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class QuestionEntry:
    """Represents a single interview question and its verified answer."""

    id: str
    question: str
    answer: str
    citations: List[str]
    key_points: List[str]


@dataclass
class ContentEvaluation:
    """Structured scoring output from the RAG-based content evaluation."""

    accuracy_score: float
    missed_concepts: List[str]
    errors_made: List[str]
    correct_points: List[str]
    confidence: float


@dataclass
class VisualAnalysis:
    """Summarises the student's delivery extracted from video analysis."""

    gestures_detected: List[str]
    confidence_estimate: str
    engagement_level: str
    notes: List[str] = field(default_factory=list)


@dataclass
class CompositeReport:
    """Final aggregated report combining content and delivery analysis."""

    question_asked: str
    student_answer_transcript: str
    content_analysis: ContentEvaluation
    delivery_analysis: VisualAnalysis
    composite_score: float


@dataclass
class ExamSession:
    """Stores session state for a student's interaction."""

    session_id: str
    question: QuestionEntry
    status: str = "awaiting_response"
    report: Optional[CompositeReport] = None
    transcript: Optional[str] = None
