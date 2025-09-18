"""Pydantic models describing core entities for the assessment pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Question(BaseModel):
    """Representation of an oral examination question."""

    id: str
    prompt: str
    answer: str
    citations: List[str] = Field(default_factory=list)


class ContextChunk(BaseModel):
    """Chunk retrieved from the vector store during RAG retrieval."""

    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContentEvaluation(BaseModel):
    """LLM-backed evaluation of a student's answer."""

    accuracy_score: float
    missed_concepts: List[str]
    errors_made: List[str]
    correct_points: List[str]
    confidence: float
    context_used: List[ContextChunk] = Field(default_factory=list)


class KeyFrame(BaseModel):
    """A single key frame extracted from the student's video recording."""

    timestamp: float
    image_path: Optional[str] = None
    precomputed_gestures: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class MediaCapture(BaseModel):
    """Container describing recorded media assets."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_path: Optional[Path] = None
    video_path: Optional[Path] = None
    transcript: Optional[str] = None
    key_frames: List[KeyFrame] = Field(default_factory=list)


class VisualAnalysis(BaseModel):
    """Structured interpretation of non-verbal communication."""

    gestures_detected: List[str]
    confidence_estimate: str
    engagement_level: str
    notes: Optional[str] = None


class AssessmentReport(BaseModel):
    """Final report synthesising content and delivery analysis."""

    question_asked: Question
    student_answer_transcript: str
    content_analysis: ContentEvaluation
    delivery_analysis: VisualAnalysis
    composite_score: float
