from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QuestionResponse(BaseModel):
    session_id: str = Field(..., description="Identifier for the exam session")
    question: str = Field(..., description="Question presented to the student")
    max_response_duration: int = Field(..., description="Allowed answer duration in seconds")


class SubmissionResponse(BaseModel):
    session_id: str
    status: str


class ContentEvaluationModel(BaseModel):
    accuracy_score: float = Field(..., ge=0, le=100)
    missed_concepts: List[str]
    errors_made: List[str]
    correct_points: List[str]
    confidence: float = Field(..., ge=0, le=100)


class VisualAnalysisModel(BaseModel):
    gestures_detected: List[str]
    confidence_estimate: str
    engagement_level: str
    notes: List[str] = Field(default_factory=list)


class ExamReportModel(BaseModel):
    question_asked: str
    student_answer_transcript: str
    content_analysis: ContentEvaluationModel
    delivery_analysis: VisualAnalysisModel
    composite_score: float = Field(..., ge=0, le=100)


class ExamStatusResponse(BaseModel):
    session_id: str
    status: str
    report: Optional[ExamReportModel] = None


class TranscriptOnlyRequest(BaseModel):
    session_id: str
    transcript: str
