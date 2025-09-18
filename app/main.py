"""FastAPI application exposing the multimodal assessment workflow."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import settings
from app.models.schema import AssessmentReport, Question
from app.orchestrator import AssessmentOrchestrator, SessionConfig
from app.services.question_bank import QuestionRepository, QuestionService
from app.services.rag import RAGEvaluator
from app.services.recorder import MockAudioVideoRecorder
from app.services.reporting import ReportSynthesizer
from app.services.transcription import FallbackTranscriptionService
from app.services.visual import VisualAnalyzer


class AssessmentRequest(BaseModel):
    """Payload submitted when requesting an automated assessment."""

    question_id: str | None = None
    manual_transcript: str | None = None
    gestures: list[str] | None = None


class AssessmentResponse(BaseModel):
    """Wrapper response returned by the API."""

    report: AssessmentReport


repository = QuestionRepository(settings.question_dataset)
question_service = QuestionService(repository)
transcription_service = FallbackTranscriptionService()
rag_evaluator = RAGEvaluator(question_service.vector_store, settings.rag_top_k)
visual_analyzer = VisualAnalyzer(settings.engagement_scale)
report_synthesizer = ReportSynthesizer(settings.content_weight, settings.delivery_weight)


def _build_orchestrator(transcript: str, gestures: list[str]) -> AssessmentOrchestrator:
    recorder = MockAudioVideoRecorder(transcript=transcript, gestures=gestures)
    return AssessmentOrchestrator(
        question_service=question_service,
        recorder=recorder,
        transcription=transcription_service,
        rag_evaluator=rag_evaluator,
        visual_analyzer=visual_analyzer,
        reporter=report_synthesizer,
    )


app = FastAPI(title="Linear Regression Oral Assessment")


@app.get("/question", response_model=Question)
async def fetch_question() -> Question:
    """Return a random question from the repository."""

    return question_service.get_random_question()


@app.post("/assess", response_model=AssessmentResponse)
async def assess(request: AssessmentRequest) -> AssessmentResponse:
    """Execute the full assessment workflow."""

    manual_transcript = request.manual_transcript or ""
    gestures = request.gestures or []
    orchestrator = _build_orchestrator(manual_transcript, gestures)
    report = await orchestrator.conduct_assessment(
        SessionConfig(question_id=request.question_id, manual_transcript=request.manual_transcript)
    )
    return AssessmentResponse(report=report)
