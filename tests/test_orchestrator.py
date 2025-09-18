import asyncio
from pathlib import Path

from app.config import settings
from app.orchestrator import AssessmentOrchestrator, SessionConfig
from app.services.question_bank import QuestionRepository, QuestionService
from app.services.rag import RAGEvaluator
from app.services.recorder import MockAudioVideoRecorder
from app.services.reporting import ReportSynthesizer
from app.services.transcription import FallbackTranscriptionService
from app.services.visual import VisualAnalyzer


async def _run_assessment(transcript: str, gestures: list[str], question_id: str):
    repository = QuestionRepository(Path("app/data/linear_regression_questions.json"))
    question_service = QuestionService(repository)
    recorder = MockAudioVideoRecorder(transcript=transcript, gestures=gestures)
    transcription = FallbackTranscriptionService()
    rag = RAGEvaluator(question_service.vector_store, settings.rag_top_k)
    visual = VisualAnalyzer(settings.engagement_scale)
    reporter = ReportSynthesizer(settings.content_weight, settings.delivery_weight)

    orchestrator = AssessmentOrchestrator(
        question_service=question_service,
        recorder=recorder,
        transcription=transcription,
        rag_evaluator=rag,
        visual_analyzer=visual,
        reporter=reporter,
    )

    return await orchestrator.conduct_assessment(SessionConfig(question_id=question_id))


def test_orchestrator_generates_report():
    transcript = (
        "Least squares minimises squared residuals and solves normal equations, giving the best linear unbiased estimators."
    )
    report = asyncio.run(_run_assessment(transcript, ["counting", "open_hand"], "q1"))

    assert report.question_asked.id == "q1"
    assert "residuals" in report.student_answer_transcript.lower()
    assert report.content_analysis.accuracy_score > 0
    assert report.delivery_analysis.gestures_detected
    assert report.composite_score > 0
