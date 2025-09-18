"""High-level orchestration of the interview pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config import settings
from app.models.schema import AssessmentReport, MediaCapture, Question
from app.services.question_bank import QuestionService
from app.services.rag import RAGEvaluator
from app.services.recorder import AudioVideoRecorder
from app.services.reporting import ReportSynthesizer
from app.services.transcription import TranscriptionService
from app.services.visual import VisualAnalyzer


@dataclass(slots=True)
class SessionConfig:
    """Runtime configuration for a single assessment session."""

    question_id: Optional[str] = None
    duration_override: Optional[int] = None
    manual_transcript: Optional[str] = None
    media_capture: Optional[MediaCapture] = None


class AssessmentOrchestrator:
    """Coordinate the end-to-end assessment workflow."""

    def __init__(
        self,
        question_service: QuestionService,
        recorder: AudioVideoRecorder,
        transcription: TranscriptionService,
        rag_evaluator: RAGEvaluator,
        visual_analyzer: VisualAnalyzer,
        reporter: ReportSynthesizer,
    ):
        self._question_service = question_service
        self._recorder = recorder
        self._transcription = transcription
        self._rag = rag_evaluator
        self._visual = visual_analyzer
        self._reporter = reporter

    async def conduct_assessment(self, config: Optional[SessionConfig] = None) -> AssessmentReport:
        config = config or SessionConfig()
        question = self._question_service.get_question(config.question_id)

        if config.media_capture is not None:
            capture = config.media_capture
        else:
            duration = config.duration_override or settings.recording_duration_seconds
            capture = self._recorder.record(question.prompt, duration)

        if config.manual_transcript:
            transcript = config.manual_transcript
        elif capture.transcript:
            transcript = capture.transcript
        elif capture.audio_path is not None:
            transcript = await self._transcription.transcribe(capture.audio_path)
        else:
            raise ValueError("No transcript available for evaluation.")

        content_eval = self._rag.evaluate(question, transcript)
        visual_eval = self._visual.analyze(capture)
        report = self._reporter.synthesize(question, transcript, content_eval, visual_eval)
        return report

    def random_question(self) -> Question:
        return self._question_service.get_random_question()
