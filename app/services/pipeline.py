from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.models.domain import VisualAnalysis
from app.services.rag import RAGEvaluator
from app.services.report import ReportBuilder
from app.services.session import SessionStore
from app.services.transcription import TranscriptionService
from app.services.visual import VisualAnalyzer


class AssessmentPipeline:
    """Coordinates audio transcription, RAG evaluation, and visual analysis."""

    def __init__(
        self,
        sessions: SessionStore,
        transcription: TranscriptionService,
        rag: RAGEvaluator,
        visual: VisualAnalyzer,
        report_builder: ReportBuilder,
    ) -> None:
        self._sessions = sessions
        self._transcription = transcription
        self._rag = rag
        self._visual = visual
        self._report_builder = report_builder

    def process(
        self,
        session_id: str,
        audio_path: Optional[Path] = None,
        video_path: Optional[Path] = None,
        transcript: Optional[str] = None,
    ) -> None:
        session = self._sessions.get(session_id)
        self._sessions.update_status(session_id, "processing")

        if transcript is None:
            if audio_path is None:
                raise ValueError("Either transcript or audio_path must be provided")
            transcript = self._transcription.transcribe(audio_path)
        self._sessions.attach_transcript(session_id, transcript)

        content = self._rag.evaluate(session.question, transcript)
        visual = self._analyze_video(video_path)
        report = self._report_builder.build(
            question_text=session.question.question,
            transcript=transcript,
            content=content,
            visual=visual,
        )
        self._sessions.attach_report(session_id, report)
        self._cleanup_file(audio_path)
        self._cleanup_file(video_path)

    def _analyze_video(self, video_path: Optional[Path]) -> VisualAnalysis:
        if video_path is None:
            return VisualAnalysis(
                gestures_detected=["analysis_unavailable"],
                confidence_estimate="low",
                engagement_level="low",
                notes=["No video provided"],
            )
        return self._visual.analyze(video_path)

    def _cleanup_file(self, file_path: Optional[Path]) -> None:
        if file_path is None:
            return
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
