from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.models.domain import CompositeReport
from app.schemas import (
    ContentEvaluationModel,
    ExamReportModel,
    ExamStatusResponse,
    QuestionResponse,
    SubmissionResponse,
    TranscriptOnlyRequest,
    VisualAnalysisModel,
)
from app.dependencies import get_pipeline, get_question_service, get_session_store
from app.services.pipeline import AssessmentPipeline
from app.services.question_service import QuestionService
from app.services.session import SessionStore

router = APIRouter()


def _persist_upload(upload: UploadFile, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload.file, tmp)
        tmp_path = Path(tmp.name)
    upload.file.close()
    return tmp_path


def _serialize_report(report: CompositeReport) -> ExamReportModel:
    return ExamReportModel(
        question_asked=report.question_asked,
        student_answer_transcript=report.student_answer_transcript,
        content_analysis=ContentEvaluationModel(
            accuracy_score=report.content_analysis.accuracy_score,
            missed_concepts=report.content_analysis.missed_concepts,
            errors_made=report.content_analysis.errors_made,
            correct_points=report.content_analysis.correct_points,
            confidence=report.content_analysis.confidence,
        ),
        delivery_analysis=VisualAnalysisModel(
            gestures_detected=report.delivery_analysis.gestures_detected,
            confidence_estimate=report.delivery_analysis.confidence_estimate,
            engagement_level=report.delivery_analysis.engagement_level,
            notes=report.delivery_analysis.notes,
        ),
        composite_score=report.composite_score,
    )


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/exam/start", response_model=QuestionResponse)
def start_exam(question_service: QuestionService = Depends(get_question_service)) -> QuestionResponse:
    session = question_service.start_new_session()
    return QuestionResponse(
        session_id=session.session_id,
        question=session.question.question,
        max_response_duration=question_service.answer_duration_seconds,
    )


@router.post("/exam/{session_id}/submit", response_model=SubmissionResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_response(
    session_id: str,
    background_tasks: BackgroundTasks,
    pipeline: AssessmentPipeline = Depends(get_pipeline),
    session_store: SessionStore = Depends(get_session_store),
    settings: Settings = Depends(get_settings),
    audio: Optional[UploadFile] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
    transcript: Optional[str] = Form(default=None),
) -> SubmissionResponse:
    if session_store.maybe_get(session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown session_id")

    audio_path: Optional[Path] = None
    video_path: Optional[Path] = None
    if audio is not None:
        audio_path = _persist_upload(audio, suffix=".wav")
    if video is not None:
        video_path = _persist_upload(video, suffix=".mp4")

    if settings.use_celery:
        from celery_app import process_submission_task

        process_submission_task.delay(
            session_id,
            str(audio_path) if audio_path else None,
            str(video_path) if video_path else None,
            transcript,
        )
    else:
        background_tasks.add_task(pipeline.process, session_id, audio_path, video_path, transcript)
    return SubmissionResponse(session_id=session_id, status="processing")


@router.post("/exam/submit-transcript", response_model=SubmissionResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_transcript(
    payload: TranscriptOnlyRequest,
    background_tasks: BackgroundTasks,
    pipeline: AssessmentPipeline = Depends(get_pipeline),
    session_store: SessionStore = Depends(get_session_store),
) -> SubmissionResponse:
    if session_store.maybe_get(payload.session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown session_id")

    background_tasks.add_task(pipeline.process, payload.session_id, None, None, payload.transcript)
    return SubmissionResponse(session_id=payload.session_id, status="processing")


@router.get("/exam/{session_id}", response_model=ExamStatusResponse)
def get_status(
    session_id: str,
    session_store: SessionStore = Depends(get_session_store),
) -> ExamStatusResponse:
    try:
        session = session_store.get(session_id)
    except KeyError as exc:  # pragma: no cover - simple pass-through
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown session_id") from exc

    report: Optional[CompositeReport] = session.report
    serialized = _serialize_report(report) if report else None
    return ExamStatusResponse(session_id=session_id, status=session.status, report=serialized)
