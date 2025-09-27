from __future__ import annotations

from pathlib import Path
from typing import Optional

from celery import Celery

from app.config import get_settings
from app.dependencies import get_pipeline, get_session_store_singleton

settings = get_settings()

celery_app = Celery(
    "assessment",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)


@celery_app.task(name="assessment.process_submission")
def process_submission_task(
    session_id: str,
    audio_path: Optional[str] = None,
    video_path: Optional[str] = None,
    transcript: Optional[str] = None,
) -> bool:
    # Ensure the worker initialises the same session backend as the API process.
    get_session_store_singleton()
    pipeline = get_pipeline()
    pipeline.process(
        session_id=session_id,
        audio_path=Path(audio_path) if audio_path else None,
        video_path=Path(video_path) if video_path else None,
        transcript=transcript,
    )
    return True
