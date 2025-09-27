from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import Depends

from app.config import Settings, get_settings
from app.db.vector_store import QuestionVectorStore
from app.services.llm import LlamaCppBackend
from app.services.pipeline import AssessmentPipeline
from app.services.question_service import QuestionService
from app.services.rag import RAGEvaluator
from app.services.report import ReportBuilder
from redis import Redis
from redis.exceptions import RedisError

from app.services.session import RedisSessionStore, SessionStore
from app.services.transcription import TranscriptionService
from app.services.visual import LlavaVisionLanguageModel, NullVisionLanguageModel, VisualAnalyzer


@lru_cache(maxsize=1)
def get_session_store_singleton() -> SessionStore:
    settings = get_settings()
    if settings.use_celery or settings.session_store_url:
        redis_url = settings.session_store_url or settings.celery_result_backend or settings.celery_broker_url
        try:
            client = Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            return RedisSessionStore(client)
        except (RedisError, OSError) as exc:
            print(f"Failed to initialise Redis session store at {redis_url}: {exc}. Falling back to in-memory store.")
    return SessionStore()


@lru_cache(maxsize=1)
def get_vector_store_singleton() -> QuestionVectorStore:
    settings = get_settings()
    return QuestionVectorStore(
        dataset_path=settings.dataset_path,
        persist_directory=settings.vector_store_path,
        collection_name=settings.collection_name,
        embedding_model=settings.embedding_model,
    )


@lru_cache(maxsize=1)
def get_transcription_singleton() -> TranscriptionService:
    settings = get_settings()
    return TranscriptionService(model_name=settings.transcription_model, language=settings.transcription_language)


@lru_cache(maxsize=1)
def get_llm_backend_singleton() -> Optional[LlamaCppBackend]:
    settings = get_settings()
    if not settings.llm_model_path:
        return None
    try:
        return LlamaCppBackend(
            model_path=settings.llm_model_path,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    except Exception as exc:  # pragma: no cover - optional dependency guard
        print(f"Failed to initialise LLM backend: {exc}")
        return None


@lru_cache(maxsize=1)
def get_visual_backend_singleton():
    settings = get_settings()
    if not settings.enable_vlm or not settings.vlm_model_name:
        return NullVisionLanguageModel()
    try:
        return LlavaVisionLanguageModel(settings.vlm_model_name, device=settings.vlm_device)
    except Exception as exc:  # pragma: no cover - optional dependency guard
        print(f"Failed to initialise VLM backend: {exc}")
        return NullVisionLanguageModel()


@lru_cache(maxsize=1)
def get_visual_analyzer_singleton() -> VisualAnalyzer:
    settings = get_settings()
    return VisualAnalyzer(settings=settings, vlm=get_visual_backend_singleton())


@lru_cache(maxsize=1)
def get_rag_singleton() -> RAGEvaluator:
    settings = get_settings()
    return RAGEvaluator(
        store=get_vector_store_singleton(),
        settings=settings,
        llm_backend=get_llm_backend_singleton(),
    )


@lru_cache(maxsize=1)
def get_report_builder_singleton() -> ReportBuilder:
    settings = get_settings()
    return ReportBuilder(settings)


@lru_cache(maxsize=1)
def get_pipeline_singleton() -> AssessmentPipeline:
    return AssessmentPipeline(
        sessions=get_session_store_singleton(),
        transcription=get_transcription_singleton(),
        rag=get_rag_singleton(),
        visual=get_visual_analyzer_singleton(),
        report_builder=get_report_builder_singleton(),
    )


@lru_cache(maxsize=1)
def get_question_service_singleton() -> QuestionService:
    settings = get_settings()
    return QuestionService(get_vector_store_singleton(), get_session_store_singleton(), settings)


def get_session_store(settings: Settings = Depends(get_settings)) -> SessionStore:  # pragma: no cover - FastAPI DI hook
    return get_session_store_singleton()


def get_question_service(settings: Settings = Depends(get_settings)) -> QuestionService:  # pragma: no cover
    return get_question_service_singleton()


def get_pipeline(settings: Settings = Depends(get_settings)) -> AssessmentPipeline:  # pragma: no cover
    return get_pipeline_singleton()
