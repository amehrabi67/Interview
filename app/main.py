from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.dependencies import get_pipeline, get_question_service


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(router)

    @app.on_event("startup")
    async def _startup_event() -> None:  # pragma: no cover - simple initialisation hook
        # Warm up key components so the first request has low latency.
        get_question_service()
        get_pipeline()

    return app


app = create_app()
