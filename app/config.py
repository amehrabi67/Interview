from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables when available."""

    app_name: str = "Linear Regression Oral Exam Platform"
    dataset_path: Path = Path(__file__).resolve().parent.parent / "data" / "questions.json"
    vector_store_path: Path = Path(__file__).resolve().parent.parent / "storage" / "vector_store"
    collection_name: str = "linear_regression_questions"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    transcription_model: str = "base"
    transcription_language: Optional[str] = "en"
    sample_rate: int = 16_000
    question_duration_seconds: int = 60

    llm_model_path: Optional[str] = None
    llm_temperature: float = 0.2
    llm_max_tokens: int = 768

    evaluation_prompt_template: str = (
        "You are a strict statistics professor assessing an oral answer. "
        "Compare the transcript with the verified context and question. "
        "Return a JSON object with keys accuracy_score (0-100), missed_concepts (list), "
        "errors_made (list), correct_points (list), and confidence (0-100)."
    )

    enable_vlm: bool = False
    vlm_model_name: Optional[str] = None
    vlm_device: str = "cpu"
    keyframe_window_seconds: float = 3.0
    max_keyframes: int = 10
    gesture_confidence_threshold: float = 0.5

    composite_score_content_weight: float = 0.7
    composite_score_delivery_weight: float = 0.3

    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"
    use_celery: bool = False

    model_config = SettingsConfigDict(env_prefix="INTERVIEW_", env_file=".env", env_file_encoding="utf-8")

    def ensure_directories(self) -> None:
        """Make sure directories referenced in settings exist."""

        self.vector_store_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
