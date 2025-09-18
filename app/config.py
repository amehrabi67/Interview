"""Configuration primitives for the interview assessment service."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(slots=True)
class Settings:
    """Runtime settings loaded during application bootstrap."""

    question_dataset: Path = Path(__file__).resolve().parent / "data" / "linear_regression_questions.json"
    recording_duration_seconds: int = 60
    rag_top_k: int = 3
    content_weight: float = 0.8
    delivery_weight: float = 0.2
    engagement_scale: List[str] = field(default_factory=lambda: ["low", "medium", "high"])


settings = Settings()
