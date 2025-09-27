from __future__ import annotations

from pathlib import Path
from typing import Optional


class TranscriptionService:
    """Wrapper around Whisper speech-to-text transcription."""

    def __init__(self, model_name: str = "base", language: Optional[str] = "en") -> None:
        self._model_name = model_name
        self._language = language
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import whisper
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "openai-whisper is not installed. Install it or provide a custom transcription backend."
            ) from exc
        self._model = whisper.load_model(self._model_name)
        return self._model

    def transcribe(self, audio_path: Path) -> str:
        model = self._load_model()
        result = model.transcribe(str(audio_path), language=self._language)
        return str(result.get("text", "")).strip()
