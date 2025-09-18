"""Speech-to-text services."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional


class TranscriptionService:
    """Abstract transcription interface."""

    async def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError


class WhisperTranscriptionService(TranscriptionService):
    """Placeholder for an actual Whisper-based transcription pipeline."""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name

    async def transcribe(self, audio_path: Path) -> str:  # pragma: no cover - illustrative placeholder
        raise NotImplementedError(
            "Integrate OpenAI Whisper or another STT solution to convert audio to text."
        )


class FallbackTranscriptionService(TranscriptionService):
    """Simple transcription implementation used for testing and development."""

    def __init__(self, default_response: Optional[str] = None):
        self._default_response = default_response or ""

    async def transcribe(self, audio_path: Path) -> str:
        await asyncio.sleep(0)
        if audio_path.exists():
            return audio_path.read_text().strip()
        return self._default_response
