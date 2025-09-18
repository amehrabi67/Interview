"""Audio/video recording helpers."""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Iterable, Optional

from app.models.schema import KeyFrame, MediaCapture


class AudioVideoRecorder:
    """Abstract base class for multimodal capture."""

    def record(self, question_prompt: str, duration_seconds: Optional[int] = None) -> MediaCapture:
        raise NotImplementedError


class MockAudioVideoRecorder(AudioVideoRecorder):
    """Recorder used during development to avoid hardware dependencies."""

    def __init__(self, transcript: str, gestures: Optional[Iterable[str]] = None):
        self._transcript = transcript
        self._gestures = list(gestures or [])
        self._output_dir = Path(tempfile.mkdtemp(prefix="mock_recorder_"))

    def _write_placeholder_file(self, suffix: str) -> Path:
        filename = f"{uuid.uuid4()}.{suffix}"
        path = self._output_dir / filename
        path.write_text(self._transcript)
        return path

    def record(self, question_prompt: str, duration_seconds: Optional[int] = None) -> MediaCapture:
        audio_path = self._write_placeholder_file("txt")
        video_path = self._write_placeholder_file("video.txt")
        key_frames = [
            KeyFrame(timestamp=index * 1.5, precomputed_gestures=[gesture])
            for index, gesture in enumerate(self._gestures)
        ]
        return MediaCapture(
            audio_path=audio_path,
            video_path=video_path,
            transcript=self._transcript,
            key_frames=key_frames,
        )
