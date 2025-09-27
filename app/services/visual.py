from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Set

import numpy as np

from app.config import Settings
from app.models.domain import VisualAnalysis

try:  # pragma: no cover - heavy optional dependency
    import cv2
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("opencv-python is required for video analysis") from exc


class BaseVisionLanguageModel:
    """Base interface for vision-language reasoning."""

    def describe(self, image: np.ndarray, prompt: str) -> str:
        raise NotImplementedError


class NullVisionLanguageModel(BaseVisionLanguageModel):
    def describe(self, image: np.ndarray, prompt: str) -> str:  # pragma: no cover - simple passthrough
        return ""


class LlavaVisionLanguageModel(BaseVisionLanguageModel):
    """Wrapper around an open VLM such as LLaVA."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install transformers[torch] and pillow to use a VLM backend") from exc

        self._processor = AutoProcessor.from_pretrained(model_name)
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        self._model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self._device = self._model.device
        self._Image = Image

    def describe(self, image: np.ndarray, prompt: str) -> str:
        pil_image = self._Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self._processor(images=pil_image, text=prompt, return_tensors="pt").to(self._device)
        output = self._model.generate(**inputs, max_new_tokens=128)
        text = self._processor.batch_decode(output, skip_special_tokens=True)[0]
        return text.strip()


class VisualAnalyzer:
    """Extracts gestures and higher-level descriptors from a video segment."""

    def __init__(
        self,
        settings: Settings,
        vlm: Optional[BaseVisionLanguageModel] = None,
    ) -> None:
        self._settings = settings
        self._vlm = vlm or NullVisionLanguageModel()
        self._holistic = self._create_holistic()

    def _create_holistic(self):  # pragma: no cover - optional dependency initialisation
        try:
            import mediapipe as mp
        except ImportError:
            return None
        return mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1)

    def analyze(self, video_path: Path) -> VisualAnalysis:
        frames = self._extract_key_frames(video_path)
        if not frames:
            return VisualAnalysis(
                gestures_detected=["analysis_unavailable"],
                confidence_estimate="low",
                engagement_level="low",
                notes=["Video contained no readable frames"],
            )
        gestures: Set[str] = set()
        notes: List[str] = []
        for frame in frames:
            gestures.update(self._detect_gestures(frame))
            description = self._vlm.describe(frame, self._vlm_prompt())
            if description:
                notes.append(description)

        engagement = self._derive_engagement(len(frames), len(gestures))
        confidence = self._derive_confidence(len(gestures), notes)

        return VisualAnalysis(
            gestures_detected=sorted(gestures),
            confidence_estimate=confidence,
            engagement_level=engagement,
            notes=notes,
        )

    def _extract_key_frames(self, video_path: Path) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():  # pragma: no cover - runtime guard
            raise RuntimeError(f"Unable to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(int(fps * self._settings.keyframe_window_seconds), 1)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: List[np.ndarray] = []

        for frame_idx in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                continue
            frames.append(frame)
            if len(frames) >= self._settings.max_keyframes:
                break

        cap.release()
        return frames

    def _detect_gestures(self, frame: np.ndarray) -> Set[str]:
        results = self._run_holistic(frame)
        if results is None:
            return {"analysis_unavailable"}

        gestures: Set[str] = set()
        mp = self._mediapipe_module()
        if results.left_hand_landmarks:
            gestures.update(self._classify_hand(results.left_hand_landmarks, results.pose_landmarks, mp))
        if results.right_hand_landmarks:
            gestures.update(self._classify_hand(results.right_hand_landmarks, results.pose_landmarks, mp))
        if not gestures:
            gestures.add("low_gesture_activity")
        return gestures

    def _run_holistic(self, frame: np.ndarray):
        if self._holistic is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._holistic.process(rgb)

    def _mediapipe_module(self):  # pragma: no cover - helper for optional dependency
        import mediapipe as mp

        return mp

    def _classify_hand(self, hand_landmarks, pose_landmarks, mp) -> Set[str]:  # pragma: no cover - heavy
        gestures: Set[str] = set()
        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
        middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

        extended_fingers = sum(
            tip.y < index_mcp.y for tip in (index_tip, middle_tip, ring_tip, pinky_tip)
        )
        if index_tip.y < index_mcp.y and middle_tip.y > index_mcp.y:
            gestures.add("pointing")
        if extended_fingers >= 3:
            gestures.add("counting")

        if pose_landmarks:
            nose = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
            distance = math.sqrt(
                (index_tip.x - nose.x) ** 2 + (index_tip.y - nose.y) ** 2 + (index_tip.z - nose.z) ** 2
            )
            if distance < 0.1:
                gestures.add("thinking")

        if not gestures:
            gestures.add("expressive_hand_gesture")
        return gestures

    def _derive_engagement(self, frame_count: int, gesture_count: int) -> str:
        if gesture_count == 0 or frame_count == 0:
            return "low"
        ratio = gesture_count / max(frame_count, 1)
        if ratio > 0.15:
            return "high"
        if ratio > 0.05:
            return "medium"
        return "low"

    def _derive_confidence(self, gesture_count: int, notes: List[str]) -> str:
        if gesture_count >= 3 or any("confident" in note.lower() for note in notes):
            return "high"
        if gesture_count == 0:
            return "low"
        return "medium"

    def _vlm_prompt(self) -> str:
        return (
            "Analyze this video frame of a student. Describe their non-verbal communication. "
            "Do they appear confident, hesitant, or thoughtful? Are they using descriptive hand gestures "
            "relevant to explaining a technical concept? Note any significant facial expressions or posture."
        )
