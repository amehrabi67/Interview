from __future__ import annotations

import json
import os
from typing import Protocol


class LLMBackend(Protocol):
    """Protocol describing a minimal interface for text generation."""

    def run(self, prompt: str) -> str:
        ...


class LlamaCppBackend:
    """LLM backend powered by llama.cpp compatible models."""

    def __init__(self, model_path: str, temperature: float = 0.1, max_tokens: int = 768) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "llama-cpp-python is not installed. Install it to enable local LLM inference."
            ) from exc

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LLM model not found at {model_path}")

        self._llama = Llama(
            model_path=model_path,
            n_ctx=4096,
            temperature=temperature,
        )
        self._max_tokens = max_tokens
        self._temperature = temperature

    def run(self, prompt: str) -> str:
        response = self._llama.create_chat_completion(
            messages=[
                {"role": "system", "content": "You produce strict, JSON-formatted evaluations."},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response["choices"][0]["message"]["content"]


class HeuristicBackend:
    """Fallback backend that produces deterministic JSON responses."""

    def run(self, prompt: str) -> str:  # pragma: no cover - simple deterministic behaviour
        try:
            payload = json.loads(prompt)
        except json.JSONDecodeError:
            return json.dumps(
                {
                    "accuracy_score": 50.0,
                    "missed_concepts": ["Unable to parse prompt"],
                    "errors_made": [],
                    "correct_points": [],
                    "confidence": 30.0,
                }
            )
        return json.dumps(payload)
