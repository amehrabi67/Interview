from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional

from app.config import Settings
from app.db.vector_store import QuestionVectorStore
from app.models.domain import ContentEvaluation, QuestionEntry
from app.services.llm import HeuristicBackend, LLMBackend


class RAGEvaluator:
    """Combines retrieval and generation to grade a spoken answer."""

    def __init__(
        self,
        store: QuestionVectorStore,
        settings: Settings,
        llm_backend: Optional[LLMBackend] = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._llm = llm_backend or HeuristicBackend()

    def evaluate(self, question: QuestionEntry, transcript: str) -> ContentEvaluation:
        contexts = self._store.retrieve_context(transcript or question.answer, top_k=3)
        prompt = self._build_prompt(question, transcript, contexts)
        raw_response = self._llm.run(prompt)
        parsed = self._parse_response(raw_response)
        if not parsed:
            parsed = self._heuristic_evaluation(question, transcript, contexts)
        return ContentEvaluation(**parsed)

    def _build_prompt(self, question: QuestionEntry, transcript: str, contexts: List[dict]) -> str:
        payload = {
            "question": question.question,
            "transcript": transcript,
            "context": contexts,
            "instructions": self._settings.evaluation_prompt_template,
        }
        return json.dumps(payload)

    def _parse_response(self, response: str) -> Optional[Dict[str, object]]:
        json_blob = self._extract_json(response)
        if not json_blob:
            return None
        try:
            parsed = json.loads(json_blob)
        except json.JSONDecodeError:
            return None

        expected_keys = {"accuracy_score", "missed_concepts", "errors_made", "correct_points", "confidence"}
        if not expected_keys.issubset(parsed):
            return None
        return parsed

    def _extract_json(self, text: str) -> Optional[str]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return match.group(0) if match else None

    def _heuristic_evaluation(
        self, question: QuestionEntry, transcript: str, contexts: List[dict]
    ) -> Dict[str, object]:  # pragma: no cover - deterministic simple heuristic
        transcript_lower = transcript.lower()
        key_points = self._aggregate_key_points(question, contexts)
        covered = []
        missed = []
        for point in key_points:
            similarity = self._phrase_similarity(point.lower(), transcript_lower)
            if similarity >= 0.55:
                covered.append(point)
            else:
                missed.append(point)

        accuracy = 100 * len(covered) / len(key_points) if key_points else 30.0
        confidence = max(40.0, min(95.0, accuracy - 5.0))

        return {
            "accuracy_score": round(accuracy, 2),
            "missed_concepts": missed,
            "errors_made": [],
            "correct_points": covered,
            "confidence": round(confidence, 2),
        }

    def _aggregate_key_points(self, question: QuestionEntry, contexts: List[dict]) -> List[str]:
        unique = set(question.key_points)
        for ctx in contexts:
            metadata = ctx.get("metadata") or {}
            for point in metadata.get("key_points", []):
                unique.add(point)
        return list(unique)

    def _phrase_similarity(self, phrase: str, transcript: str) -> float:
        if not phrase:
            return 0.0
        overlap = sum(1 for token in phrase.split() if token in transcript)
        return overlap / math.sqrt(len(phrase.split()) or 1)
