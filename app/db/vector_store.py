from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import chromadb
from chromadb.utils import embedding_functions

from app.models.domain import QuestionEntry


class QuestionVectorStore:
    """Lightweight wrapper around ChromaDB for managing the question bank."""

    def __init__(
        self,
        dataset_path: Path,
        persist_directory: Path,
        collection_name: str,
        embedding_model: str,
    ) -> None:
        self._dataset_path = dataset_path
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._embedding_model = embedding_model

        self._client = chromadb.PersistentClient(path=str(persist_directory))
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_fn
        )
        self._questions: Dict[str, QuestionEntry] = {}
        self._load_dataset()

    def _load_dataset(self) -> None:
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self._dataset_path}")

        with self._dataset_path.open("r", encoding="utf-8") as fh:
            payload: Sequence[Dict[str, object]] = json.load(fh)

        existing_ids = set(self._collection.get()["ids"] or [])

        for raw_entry in payload:
            entry = QuestionEntry(
                id=str(raw_entry["id"]),
                question=str(raw_entry["question"]),
                answer=str(raw_entry["answer"]),
                citations=list(raw_entry.get("citations", [])),
                key_points=list(raw_entry.get("key_points", [])),
            )
            self._questions[entry.id] = entry

            if entry.id in existing_ids:
                continue

            self._collection.add(
                ids=[entry.id],
                documents=[entry.answer],
                metadatas=[
                    {
                        "question": entry.question,
                        "citations": entry.citations,
                        "key_points": entry.key_points,
                    }
                ],
            )

    def question_ids(self) -> Iterable[str]:
        return self._questions.keys()

    def random_question(self) -> QuestionEntry:
        if not self._questions:
            raise RuntimeError("Question bank is empty")
        return random.choice(list(self._questions.values()))

    def get_question(self, question_id: str) -> QuestionEntry:
        return self._questions[question_id]

    def retrieve_context(self, query: str, top_k: int = 3) -> List[dict]:
        """Retrieve the most relevant knowledge snippets for a given query."""

        results = self._collection.query(
            query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"]
        )
        contexts: List[dict] = []
        for idx, doc_id in enumerate(results.get("ids", [[]])[0]):
            metadata = results.get("metadatas", [[]])[0][idx]
            document = results.get("documents", [[]])[0][idx]
            distance = results.get("distances", [[]])[0][idx]
            contexts.append(
                {
                    "id": doc_id,
                    "context": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )
        return contexts
