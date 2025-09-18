from pathlib import Path

from app.services.question_bank import QuestionRepository
from app.services.vector_store import InMemoryVectorStore


def test_vector_store_prioritises_relevant_question():
    repository = QuestionRepository(Path("app/data/linear_regression_questions.json"))
    store = InMemoryVectorStore(repository.load())

    contexts = store.retrieve("least squares residuals normal equations", top_k=1)

    assert contexts
    assert contexts[0].id == "q1"
