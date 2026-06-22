from unittest.mock import MagicMock

from app.generator import generate_answer
from app.schemas import RAGResponse, Source


def _sources() -> list[Source]:
    return [
        Source(
            document="who_hypertension.pdf",
            page=3,
            excerpt="Blood pressure target is below 130/80 mmHg per WHO 2023 guidelines.",
        )
    ]


def _mock_client(answer_text: str) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.content = [MagicMock(text=answer_text)]
    client.messages.create.return_value = response
    return client


def test_generate_answer_returns_rag_response():
    result = generate_answer(
        "What is the BP target?",
        _sources(),
        _mock_client("Target is below 130/80 mmHg per WHO guidelines (who_hypertension.pdf, page 3)."),
    )
    assert isinstance(result, RAGResponse)


def test_generate_answer_preserves_query():
    result = generate_answer("What is the BP target?", _sources(), _mock_client("Answer here."))
    assert result.query == "What is the BP target?"


def test_generate_answer_passes_sources_through():
    result = generate_answer("What is the BP target?", _sources(), _mock_client("Answer here."))
    assert len(result.sources) == 1
    assert result.sources[0].document == "who_hypertension.pdf"


def test_generate_answer_returns_model_text():
    result = generate_answer(
        "What is the BP target?",
        _sources(),
        _mock_client("130/80 is the target."),
    )
    assert "130/80" in result.answer
