import numpy as np

import app.retriever as r
from app.schemas import Source


def _setup(chunks: list[dict], indices: list[int]) -> None:
    from unittest.mock import MagicMock
    r._model = MagicMock()
    r._model.encode.return_value = np.zeros((1, 384), dtype="float32")
    r._index = MagicMock()
    r._index.search.return_value = (None, np.array([indices]))
    r._chunks = chunks


def test_retrieve_returns_sources():
    _setup(
        chunks=[
            {"text": "Hypertension management per WHO guidelines.", "source": "who_htn.pdf", "page": 3},
            {"text": "Blood pressure targets for adults.", "source": "cdc_bp.pdf", "page": 7},
        ],
        indices=[0, 1],
    )
    sources = r.retrieve("What is the BP target?", top_k=2)
    assert len(sources) == 2
    assert all(isinstance(s, Source) for s in sources)


def test_retrieve_source_fields():
    _setup(
        chunks=[{"text": "Hypertension management per WHO guidelines.", "source": "who_htn.pdf", "page": 3}],
        indices=[0],
    )
    source = r.retrieve("hypertension", top_k=1)[0]
    assert source.document == "who_htn.pdf"
    assert source.page == 3
    assert isinstance(source.excerpt, str)


def test_retrieve_excerpt_truncated_at_300():
    _setup(
        chunks=[{"text": "x" * 500, "source": "test.pdf", "page": 1}],
        indices=[0],
    )
    source = r.retrieve("test", top_k=1)[0]
    assert len(source.excerpt) <= 300
