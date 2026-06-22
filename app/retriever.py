import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .schemas import Source

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = Path("indexes/guidelines.faiss")
CHUNKS_PATH = Path("indexes/chunks.json")

_model: SentenceTransformer | None = None
_index: faiss.IndexFlatL2 | None = None
_chunks: list[dict] | None = None


def _load() -> None:
    global _model, _index, _chunks
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
        _index = faiss.read_index(str(INDEX_PATH))
        with open(CHUNKS_PATH) as f:
            _chunks = json.load(f)


def retrieve(question: str, top_k: int = 5) -> list[Source]:
    _load()
    embedding = np.array(_model.encode([question])).astype("float32")
    _, indices = _index.search(embedding, top_k)
    return [
        Source(
            document=_chunks[idx]["source"],
            page=_chunks[idx]["page"],
            excerpt=_chunks[idx]["text"][:300],
        )
        for idx in indices[0]
        if idx < len(_chunks)
    ]
