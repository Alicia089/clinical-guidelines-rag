import json
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
INDEX_DIR = Path("indexes")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _chunk_text(text: str, source: str, page: int) -> list[dict]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_words = words[i:i + CHUNK_SIZE]
        if chunk_words:
            chunks.append({"text": " ".join(chunk_words), "source": source, "page": page})
    return chunks


def ingest_pdfs(data_dir: str = "data/guidelines") -> tuple[faiss.IndexFlatL2, list[dict]]:
    model = SentenceTransformer(MODEL_NAME)
    all_chunks: list[dict] = []

    for pdf_path in Path(data_dir).glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            all_chunks.extend(_chunk_text(text, pdf_path.name, page_num + 1))

    if not all_chunks:
        raise ValueError(f"No PDFs found in {data_dir}")

    texts = [c["text"] for c in all_chunks]
    embeddings = np.array(model.encode(texts, show_progress_bar=True)).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "guidelines.faiss"))
    with open(INDEX_DIR / "chunks.json", "w") as f:
        json.dump(all_chunks, f)

    return index, all_chunks
