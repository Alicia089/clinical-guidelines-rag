# Clinical Guidelines RAG

A retrieval-augmented generation system for querying public clinical guidelines.
POST a clinical question, receive a grounded answer with source citations from
CDC, WHO, and NIH documents.

## How it works

1. **Ingest** — chunk public guideline PDFs, embed with `sentence-transformers/all-MiniLM-L6-v2`, index with FAISS
2. **Retrieve** — embed the question, find the top-k most relevant chunks
3. **Generate** — pass retrieved chunks with a citation-enforcing prompt

## Stack

Python 3.11, FastAPI, FAISS, sentence-transformers, Docker, GitHub Actions

## Run locally

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload

# Build the FAISS index (add PDFs to data/guidelines/ first)
curl -X POST http://localhost:8000/ingest

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recommended blood pressure target for adults?"}'
```

## Data sources

Place clinical guideline PDFs in `data/guidelines/`. The `indexes/` directory is
gitignored — FAISS indexes are built at runtime via the `/ingest` endpoint.
Tested with publicly available CDC, WHO, and NIH hypertension and diabetes guidelines.

## Design decisions

FAISS over a hosted vector DB keeps this self-contained and free to run locally.
`all-MiniLM-L6-v2` is fast enough for real-time retrieval at this scale.
