from fastapi import FastAPI, HTTPException

from .generator import generate_answer
from .retriever import retrieve
from .schemas import ClinicalQuery, RAGResponse

app = FastAPI(title="Clinical Guidelines RAG", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest():
    from .ingest import ingest_pdfs
    try:
        _, chunks = ingest_pdfs()
        return {"message": f"Indexed {len(chunks)} chunks"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query", response_model=RAGResponse)
def query(request: ClinicalQuery):
    try:
        sources = retrieve(request.question, top_k=request.top_k)
        return generate_answer(request.question, sources)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Index not built. POST to /ingest first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
