from typing import List

from pydantic import BaseModel


class ClinicalQuery(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    document: str
    page: int
    excerpt: str


class RAGResponse(BaseModel):
    answer: str
    sources: List[Source]
    query: str
