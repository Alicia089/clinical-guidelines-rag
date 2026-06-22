from .schemas import RAGResponse, Source

SYSTEM = (
    "You are a clinical guidelines assistant. Answer questions using only the provided "
    "guideline excerpts. Always cite the source document and page number. If the answer "
    "is not in the provided excerpts, say so explicitly."
)


def generate_answer(question: str, sources: list[Source]) -> RAGResponse:
    """
    Generate a grounded answer from retrieved guideline excerpts.

    To wire in your preferred model, replace this function body with:

        context = "\n\n".join(
            f"[{s.document}, page {s.page}]\n{s.excerpt}" for s in sources
        )
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Guideline excerpts:\n\n{context}\n\nQuestion: {question}"},
        ]
        response = <your_client>.chat(model=<your_model>, messages=messages)
        answer = response.text  # adjust to your SDK's response shape

        return RAGResponse(answer=answer, sources=sources, query=question)
    """
    raise NotImplementedError("Generation backend not configured")
