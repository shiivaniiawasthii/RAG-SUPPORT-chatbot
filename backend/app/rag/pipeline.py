import uuid

from app.config import settings
from app.llm.base import LLMProvider
from app.rag.chunking import chunk_text
from app.rag.extraction import extract_text
from app.rag.vector_store import VectorStore


def ingest_document(filename: str, content: bytes, provider: LLMProvider) -> tuple[str, int]:
    text = extract_text(filename, content)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("No extractable text found in document")

    embeddings = provider.embed(chunks)

    document_id = str(uuid.uuid4())
    store = VectorStore(settings.vector_store_dir, document_id)
    store.save(embeddings, chunks, filename)
    return document_id, len(chunks)


def answer_question(
    document_id: str, question: str, provider: LLMProvider
) -> tuple[str, list[tuple[str, float]]]:
    store = VectorStore(settings.vector_store_dir, document_id)
    query_embedding = provider.embed_query(question)
    results = store.search(query_embedding, settings.top_k)

    context = "\n\n---\n\n".join(chunk for chunk, _ in results)
    answer = provider.generate(question, context)
    return answer, results
