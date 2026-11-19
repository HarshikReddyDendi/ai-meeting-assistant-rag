"""Meeting-scoped semantic retrieval."""

from __future__ import annotations
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.vector_store import get_vector_store


def retrieve_relevant_chunks(
    question: str,
    *,
    meeting_id: str,
    top_k: int = 4,
    embeddings: Embeddings | None = None,
    persist_directory: str | Path | None = None,
) -> list[Document]:
    """Retrieve semantically relevant chunks from only the requested meeting."""
    if not question.strip():
        raise ValueError("question must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    store = get_vector_store(
        meeting_id,
        embeddings=embeddings,
        persist_directory=persist_directory,
    )
    if not store.get(include=["metadatas"])["ids"]:
        return []
    return store.similarity_search(
        question,
        k=top_k,
        filter={"meeting_id": meeting_id.strip()},
    )
