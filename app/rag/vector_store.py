"""Persistent, per-meeting Chroma vector-store operations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app import config
from app.rag.embeddings import get_embeddings


def _validate_meeting_id(meeting_id: str) -> str:
    normalized_id = meeting_id.strip()
    if not normalized_id:
        raise ValueError("meeting_id must not be empty.")
    return normalized_id


def _collection_name(meeting_id: str) -> str:
    digest = hashlib.sha256(_validate_meeting_id(meeting_id).encode("utf-8")).hexdigest()
    return f"meeting-{digest[:32]}"


def get_vector_store(
    meeting_id: str,
    *,
    embeddings: Embeddings | None = None,
    persist_directory: str | Path | None = None,
) -> Chroma:
    """Open or create the persistent Chroma collection for one meeting."""
    normalized_id = _validate_meeting_id(meeting_id)
    root = Path(persist_directory) if persist_directory else config.CHROMA_DB_PATH
    root.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=_collection_name(normalized_id),
        embedding_function=(
            embeddings if embeddings is not None else get_embeddings()
        ),
        persist_directory=str(root),
        collection_metadata={"hnsw:space": "cosine"},
    )


def index_documents(
    documents: Sequence[Document],
    *,
    meeting_id: str,
    embeddings: Embeddings | None = None,
    persist_directory: str | Path | None = None,
) -> Chroma:
    """Upsert transcript chunks and remove stale chunks for this meeting."""
    normalized_id = _validate_meeting_id(meeting_id)
    if not documents:
        raise ValueError("documents must not be empty.")

    indexed_documents: list[Document] = []
    document_ids: list[str] = []
    for index, document in enumerate(documents):
        if document.metadata.get("meeting_id") != normalized_id:
            raise ValueError("All indexed documents must belong to meeting_id.")
        document_id = hashlib.sha256(
            f"{normalized_id}:{index}:{document.page_content}".encode("utf-8")
        ).hexdigest()
        document_ids.append(document_id)
        indexed_documents.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    "chunk_index": index,
                    "chunk_id": document_id,
                },
            )
        )

    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Document chunk IDs must be unique.")

    store = get_vector_store(
        normalized_id,
        embeddings=embeddings,
        persist_directory=persist_directory,
    )
    existing_ids = set(store.get(include=["metadatas"])["ids"])
    store.add_documents(indexed_documents, ids=document_ids)
    stale_ids = existing_ids.difference(document_ids)
    if stale_ids:
        store.delete(ids=list(stale_ids))
    return store
