"""Split transcript documents into retrievable chunks."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.document_loader import _format_timestamp


def split_transcript(
    documents: Sequence[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split transcript documents and attach stable chunk citation metadata."""
    if not documents:
        raise ValueError("documents must not be empty.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

    meeting_ids = {document.metadata.get("meeting_id") for document in documents}
    if len(meeting_ids) != 1 or None in meeting_ids:
        raise ValueError("All documents must belong to exactly one meeting.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(list(documents))
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        start = chunk.metadata.get("start")
        end = chunk.metadata.get("end")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            chunk.metadata["source_ref"] = (
                f"{_format_timestamp(float(start))} - {_format_timestamp(float(end))}"
            )
        else:
            chunk.metadata["source_ref"] = f"chunk-{index + 1}"
    return chunks
