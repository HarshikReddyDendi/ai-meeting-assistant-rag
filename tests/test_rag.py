from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag import qa_chain
from app.rag.document_loader import load_transcript
from app.rag.retriever import retrieve_relevant_chunks
from app.rag.text_splitter import split_transcript
from app.rag.vector_store import index_documents
from app.rag.qa_chain import ConversationTurn
from app.transcription.models import (
    TranscriptSegment,
    TranscriptionResult,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("launch") + 1),
            float(lowered.count("deadline") + 1),
            float(len(text) % 17 + 1),
        ]


def test_loader_preserves_available_timestamp_and_speaker() -> None:
    transcription = TranscriptionResult(
        text="Mira will send the release notes.",
        language="en",
        segments=(
            TranscriptSegment(
                start=1.25,
                end=3.5,
                text="Mira will send the release notes.",
                speaker="Speaker 1",
            ),
        ),
    )

    documents = load_transcript(transcription, meeting_id="meeting-1")

    assert documents[0].page_content.startswith("[00:00:01.250]")
    assert documents[0].metadata["speaker"] == "Speaker 1"
    assert documents[0].metadata["meeting_id"] == "meeting-1"


def test_splitter_adds_chunk_indexes_and_citations() -> None:
    document = Document(
        page_content="A launch discussion. " * 14,
        metadata={"meeting_id": "meeting-1", "source": "transcript"},
    )

    chunks = split_transcript([document], chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert chunks[0].metadata["source_ref"] == "chunk-1"


def test_chroma_indexing_persists_and_scopes_retrieval_per_meeting(
    tmp_path: Path,
) -> None:
    embeddings = FakeEmbeddings()
    meeting_one = split_transcript(
        load_transcript("The launch deadline is Friday.", meeting_id="meeting-1")
    )
    meeting_two = split_transcript(
        load_transcript("The launch deadline is Monday.", meeting_id="meeting-2")
    )

    index_documents(
        meeting_one,
        meeting_id="meeting-1",
        embeddings=embeddings,
        persist_directory=tmp_path,
    )
    index_documents(
        meeting_two,
        meeting_id="meeting-2",
        embeddings=embeddings,
        persist_directory=tmp_path,
    )

    results = retrieve_relevant_chunks(
        "What is the launch deadline?",
        meeting_id="meeting-1",
        embeddings=embeddings,
        persist_directory=tmp_path,
    )

    assert results
    assert all(doc.metadata["meeting_id"] == "meeting-1" for doc in results)
    assert all("Friday" in doc.page_content for doc in results)


def test_reindexing_removes_stale_chunks(
    tmp_path: Path,
) -> None:
    embeddings = FakeEmbeddings()
    old_chunks = split_transcript(
        load_transcript(
            "Old content. " * 20,
            meeting_id="meeting-update",
        ),
        chunk_size=30,
        chunk_overlap=0,
    )
    index_documents(
        old_chunks,
        meeting_id="meeting-update",
        embeddings=embeddings,
        persist_directory=tmp_path,
    )
    replacement = split_transcript(
        load_transcript("Replacement content.", meeting_id="meeting-update")
    )

    store = index_documents(
        replacement,
        meeting_id="meeting-update",
        embeddings=embeddings,
        persist_directory=tmp_path,
    )
    stored = store.get(include=["documents", "metadatas"])

    assert len(stored["ids"]) == 1
    assert stored["documents"] == ["Replacement content."]


def test_question_without_retrieved_chunks_abstains_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def retrieve_none(
        question: str,
        *,
        meeting_id: str,
        top_k: int = 4,
        embeddings: Embeddings | None = None,
        persist_directory: str | Path | None = None,
    ) -> list[Document]:
        assert question
        assert meeting_id == "meeting-1"
        assert top_k > 0
        assert embeddings is None
        assert persist_directory is None
        return []

    monkeypatch.setattr(qa_chain, "retrieve_relevant_chunks", retrieve_none)
    monkeypatch.setattr(
        qa_chain,
        "invoke_llm",
        lambda prompt: pytest.fail(
            "LLM must not be called without evidence" if prompt else "Empty prompt"
        ),
    )

    result = qa_chain.answer_question("What was decided?", meeting_id="meeting-1")

    assert result.answer == qa_chain.ANSWER_NOT_FOUND
    assert result.sources == ()


def test_question_answer_includes_retrieved_source_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def retrieve_one(
        question: str,
        *,
        meeting_id: str,
        top_k: int = 4,
        embeddings: Embeddings | None = None,
        persist_directory: str | Path | None = None,
    ) -> list[Document]:
        assert question
        assert meeting_id == "meeting-1"
        assert top_k > 0
        assert embeddings is None
        assert persist_directory is None
        return [
            Document(
                page_content="The launch deadline is Friday.",
                metadata={"source_ref": "chunk-2"},
            )
        ]

    monkeypatch.setattr(
        qa_chain,
        "retrieve_relevant_chunks",
        retrieve_one,
    )

    def fake_invoke(prompt: str, *, model: str | None = None) -> str:
        assert "[chunk-2]" in prompt
        assert "The launch deadline is Friday." in prompt
        assert model is None
        return "The deadline is Friday [chunk-2]."

    monkeypatch.setattr(qa_chain, "invoke_llm", fake_invoke)

    result = qa_chain.answer_question("When is the deadline?", meeting_id="meeting-1")

    assert result.answer == "The deadline is Friday [chunk-2]."
    assert result.sources == ("chunk-2",)


def test_follow_up_questions_use_history_for_retrieval_but_not_as_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_turn = ConversationTurn(
        question="What did they decide about the launch?",
        answer="They chose the Friday launch date.",
        sources=("chunk-1",),
    )

    def retrieve_follow_up(
        query: str,
        *,
        meeting_id: str,
        top_k: int = 4,
        embeddings: Embeddings | None = None,
        persist_directory: str | Path | None = None,
    ) -> list[Document]:
        assert "Friday launch date" in query
        assert "Who is responsible?" in query
        assert meeting_id == "meeting-1"
        assert top_k == 4
        assert embeddings is None
        assert persist_directory is None
        return [
            Document(
                page_content="Mira will coordinate the Friday launch.",
                metadata={"source_ref": "chunk-3"},
            )
        ]

    def fake_invoke(prompt: str, *, model: str | None = None) -> str:
        assert "Conversation history:" in prompt
        assert "prior assistant messages are not evidence" in prompt.lower()
        assert "Mira will coordinate the Friday launch." in prompt
        assert model is None
        return "Mira is responsible [chunk-3]."

    monkeypatch.setattr(qa_chain, "retrieve_relevant_chunks", retrieve_follow_up)
    monkeypatch.setattr(qa_chain, "invoke_llm", fake_invoke)

    result = qa_chain.answer_question(
        "Who is responsible?",
        meeting_id="meeting-1",
        history=(previous_turn,),
    )

    assert result.answer == "Mira is responsible [chunk-3]."
    assert result.sources == ("chunk-3",)


@pytest.mark.parametrize(
    ("text", "expected_error"),
    [("", "question"), ("   ", "question")],
)
def test_retriever_rejects_empty_questions(text: str, expected_error: str) -> None:
    with pytest.raises(ValueError, match=expected_error):
        retrieve_relevant_chunks(text, meeting_id="meeting-1")
