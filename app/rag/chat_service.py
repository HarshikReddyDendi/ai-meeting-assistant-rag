"""In-memory, meeting-scoped conversation service."""

from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import Embeddings

from app.rag.qa_chain import ConversationTurn, answer_question


class MeetingChatService:
    """Manage a conversation whose answers are grounded in one meeting."""

    def __init__(
        self,
        meeting_id: str,
        *,
        top_k: int = 4,
        model: str | None = None,
        embeddings: Embeddings | None = None,
        persist_directory: str | Path | None = None,
    ) -> None:
        self.meeting_id = meeting_id.strip()
        if not self.meeting_id:
            raise ValueError("meeting_id must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        self.top_k = top_k
        self.model = model
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self._history: list[ConversationTurn] = []

    @property
    def history(self) -> tuple[ConversationTurn, ...]:
        """Return a read-only snapshot of this meeting's conversation."""
        return tuple(self._history)

    def ask(self, question: str) -> ConversationTurn:
        """Ask a transcript-grounded question and append the completed turn."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty.")

        result = answer_question(
            normalized_question,
            meeting_id=self.meeting_id,
            top_k=self.top_k,
            model=self.model,
            embeddings=self.embeddings,
            persist_directory=self.persist_directory,
            history=self._history,
        )
        turn = ConversationTurn(
            question=normalized_question,
            answer=result.answer,
            sources=result.sources,
        )
        self._history.append(turn)
        return turn

    def clear_history(self) -> None:
        """Clear chat messages without changing the persistent meeting index."""
        self._history.clear()
