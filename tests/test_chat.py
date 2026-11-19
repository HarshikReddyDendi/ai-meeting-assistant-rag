import pytest

from app.rag import chat_service
from app.rag.qa_chain import ConversationTurn, GroundedAnswer


def test_chat_service_passes_prior_turns_to_followup_and_can_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_histories: list[tuple[ConversationTurn, ...]] = []

    def fake_answer(
        question: str,
        *,
        meeting_id: str,
        top_k: int,
        model: str | None,
        embeddings: object,
        persist_directory: object,
        history: tuple[ConversationTurn, ...] | list[ConversationTurn],
    ) -> GroundedAnswer:
        assert meeting_id == "planning"
        assert top_k == 4
        assert model is None
        assert embeddings is None
        assert persist_directory is None
        seen_histories.append(tuple(history))
        return GroundedAnswer(
            answer=f"Answer to: {question}",
            sources=("00:00:01.000 - 00:00:03.000",),
        )

    monkeypatch.setattr(chat_service, "answer_question", fake_answer)
    chat = chat_service.MeetingChatService(" planning ")

    first = chat.ask("What was decided?")
    follow_up = chat.ask("Who owns it?")

    assert first.question == "What was decided?"
    assert follow_up.sources == ("00:00:01.000 - 00:00:03.000",)
    assert len(seen_histories[0]) == 0
    assert seen_histories[1] == (first,)
    assert len(chat.history) == 2

    chat.clear_history()

    assert chat.history == ()


def test_chat_service_rejects_empty_meeting_and_question() -> None:
    with pytest.raises(ValueError, match="meeting_id"):
        chat_service.MeetingChatService("  ")

    chat = chat_service.MeetingChatService("planning")
    with pytest.raises(ValueError, match="question"):
        chat.ask("  ")
