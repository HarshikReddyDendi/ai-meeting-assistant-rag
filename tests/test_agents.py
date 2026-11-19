from collections.abc import Callable

import pytest

from app.agents import action_item_extractor, decision_extractor, summarizer
from app.agents._structured_output import StructuredOutputError


def _fixed_response(response: str) -> Callable[[str], str]:
    def invoke(prompt: str) -> str:
        assert prompt.strip()
        return response

    return invoke


def test_summarize_meeting_returns_structured_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        summarizer,
        "invoke_llm",
        _fixed_response(
            '{"executive_summary":"Launch planning","detailed_summary":"Team '
            'reviewed launch readiness.","main_topics":["Launch"],'
            '"unresolved_questions":["Pricing"],"follow_up_requirements":[]}'
        ),
    )

    result = summarizer.summarize_meeting("The team reviewed launch readiness.")

    assert result.executive_summary == "Launch planning"
    assert result.main_topics == ["Launch"]
    assert result.unresolved_questions == ["Pricing"]


def test_extract_decisions_accepts_json_code_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        decision_extractor,
        "invoke_llm",
        _fixed_response('```json\n{"decisions":["Ship on Friday"]}\n```'),
    )

    result = decision_extractor.extract_decisions("We agreed to ship on Friday.")

    assert result.decisions == ["Ship on Friday"]


def test_action_items_default_missing_owner_and_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        action_item_extractor,
        "invoke_llm",
        _fixed_response(
            '{"action_items":[{"task":"Send the agenda","owner":null,'
            '"deadline":""}]}'
        ),
    )

    result = action_item_extractor.extract_action_items("Send the agenda.")

    assert result.action_items[0].owner == "Not specified"
    assert result.action_items[0].deadline == "Not specified"


def test_invalid_structured_llm_response_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(summarizer, "invoke_llm", _fixed_response("not JSON"))

    with pytest.raises(StructuredOutputError, match="MeetingSummary"):
        summarizer.summarize_meeting("A valid transcript.")


@pytest.mark.parametrize(
    "agent_function",
    [
        summarizer.summarize_meeting,
        decision_extractor.extract_decisions,
        action_item_extractor.extract_action_items,
    ],
)
def test_agent_functions_reject_empty_transcript(
    agent_function: Callable[[str], object],
) -> None:
    with pytest.raises(ValueError, match="transcript"):
        agent_function("  ")
