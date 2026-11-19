import pytest
from groq import GroqError
from types import SimpleNamespace

from app import config
from app import llm
from app.llm import LLMConfigurationError, LLMInvocationError, get_llm


def test_get_llm_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_MODEL", "test-model")

    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        get_llm()


def test_get_llm_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(config, "GROQ_MODEL", "")

    with pytest.raises(LLMConfigurationError, match="GROQ_MODEL"):
        get_llm()


def test_get_llm_builds_client_without_calling_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(config, "GROQ_MODEL", "test-model")

    llm = get_llm()

    assert llm.model_name == "test-model"


def test_invoke_llm_surfaces_groq_failures_without_api_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedClient:
        def invoke(self, prompt: str) -> object:
            assert prompt == "Hello"
            raise GroqError("simulated network failure")

    def fake_get_llm(*, model: str | None = None) -> FailedClient:
        assert model is None
        return FailedClient()

    monkeypatch.setattr(llm, "get_llm", fake_get_llm)

    with pytest.raises(LLMInvocationError, match="Groq could not complete") as exc_info:
        llm.invoke_llm("Hello")

    assert isinstance(exc_info.value.__cause__, GroqError)


def test_invoke_llm_rejects_non_text_response_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NonTextClient:
        def invoke(self, prompt: str) -> object:
            assert prompt
            return SimpleNamespace(content=[{"text": "not plain text"}])

    def fake_get_llm(*, model: str | None = None) -> NonTextClient:
        assert model is None
        return NonTextClient()

    monkeypatch.setattr(llm, "get_llm", fake_get_llm)

    with pytest.raises(LLMInvocationError, match="without text content"):
        llm.invoke_llm("Hello")
