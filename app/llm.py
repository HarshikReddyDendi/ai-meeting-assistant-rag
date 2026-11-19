"""Groq LLM factory and invocation helpers."""

from __future__ import annotations

import logging

from groq import GroqError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from app import config

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when required LLM configuration is missing."""


class LLMInvocationError(RuntimeError):
    """Raised when Groq cannot complete an invocation."""


def get_llm(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> BaseChatModel:
    """Build a configured ChatGroq client without making a network request."""
    api_key = config.GROQ_API_KEY.strip()
    selected_model = (model or config.GROQ_MODEL).strip()

    if not api_key:
        raise LLMConfigurationError(
            "GROQ_API_KEY is not configured. Add it to the project .env file."
        )
    if not selected_model:
        raise LLMConfigurationError(
            "GROQ_MODEL is not configured. Add a supported Groq model to .env."
        )
    if not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2.")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")

    return ChatGroq(
        model=selected_model,
        temperature=temperature,
        max_retries=max_retries,
        api_key=api_key,
    )


def invoke_llm(prompt: str, *, model: str | None = None) -> str:
    """Invoke Groq for a prompt and return its text response."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty.")

    try:
        response = get_llm(model=model).invoke(prompt)
    except GroqError as exc:
        logger.exception("Groq invocation failed.")
        raise LLMInvocationError("Groq could not complete the request.") from exc

    content = response.content
    if not isinstance(content, str):
        raise LLMInvocationError("Groq returned a response without text content.")
    return content