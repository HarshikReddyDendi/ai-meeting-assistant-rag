"""Helpers for validating JSON responses returned by an LLM."""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredOutputError(ValueError):
    """Raised when an LLM response does not match the requested schema."""


def parse_model_response(response: str, model: type[ModelT]) -> ModelT:
    """Parse JSON (optionally in a Markdown code fence) into a Pydantic model."""
    content = response.strip()
    fenced_response = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_response:
        content = fenced_response.group(1)

    try:
        payload = json.loads(content)
        return model.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise StructuredOutputError(
            f"The LLM response was not valid {model.__name__} JSON."
        ) from exc
