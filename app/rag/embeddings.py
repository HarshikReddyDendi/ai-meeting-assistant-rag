"""Hugging Face embedding model factory."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app import config


@lru_cache(maxsize=2)
def _load_embeddings(model_name: str) -> Embeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embeddings(model_name: str | None = None) -> Embeddings:
    """Return a cached Hugging Face sentence-transformer embedding model."""
    selected_model = (model_name or config.EMBEDDING_MODEL).strip()
    if not selected_model:
        raise ValueError("model_name must not be empty.")
    return _load_embeddings(selected_model)
