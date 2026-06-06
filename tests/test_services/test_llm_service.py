from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.exceptions import ExternalServiceError
from services.llm_service import (
    EmbeddingService,
    _is_openrouter_api_key,
    _normalise_openrouter_embedding_model,
    _normalise_openrouter_model,
)


def test_detects_openrouter_key() -> None:
    assert _is_openrouter_api_key("sk-or-v1-test")
    assert not _is_openrouter_api_key("sk-proj-test")


def test_normalises_openrouter_chat_models() -> None:
    assert _normalise_openrouter_model("gpt-4o-mini") == "openai/gpt-4o-mini"
    assert _normalise_openrouter_model("claude-3-5-haiku") == "anthropic/claude-3-5-haiku"
    assert _normalise_openrouter_model("google/gemini-flash-1.5") == "google/gemini-flash-1.5"


def test_normalises_openrouter_embedding_models() -> None:
    assert _normalise_openrouter_embedding_model("text-embedding-3-large") == "openai/text-embedding-3-large"
    assert _normalise_openrouter_embedding_model("openai/text-embedding-3-small") == "openai/text-embedding-3-small"


@pytest.mark.asyncio
async def test_embedding_service_rejects_empty_vector_response() -> None:
    service = EmbeddingService.__new__(EmbeddingService)
    service._api_key = "test-key"
    service._model_en = "text-embedding-3-large"
    service._dims = 1536
    service._client = object()
    service._provider_name = "openai"

    async def fake_create_embeddings(request: dict) -> SimpleNamespace:  # noqa: ARG001
        return SimpleNamespace(data=[], usage=None)

    service._create_embeddings = fake_create_embeddings

    with pytest.raises(ExternalServiceError, match="returned no vectors"):
        await service.embed_many(["ກົດໝາຍລາວ"], multilingual=True)
