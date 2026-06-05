from __future__ import annotations

from services.llm_service import (
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
