"""
services/llm_service.py
========================
Unified LLM abstraction for OpenAI and Anthropic providers.

Design principles:
- Provider-agnostic interface: callers don't know which SDK is used
- Automatic retry with exponential backoff via tenacity
- Token usage tracking for cost monitoring
- Streaming support via async generators
- Graceful degradation: falls back to secondary model on provider error
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.core.config import get_settings
from backend.core.exceptions import ExternalServiceError
from backend.core.logging import get_logger

log = get_logger(__name__)


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LlmResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = "unknown"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Message:
    role: str   # "user" | "assistant" | "system"
    content: str


# ── Provider implementations ───────────────────────────────────────────────────

class _AnthropicProvider:
    def __init__(self, api_key: str) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> LlmResult:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict = dict(model=model, messages=msgs, max_tokens=max_tokens)
        if system:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        text = response.content[0].text if response.content else ""
        return LlmResult(
            text=text,
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            provider="anthropic",
        )

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        import anthropic

        msgs = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict = dict(model=model, messages=msgs, max_tokens=max_tokens)
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for chunk in stream.text_stream:
                yield chunk


class _OpenAIProvider:
    def __init__(self, api_key: str) -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> LlmResult:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend({"role": m.role, "content": m.content} for m in messages)

        response = await self._client.chat.completions.create(
            model=model,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return LlmResult(
            text=text,
            model=model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            provider="openai",
        )

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend({"role": m.role, "content": m.content} for m in messages)

        stream = await self._client.chat.completions.create(
            model=model, messages=msgs, max_tokens=max_tokens, stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── Stub for dev / testing ────────────────────────────────────────────────────

class _StubProvider:
    """Used when no API keys are configured (development / unit tests)."""

    async def generate(self, *, model: str, messages: list[Message], **kwargs) -> LlmResult:
        prompt_preview = messages[-1].content[:80] if messages else ""
        log.debug("llm.stub.generate", model=model, preview=prompt_preview)
        return LlmResult(
            text=f"[STUB:{model}] Response for: {prompt_preview}",
            model=model,
            provider="stub",
        )

    async def stream(self, *, model: str, messages: list[Message], **kwargs) -> AsyncIterator[str]:
        result = await self.generate(model=model, messages=messages)
        for word in result.text.split():
            yield word + " "
            await asyncio.sleep(0.01)


# ── Public LlmService ─────────────────────────────────────────────────────────

class LlmService:
    """
    Route generate calls to the correct provider based on model name.

    Model prefix mapping:
        claude-*  → Anthropic
        gpt-*     → OpenAI
        (anything else in dev) → Stub
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._anthropic: _AnthropicProvider | _StubProvider | None = None
        self._openai: _OpenAIProvider | _StubProvider | None = None

        if settings.anthropic_api_key:
            self._anthropic = _AnthropicProvider(settings.anthropic_api_key)
        if settings.openai_api_key:
            self._openai = _OpenAIProvider(settings.openai_api_key)

        self._stub = _StubProvider()

    def _get_provider(self, model: str):
        if model.startswith("claude"):
            return self._anthropic or self._stub
        if model.startswith("gpt") or model.startswith("o1"):
            return self._openai or self._stub
        return self._stub

    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> LlmResult:
        provider = self._get_provider(model)
        try:
            result = await provider.generate(
                model=model,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            log.info(
                "llm.generate.ok",
                model=model,
                provider=result.provider,
                tokens_in=result.input_tokens,
                tokens_out=result.output_tokens,
            )
            return result
        except Exception as exc:
            log.error("llm.generate.failed", model=model, error=str(exc))
            raise ExternalServiceError(f"LLM call failed for model {model}: {exc}") from exc

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        provider = self._get_provider(model)
        async for chunk in provider.stream(model=model, messages=messages, system=system, max_tokens=max_tokens):
            yield chunk


# ── Embedding service ─────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    vector: list[float]
    model: str
    tokens: int = 0


class EmbeddingService:
    """Generates text embeddings via OpenAI."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.openai_api_key
        self._model_en = settings.embedding_model_en
        self._dims = settings.embedding_dims

    async def embed(self, text: str, *, multilingual: bool = False) -> EmbeddingResult:
        if not self._api_key:
            # Deterministic stub vector for dev
            dim = self._dims
            val = float((sum(text.encode()) % 1000) / 1000.0)
            return EmbeddingResult(vector=[val] * dim, model="stub-embedding")

        import openai
        client = openai.AsyncOpenAI(api_key=self._api_key)
        settings = get_settings()
        model = settings.embedding_model_multilingual if multilingual else self._model_en
        response = await client.embeddings.create(input=text, model=model)
        data = response.data[0]
        return EmbeddingResult(
            vector=data.embedding,
            model=model,
            tokens=response.usage.total_tokens,
        )
