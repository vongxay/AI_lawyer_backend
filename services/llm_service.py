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

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import get_settings
from core.exceptions import ExternalServiceError, ProviderNotConfiguredError
from core.logging import get_logger

log = get_logger(__name__)


def _has_real_api_key(value: str | None) -> bool:
    if not value:
        return False
    stripped = value.strip()
    return bool(stripped) and "..." not in stripped and not stripped.endswith("-")


def _is_openrouter_api_key(value: str | None) -> bool:
    return bool(value and value.strip().startswith("sk-or-"))


def _is_openrouter_base_url(value: str | None) -> bool:
    return bool(value and "openrouter.ai" in value.lower())


def _clean_base_url(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().rstrip("/") or None


def _openrouter_headers() -> dict[str, str]:
    settings = get_settings()
    headers: dict[str, str] = {}
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_title:
        headers["X-OpenRouter-Title"] = settings.openrouter_app_title
    return headers


def _is_openai_family_model(model: str) -> bool:
    return model.lower().startswith(("gpt", "o1", "o3", "o4", "chatgpt"))


def _is_anthropic_family_model(model: str) -> bool:
    return model.lower().startswith("claude")


def _looks_like_openrouter_model(model: str) -> bool:
    return "/" in model


def _normalise_openrouter_model(model: str) -> str:
    if _looks_like_openrouter_model(model):
        return model
    if _is_openai_family_model(model):
        return f"openai/{model}"
    if _is_anthropic_family_model(model):
        return f"anthropic/{model}"
    return model


def _normalise_openrouter_embedding_model(model: str) -> str:
    if "/" in model:
        return model
    if model.startswith("text-embedding-"):
        return f"openai/{model}"
    return model


def _base_model_name(model: str) -> str:
    return model.rsplit("/", 1)[-1]


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
        kwargs: dict = dict(model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature)
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
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        kwargs: dict = dict(model=model, messages=msgs, max_tokens=max_tokens)
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for chunk in stream.text_stream:
                yield chunk


class _OpenAIProvider:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        provider_name: str = "openai",
    ) -> None:
        import openai

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if default_headers:
            kwargs["default_headers"] = default_headers

        self._client = openai.AsyncOpenAI(**kwargs)
        self._provider_name = provider_name

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
            provider=self._provider_name,
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




class LlmService:
    """
    Route generate calls to the correct provider based on model name.

    Model prefix mapping:
        claude-*  → Anthropic
        gpt-*     → OpenAI
        unknown   → provider-not-configured error
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._anthropic: _AnthropicProvider | None = None
        self._openai: _OpenAIProvider | None = None
        self._openrouter: _OpenAIProvider | None = None

        if _has_real_api_key(settings.anthropic_api_key):
            self._anthropic = _AnthropicProvider(settings.anthropic_api_key)

        if _has_real_api_key(settings.openai_api_key):
            base_url = _clean_base_url(settings.openai_base_url)
            if _is_openrouter_api_key(settings.openai_api_key) or _is_openrouter_base_url(base_url):
                self._openrouter = _OpenAIProvider(
                    settings.openai_api_key,
                    base_url=base_url or _clean_base_url(settings.openrouter_base_url),
                    default_headers=_openrouter_headers(),
                    provider_name="openrouter",
                )
            else:
                self._openai = _OpenAIProvider(
                    settings.openai_api_key,
                    base_url=base_url,
                    provider_name="openai-compatible" if base_url else "openai",
                )

        if _has_real_api_key(settings.openrouter_api_key):
            self._openrouter = _OpenAIProvider(
                settings.openrouter_api_key,
                base_url=_clean_base_url(settings.openrouter_base_url),
                default_headers=_openrouter_headers(),
                provider_name="openrouter",
            )


    def _get_provider(self, model: str):
        if _looks_like_openrouter_model(model):
            return model, self._openrouter
        if _is_anthropic_family_model(model):
            if self._anthropic is not None:
                return model, self._anthropic
            return _normalise_openrouter_model(model), self._openrouter
        if _is_openai_family_model(model):
            if self._openai is not None:
                return model, self._openai
            return _normalise_openrouter_model(model), self._openrouter
        return None

    def _required_env_for_model(self, model: str) -> list[str]:
        if _looks_like_openrouter_model(model):
            return ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]
        if _is_anthropic_family_model(model):
            return ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"]
        if _is_openai_family_model(model):
            return ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]
        return ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"]

    def _cap_max_tokens(self, requested: int) -> int:
        settings = get_settings()
        cap = max(1, int(settings.llm_max_tokens_absolute))
        effective = max(1, min(int(requested), cap))
        if effective != requested:
            log.warning("llm.max_tokens.capped", requested_max_tokens=requested, max_tokens=effective)
        return effective

    def _resolve_model_and_provider(self, model: str):
        resolved = self._get_provider(model)
        if resolved is not None:
            effective_model, provider = resolved
            if provider is not None:
                return effective_model, provider

        settings = get_settings()
        fallback_model = settings.model_economy
        fallback_resolved = self._get_provider(fallback_model)
        if (
            settings.llm_fallback_to_economy_model
            and fallback_resolved is not None
            and fallback_resolved[1] is not None
            and fallback_model != model
        ):
            log.warning(
                "llm.model_fallback",
                requested_model=model,
                fallback_model=fallback_model,
            )
            return fallback_resolved

        return model, None

    def _fallback_model_and_provider(self, failed_model: str):
        settings = get_settings()
        if not settings.llm_fallback_to_economy_model:
            return None, None

        fallback_model = settings.model_economy
        if not fallback_model or fallback_model == failed_model:
            return None, None

        resolved = self._get_provider(fallback_model)
        if resolved is None or resolved[1] is None:
            return None, None
        return resolved

    async def generate(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> LlmResult:
        effective_model, provider = self._resolve_model_and_provider(model)
        if provider is None:
            raise ProviderNotConfiguredError(
                "No real LLM API key is configured for this model.",
                details={"model": model, "required_env": self._required_env_for_model(effective_model)},
            )
        effective_max_tokens = self._cap_max_tokens(max_tokens)
        try:
            result = await provider.generate(
                model=effective_model,
                messages=messages,
                system=system,
                max_tokens=effective_max_tokens,
                temperature=temperature,
            )
            log.info(
                "llm.generate.ok",
                model=effective_model,
                requested_model=model,
                provider=result.provider,
                tokens_in=result.input_tokens,
                tokens_out=result.output_tokens,
            )
            return result
        except Exception as exc:
            fallback_model, fallback_provider = self._fallback_model_and_provider(effective_model)
            if fallback_provider is not None and fallback_model:
                log.warning(
                    "llm.generate.fallback_after_error",
                    model=effective_model,
                    fallback_model=fallback_model,
                    error=str(exc),
                )
                try:
                    result = await fallback_provider.generate(
                        model=fallback_model,
                        messages=messages,
                        system=system,
                        max_tokens=effective_max_tokens,
                        temperature=temperature,
                    )
                    log.info(
                        "llm.generate.fallback_ok",
                        model=fallback_model,
                        requested_model=model,
                        provider=result.provider,
                        tokens_in=result.input_tokens,
                        tokens_out=result.output_tokens,
                    )
                    return result
                except Exception as fallback_exc:
                    log.error(
                        "llm.generate.fallback_failed",
                        model=effective_model,
                        fallback_model=fallback_model,
                        error=str(fallback_exc),
                    )
                    raise ExternalServiceError(
                        f"LLM call failed for model {effective_model}; fallback {fallback_model} also failed: "
                        f"{fallback_exc}"
                    ) from fallback_exc

            log.error("llm.generate.failed", model=effective_model, requested_model=model, error=str(exc))
            raise ExternalServiceError(f"LLM call failed for model {effective_model}: {exc}") from exc

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        effective_model, provider = self._resolve_model_and_provider(model)
        if provider is None:
            raise ProviderNotConfiguredError(
                "No real LLM API key is configured for streaming.",
                details={"model": model, "required_env": self._required_env_for_model(effective_model)},
            )
        effective_max_tokens = self._cap_max_tokens(max_tokens)
        try:
            async for chunk in provider.stream(
                model=effective_model,
                messages=messages,
                system=system,
                max_tokens=effective_max_tokens,
            ):
                yield chunk
        except Exception as exc:
            fallback_model, fallback_provider = self._fallback_model_and_provider(effective_model)
            if fallback_provider is not None and fallback_model:
                log.warning(
                    "llm.stream.fallback_after_error",
                    model=effective_model,
                    fallback_model=fallback_model,
                    error=str(exc),
                )
                try:
                    async for chunk in fallback_provider.stream(
                        model=fallback_model,
                        messages=messages,
                        system=system,
                        max_tokens=effective_max_tokens,
                    ):
                        yield chunk
                    return
                except Exception as fallback_exc:
                    log.error(
                        "llm.stream.fallback_failed",
                        model=effective_model,
                        fallback_model=fallback_model,
                        error=str(fallback_exc),
                    )
                    raise ExternalServiceError(
                        f"LLM stream failed for model {effective_model}; fallback {fallback_model} also failed: "
                        f"{fallback_exc}"
                    ) from fallback_exc

            raise ExternalServiceError(f"LLM stream failed for model {effective_model}: {exc}") from exc


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
        self._api_key: str | None = None
        self._model_en = settings.embedding_model_en
        self._dims = settings.embedding_dims
        self._client = None
        self._provider_name = "openai"

        if _has_real_api_key(settings.openrouter_api_key):
            self._api_key = settings.openrouter_api_key
            self._provider_name = "openrouter"
            base_url = _clean_base_url(settings.openrouter_base_url)
            default_headers = _openrouter_headers()
        elif _has_real_api_key(settings.openai_api_key):
            self._api_key = settings.openai_api_key
            base_url = _clean_base_url(settings.openai_base_url)
            if _is_openrouter_api_key(self._api_key) or _is_openrouter_base_url(base_url):
                self._provider_name = "openrouter"
                base_url = base_url or _clean_base_url(settings.openrouter_base_url)
                default_headers = _openrouter_headers()
            else:
                self._provider_name = "openai-compatible" if base_url else "openai"
                default_headers = None
        else:
            base_url = None
            default_headers = None

        if self._api_key:
            import openai

            kwargs: dict = {"api_key": self._api_key}
            if base_url:
                kwargs["base_url"] = base_url
            if default_headers:
                kwargs["default_headers"] = default_headers
            self._client = openai.AsyncOpenAI(**kwargs)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _create_embeddings(self, request: dict[str, Any]) -> Any:
        return await self._client.embeddings.create(**request)

    async def embed(self, text: str, *, multilingual: bool = False) -> EmbeddingResult:
        results = await self.embed_many([text], multilingual=multilingual)
        return results[0]

    async def embed_many(self, texts: list[str], *, multilingual: bool = False) -> list[EmbeddingResult]:
        if not self._api_key:
            raise ProviderNotConfiguredError(
                "No real embedding API key is configured.",
                details={"required_env": ["OPENAI_API_KEY"]},
            )
        if not texts:
            return []

        settings = get_settings()
        model = settings.embedding_model_multilingual if multilingual else self._model_en
        if self._provider_name == "openrouter":
            model = _normalise_openrouter_embedding_model(model)
        request: dict = {"input": texts, "model": model}
        if self._dims and _base_model_name(model).startswith("text-embedding-3"):
            request["dimensions"] = int(self._dims)

        response = await self._create_embeddings(request)
        data = list(getattr(response, "data", None) or [])
        if not data:
            raise ExternalServiceError(
                "Embedding provider returned no vectors.",
                details={
                    "provider": self._provider_name,
                    "model": model,
                    "input_count": len(texts),
                },
            )

        usage = getattr(response, "usage", None)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        per_item_tokens = total_tokens // max(1, len(data))
        return [
            EmbeddingResult(
                vector=item.embedding,
                model=model,
                tokens=per_item_tokens,
            )
            for item in sorted(data, key=lambda item: getattr(item, "index", 0))
        ]
