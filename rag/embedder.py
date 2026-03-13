"""
rag/embedder.py
===============
Embedding generation with Redis caching.
Cache key = SHA256(text + model) → avoids re-embedding identical queries.
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from core.config import get_settings
from core.logging import get_logger
from services.llm_service import EmbeddingResult, EmbeddingService

if TYPE_CHECKING:
    import redis.asyncio as aioredis

log = get_logger(__name__)


class Embedder:
    def __init__(self, redis: "aioredis.Redis | None" = None) -> None:
        self._svc = EmbeddingService()
        self._redis = redis
        self._settings = get_settings()

    def _cache_key(self, text: str, multilingual: bool) -> str:
        raw = f"{text[:500]}|multi={multilingual}"
        return f"embed:{hashlib.sha256(raw.encode()).hexdigest()}"

    async def embed(self, text: str, *, multilingual: bool = False) -> EmbeddingResult:
        key = self._cache_key(text, multilingual)

        # Try cache first
        if self._redis:
            try:
                cached = await self._redis.get(key)
                if cached:
                    data = json.loads(cached)
                    return EmbeddingResult(**data)
            except Exception:
                pass

        result = await self._svc.embed(text, multilingual=multilingual)

        # Cache with TTL
        if self._redis:
            try:
                await self._redis.setex(
                    key,
                    self._settings.cache_ttl_embedding_seconds,
                    json.dumps({"vector": result.vector, "model": result.model, "tokens": result.tokens}),
                )
            except Exception as exc:
                log.debug("embedder.cache_set.failed", error=str(exc))

        return result
