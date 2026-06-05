"""
services/cache_service.py
==========================
Redis-based caching service with automatic serialization and TTL management.

Features:
- Type-safe caching with Pydantic models
- Automatic JSON serialization/deserialization
- Configurable TTL per cache type
- Graceful degradation when Redis unavailable
- Cache key namespacing for organization
"""
from __future__ import annotations

import json
from typing import Any, Generic, TypeVar

import redis.asyncio as aioredis

from core.config import get_settings
from core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class CacheService:
    """
    Async Redis cache service with automatic TTL management.

    Usage:
        cache = CacheService(redis_client)
        await cache.set("key", data, ttl=3600)
        data = await cache.get("key")
    """

    def __init__(self, redis_client: aioredis.Redis | None = None) -> None:
        self._redis = redis_client
        self._settings = get_settings()

        # Namespace prefixes
        self._prefixes = {
            "embedding": "cache:embedding:",
            "legal_qa": "cache:legal_qa:",
            "law_summary": "cache:law_summary:",
            "session": "session:",
            "rate_limit": "ratelimit:",
        }

        # Default TTLs (seconds)
        self._ttls = {
            "embedding": self._settings.cache_ttl_embedding_seconds,
            "legal_qa": self._settings.cache_ttl_legal_qa_seconds,
            "law_summary": self._settings.cache_ttl_law_summary_seconds,
            "session": 86400,  # 24 hours
            "rate_limit": 60,   # 1 minute
        }

    async def get(self, key: str, namespace: str = "default") -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key
            namespace: Key namespace (embedding, legal_qa, law_summary, session)

        Returns:
            Cached value or None if not found/error
        """
        if not self._redis:
            return None

        try:
            full_key = self._build_key(key, namespace)
            value = await self._redis.get(full_key)

            if value is None:
                return None

            # Try to parse as JSON first
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except Exception as exc:
            log.warning("cache.get.failed", key=key, namespace=namespace, error=str(exc))
            return None

    async def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: int | None = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            namespace: Key namespace
            ttl: Time-to-live in seconds (uses namespace default if not specified)

        Returns:
            True if successful, False otherwise
        """
        if not self._redis:
            return False

        try:
            full_key = self._build_key(key, namespace)

            # Serialize value
            if isinstance(value, (str, int, float, bytes)):
                serialized = value
            else:
                serialized = json.dumps(value, ensure_ascii=False, default=str)

            # Get TTL
            cache_ttl = ttl or self._ttls.get(namespace, 3600)

            await self._redis.set(full_key, serialized, ex=cache_ttl)

            log.debug(
                "cache.set.ok",
                key=key,
                namespace=namespace,
                ttl=cache_ttl,
            )
            return True

        except Exception as exc:
            log.error("cache.set.failed", key=key, namespace=namespace, error=str(exc))
            return False

    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete value from cache."""
        if not self._redis:
            return False

        try:
            full_key = self._build_key(key, namespace)
            await self._redis.delete(full_key)
            return True
        except Exception as exc:
            log.warning("cache.delete.failed", key=key, error=str(exc))
            return False

    async def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if key exists in cache."""
        if not self._redis:
            return False

        try:
            full_key = self._build_key(key, namespace)
            return await self._redis.exists(full_key) > 0
        except Exception as exc:
            log.warning("cache.exists.failed", key=key, error=str(exc))
            return False

    async def clear_namespace(self, namespace: str) -> bool:
        """
        Clear all keys in a namespace.

        WARNING: Uses SCAN - may be slow on large datasets
        """
        if not self._redis:
            return False

        try:
            prefix = self._prefixes.get(namespace, f"cache:{namespace}:")
            cursor = 0

            while True:
                cursor, keys = await self._redis.scan(cursor, match=f"{prefix}*", count=100)
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break

            log.info("cache.namespace.cleared", namespace=namespace)
            return True

        except Exception as exc:
            log.error("cache.clear_namespace.failed", namespace=namespace, error=str(exc))
            return False

    def _build_key(self, key: str, namespace: str) -> str:
        """Build full cache key with namespace prefix."""
        prefix = self._prefixes.get(namespace, f"cache:{namespace}:")
        return f"{prefix}{key}"

    async def ping(self) -> bool:
        """Test Redis connectivity."""
        if not self._redis:
            return False

        try:
            return await self._redis.ping()
        except Exception:
            return False

    async def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not self._redis:
            return {"available": False}

        try:
            info = await self._redis.info("memory")
            db_size = await self._redis.dbsize()

            return {
                "available": True,
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "keys_count": db_size,
                "connected_clients": info.get("connected_clients", 0),
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}


# Convenience function for dependency injection
async def get_cache_service() -> CacheService:
    """Get cache service instance with Redis connection."""
    from core.database import get_redis

    redis_client = await get_redis()
    return CacheService(redis_client)
