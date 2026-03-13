"""
core/database.py
================
Manages Supabase + Redis client lifecycle.

Design decisions:
- Clients are created lazily on first access (not at import time)
- Redis uses connection pooling (max_connections from settings)
- Supabase client is optional — app degrades gracefully when not configured
- Provides async health-check helpers used by /health endpoint
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

from backend.core.config import get_settings
from backend.core.logging import get_logger

if TYPE_CHECKING:
    from supabase import AsyncClient  # pragma: no cover

log = get_logger(__name__)

# ── Module-level singletons ────────────────────────────────────────────────────
_redis_pool: ConnectionPool | None = None
_supabase_client: "AsyncClient | None" = None
_init_lock = asyncio.Lock()


# ── Redis ──────────────────────────────────────────────────────────────────────

async def get_redis() -> aioredis.Redis:
    """Return a Redis connection from the shared pool."""
    global _redis_pool
    if _redis_pool is None:
        settings = get_settings()
        _redis_pool = ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )
        log.info("redis.pool.created", url=settings.redis_url)
    return aioredis.Redis(connection_pool=_redis_pool)


async def ping_redis() -> bool:
    """Health check — returns True if Redis responds."""
    try:
        r = await get_redis()
        return await r.ping()
    except Exception as exc:
        log.warning("redis.ping.failed", error=str(exc))
        return False


# ── Supabase ───────────────────────────────────────────────────────────────────

async def get_supabase() -> "AsyncClient | None":
    """Return the Supabase async client, or None if not configured."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    async with _init_lock:
        if _supabase_client is not None:   # double-check inside lock
            return _supabase_client

        settings = get_settings()
        if not (settings.supabase_url and settings.supabase_key):
            log.warning("supabase.not_configured", reason="missing SUPABASE_URL or SUPABASE_KEY")
            return None

        try:
            from supabase._async.client import create_client  # type: ignore

            _supabase_client = await create_client(settings.supabase_url, settings.supabase_key)
            log.info("supabase.connected", url=settings.supabase_url)
        except Exception as exc:
            log.error("supabase.init.failed", error=str(exc))
            return None

    return _supabase_client


async def ping_supabase() -> bool:
    """Health check — returns True if Supabase responds."""
    try:
        client = await get_supabase()
        if client is None:
            return False
        # Lightweight query — just check auth health
        await client.table("users").select("id").limit(1).execute()
        return True
    except Exception as exc:
        log.warning("supabase.ping.failed", error=str(exc))
        return False


# ── Teardown ───────────────────────────────────────────────────────────────────

async def close_connections() -> None:
    """Graceful shutdown — call in app lifespan."""
    global _redis_pool, _supabase_client
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None
        log.info("redis.pool.closed")
    _supabase_client = None


# ── Convenience dataclass for DI ──────────────────────────────────────────────

@dataclass(frozen=True)
class DbContext:
    """Passed into services that need both clients."""
    supabase: "AsyncClient | None"
    redis: aioredis.Redis
