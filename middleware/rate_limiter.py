"""
middleware/rate_limiter.py
===========================
Token bucket rate limiting middleware for FastAPI.

Features:
- Per-user rate limiting using Redis
- Configurable limits per endpoint type
- Graceful degradation when Redis unavailable
- Returns 429 Too Many Requests with retry-after header
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from core.config import get_settings
from core.logging import get_logger

log = get_logger(__name__)


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    reset_at: int
    retry_after: int | None = None


class InMemoryRateLimiter:
    """Fallback rate limiter when Redis is unavailable."""

    def __init__(self) -> None:
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def check(self, key: str, limit: int, window_seconds: int = 60) -> RateLimitResult:
        now = time.time()
        cutoff = now - window_seconds

        # Remove old entries
        self._buckets[key] = [t for t in self._buckets[key] if t > cutoff]

        if len(self._buckets[key]) >= limit:
            reset_at = int(self._buckets[key][0] + window_seconds)
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=reset_at,
                retry_after=max(1, reset_at - int(now)),
            )

        self._buckets[key].append(now)
        return RateLimitResult(
            allowed=True,
            remaining=limit - len(self._buckets[key]),
            reset_at=int(now + window_seconds),
        )


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with Redis backend and in-memory fallback.

    Usage:
        app.add_middleware(RateLimiterMiddleware)

    Rate limits are applied based on:
    1. User ID (if authenticated)
    2. IP address (if anonymous)
    """

    def __init__(
        self,
        app,
        redis=None,
        default_limit: int | None = None,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self._settings = get_settings()
        self._redis = redis
        self._in_memory = InMemoryRateLimiter()
        self._default_limit = default_limit or self._settings.rate_limit_per_minute
        self._window = window_seconds

        # Endpoint-specific limits (override default)
        self._endpoint_limits: dict[str, int] = {
            "/api/v1/legal/query": 10,      # More restrictive for expensive queries
            "/api/v1/legal/query/stream": 10,
            "/api/v1/documents/analyze": 5,  # Very restrictive - expensive operation
            "/api/v1/evidence/analyze": 5,
        }

    async def dispatch(self, request: Request, call_next: Callable):
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Get client identifier
        client_id = await self._get_client_id(request)
        endpoint = request.url.path

        # Get rate limit for this endpoint
        limit = self._endpoint_limits.get(endpoint, self._default_limit)

        # Check rate limit
        result = await self._check_rate_limit(client_id, limit)

        # Add rate limit headers to response
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(result.reset_at),
        }

        if not result.allowed:
            log.warning(
                "rate_limit.exceeded",
                client_id=client_id,
                endpoint=endpoint,
                limit=limit,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": result.retry_after,
                },
                headers=headers,
            )

        response = await call_next(request)

        # Add headers to successful response
        for key, value in headers.items():
            response.headers[key] = value

        return response

    async def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Try user ID from state (set by auth middleware)
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.sub}"

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}"

    async def _check_rate_limit(self, key: str, limit: int) -> RateLimitResult:
        """Check rate limit using Redis or fallback."""
        if self._redis:
            try:
                return await self._redis_check(key, limit)
            except Exception as exc:
                log.warning("rate_limit.redis.failed", error=str(exc))
                # Fall through to in-memory

        return await self._in_memory.check(key, limit, self._window)

    async def _redis_check(self, key: str, limit: int) -> RateLimitResult:
        """Redis-based rate limiting using sliding window."""
        now = time.time()
        window_key = f"ratelimit:{key}:{int(now // self._window)}"

        pipe = self._redis.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, self._window * 2)
        results = await pipe.execute()

        current_count = results[0]
        reset_at = int((int(now // self._window) + 1) * self._window)

        if current_count > limit:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=reset_at,
                retry_after=max(1, reset_at - int(now)),
            )

        return RateLimitResult(
            allowed=True,
            remaining=limit - current_count,
            reset_at=reset_at,
        )
