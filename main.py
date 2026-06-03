"""
main.py
=======
FastAPI application factory.

Design:
- `create_app()` is the single entry point — never import `app` directly from tests
- Lifespan context manager handles graceful startup/shutdown
- All middleware registered in explicit order (CORS → RateLimit → Logging)
- Exception handlers registered before routes
- Health endpoint bypasses auth — used by load balancers
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

import structlog.contextvars
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import admin as admin_router
from api import billing as billing_router
from api import documents as documents_router
from api import evidence as evidence_router
from api import evaluation as evaluation_router
from api import feedback as feedback_router
from api import knowledge as knowledge_router
from api import legal as legal_router
from api import memory as memory_router
from core.config import get_settings
from core.database import close_connections, ping_redis, ping_supabase
from core.exceptions import register_exception_handlers
from core.logging import configure_logging, get_logger
from middleware.rate_limiter import RateLimiterMiddleware

log = get_logger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    # Initialize Sentry for production error tracking
    if settings.is_production() and SENTRY_AVAILABLE and settings.sentry_dsn:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            traces_sample_rate=0.1,  # 10% sampling for performance monitoring
            profiles_sample_rate=0.1,
            environment=settings.app_env,
            release=f"ai-lawyer@{settings.app_version}",
        )
        log.info("sentry.initialized")
    elif settings.is_production() and not SENTRY_AVAILABLE:
        log.warning("sentry.not_installed", reason="sentry-sdk not installed but running in production")

    log.info(
        "app.startup",
        name=settings.app_name,
        version=settings.app_version,
        env=settings.app_env,
    )

    # Warm-up checks (non-blocking — app starts even if services are down)
    redis_ok = await ping_redis()
    supabase_ok = await ping_supabase()

    log.info(
        "app.connectivity",
        redis=redis_ok,
        supabase=supabase_ok,
    )

    if settings.is_production() and not supabase_ok:
        log.error("app.startup.failed", reason="Supabase required in production")
        raise RuntimeError("Cannot start: Supabase is required in production mode.")

    yield  # ← app is running

    log.info("app.shutdown")
    await close_connections()


# ── Request logging middleware ─────────────────────────────────────────────────

async def _request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Request-ID"] = request_id
    log.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )
    return response


async def _security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response


async def _body_size_limit_middleware(request: Request, call_next):
    settings = get_settings()
    limit_bytes = settings.max_request_body_mb * 1024 * 1024
    content_length = request.headers.get("content-length")

    if content_length:
        try:
            body_bytes = int(content_length)
        except ValueError:
            body_bytes = 0

        if body_bytes > limit_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "REQUEST_TOO_LARGE",
                    "message": f"Request body exceeds {settings.max_request_body_mb}MB limit.",
                },
            )

    return await call_next(request)


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI Lawyer Backend",
        description=(
            "Multi-Agent RAG Legal Advisory System with IRAC Reasoning, "
            "Citation Verification, Case Memory, and Evidence Analysis."
        ),
        version=settings.app_version,
        docs_url="/docs" if not settings.is_production() else None,
        redoc_url="/redoc" if not settings.is_production() else None,
        lifespan=lifespan,
    )

    # ── Middleware (order matters — outermost = last registered) ───────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )
    
    # Rate limiting middleware (lazy initialization - doesn't need Redis at startup)
    app.add_middleware(RateLimiterMiddleware, redis=None)
    
    app.middleware("http")(_body_size_limit_middleware)
    app.middleware("http")(_security_headers_middleware)
    app.middleware("http")(_request_logging_middleware)

    # ── Exception handlers ─────────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(legal_router.router)
    app.include_router(documents_router.router)
    app.include_router(evidence_router.router)
    app.include_router(memory_router.router)
    app.include_router(feedback_router.router)
    app.include_router(billing_router.router)
    app.include_router(evaluation_router.router)
    app.include_router(knowledge_router.router)
    app.include_router(admin_router.router)

    # ── System endpoints ───────────────────────────────────────────────────────

    @app.get("/health", tags=["system"], summary="Health check for load balancers")
    async def health() -> dict:
        redis_ok = await ping_redis()
        supabase_ok = await ping_supabase()
        overall = "ok" if (redis_ok and supabase_ok) else "degraded"
        return {
            "status": overall,
            "redis": redis_ok,
            "supabase": supabase_ok,
            "version": settings.app_version,
        }

    @app.get("/", tags=["system"], include_in_schema=False)
    async def root() -> dict:
        return {"service": settings.app_name, "version": settings.app_version, "docs": "/docs"}

    return app


# ── ASGI entry point ───────────────────────────────────────────────────────────
app = create_app()
