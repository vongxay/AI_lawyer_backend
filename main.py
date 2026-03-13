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
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import admin as admin_router
from backend.api import documents as documents_router
from backend.api import evidence as evidence_router
from backend.api import feedback as feedback_router
from backend.api import legal as legal_router
from backend.api import memory as memory_router
from backend.core.config import get_settings
from backend.core.database import close_connections, ping_redis, ping_supabase
from backend.core.exceptions import register_exception_handlers
from backend.core.logging import configure_logging, get_logger

log = get_logger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

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
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )
    return response


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
    app.middleware("http")(_request_logging_middleware)

    # ── Exception handlers ─────────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(legal_router.router)
    app.include_router(documents_router.router)
    app.include_router(evidence_router.router)
    app.include_router(memory_router.router)
    app.include_router(feedback_router.router)
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
