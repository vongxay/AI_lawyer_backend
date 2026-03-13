"""
core/exceptions.py
==================
Domain exceptions with FastAPI exception handler registration.

Every layer raises domain exceptions — never raw HTTPException from business logic.
The handlers here translate domain exceptions to HTTP responses at the boundary.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from core.logging import get_logger

log = get_logger(__name__)


# ── Domain Exceptions ──────────────────────────────────────────────────────────

class AILawyerError(Exception):
    """Base class for all domain errors."""
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(AILawyerError):
    status_code = 422
    error_code = "VALIDATION_ERROR"


class NotFoundError(AILawyerError):
    status_code = 404
    error_code = "NOT_FOUND"


class LowConfidenceError(AILawyerError):
    """Raised when agent confidence falls below refusal threshold."""
    status_code = 422
    error_code = "LOW_CONFIDENCE"


class CitationHallucinationError(AILawyerError):
    """Raised when citation rejection rate exceeds safety threshold."""
    status_code = 422
    error_code = "CITATION_HALLUCINATION"


class AgentTimeoutError(AILawyerError):
    status_code = 504
    error_code = "AGENT_TIMEOUT"


class ExternalServiceError(AILawyerError):
    """LLM API or Supabase returned an unexpected error."""
    status_code = 502
    error_code = "EXTERNAL_SERVICE_ERROR"


class TenantIsolationError(AILawyerError):
    """Attempt to access data outside caller's tenant."""
    status_code = 403
    error_code = "TENANT_ISOLATION_VIOLATION"


class FileTooLargeError(AILawyerError):
    status_code = 413
    error_code = "FILE_TOO_LARGE"


class UnsupportedFileTypeError(AILawyerError):
    status_code = 415
    error_code = "UNSUPPORTED_FILE_TYPE"


# ── Handler registration ───────────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AILawyerError)
    async def domain_exception_handler(request: Request, exc: AILawyerError) -> JSONResponse:
        log.warning(
            "domain.exception",
            error_code=exc.error_code,
            message=exc.message,
            path=request.url.path,
            details=exc.details,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.error(
            "unhandled.exception",
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Please try again.",
            },
        )
