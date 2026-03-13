"""
core/security.py
================
JWT creation / verification and FastAPI dependency injection for auth.

Roles:
    admin   — full access including audit log and expert queue
    lawyer  — full legal query + case management
    client  — query + own case memory only
    auditor — read-only access to audit log

Usage in routes:
    @router.get("/sensitive")
    async def endpoint(user: CurrentUser = Depends(require_roles("lawyer", "admin"))):
        ...
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.config import get_settings
from core.logging import get_logger

log = get_logger(__name__)

Role = Literal["admin", "lawyer", "client", "auditor"]

_bearer = HTTPBearer(auto_error=True)


# ── Token payload ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CurrentUser:
    sub: str          # user UUID
    role: Role
    tenant_id: str
    exp: int


# ── Token utilities ────────────────────────────────────────────────────────────

def create_access_token(
    subject: str,
    role: Role = "client",
    tenant_id: str = "",
    ttl_seconds: int | None = None,
) -> str:
    settings = get_settings()
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "tenant_id": tenant_id,
        "exp": now + (ttl_seconds or settings.jwt_access_ttl_seconds),
        "iat": now,
        "type": "access",
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> CurrentUser:
    settings = get_settings()
    try:
        decoded = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        return CurrentUser(
            sub=str(decoded["sub"]),
            role=decoded.get("role", "client"),
            tenant_id=str(decoded.get("tenant_id", "")),
            exp=int(decoded["exp"]),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI Dependencies ───────────────────────────────────────────────────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> CurrentUser:
    return decode_token(credentials.credentials)


def require_roles(*roles: Role):
    """
    Dependency factory.

    Usage:
        Depends(require_roles("admin", "lawyer"))
    """
    async def _check(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in roles:
            log.warning(
                "auth.forbidden",
                user_id=user.sub,
                user_role=user.role,
                required_roles=roles,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' is not authorised for this resource.",
            )
        return user

    return _check


# Optional — open endpoints still get a user if token present, else anonymous
async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> CurrentUser | None:
    if credentials is None:
        return None
    try:
        return decode_token(credentials.credentials)
    except HTTPException:
        return None
