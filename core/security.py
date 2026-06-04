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

import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.config import get_settings
from core.logging import get_logger

log = get_logger(__name__)

Role = Literal["admin", "super_admin", "lawyer", "client", "auditor", "expert_reviewer"]
AdminRole = Literal["admin", "super_admin"]

_bearer = HTTPBearer(auto_error=True)
_VALID_ROLES: set[str] = {"admin", "super_admin", "lawyer", "client", "auditor", "expert_reviewer"}


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
            role=_normalise_role(decoded.get("role", "client")),
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
    return await _current_user_from_token(credentials.credentials)


async def get_admin_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> CurrentUser:
    """
    Admin dependency that accepts either backend-issued JWTs or Supabase JWTs.

    Supabase tokens are verified by calling Supabase Auth /user with the bearer
    token, then role is loaded from the server-side users table.
    """
    user = await _current_user_from_token(credentials.credentials)
    if user.role not in {"admin", "super_admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role is required for this resource.",
        )
    return user


def require_roles(*roles: Role):
    """
    Dependency factory.

    Usage:
        Depends(require_roles("admin", "lawyer"))
    """
    async def _check(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if not _role_allowed(user.role, roles):
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
        return await _current_user_from_token(credentials.credentials)
    except HTTPException:
        return None


async def _current_user_from_token(token: str) -> CurrentUser:
    try:
        return decode_token(token)
    except HTTPException as jwt_error:
        settings = get_settings()
        if not settings.supabase_url or not (settings.supabase_anon_key or settings.supabase_key):
            raise jwt_error

    try:
        auth_user_id = await _verify_supabase_token(token)
        profile = await _fetch_supabase_user_profile(auth_user_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Supabase session: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    settings = get_settings()
    return CurrentUser(
        sub=auth_user_id,
        role=_normalise_role(profile.get("role") or "client"),
        tenant_id=str(profile.get("tenant_id") or ""),
        exp=int(time.time()) + settings.jwt_access_ttl_seconds,
    )


def _normalise_role(value: Any) -> Role:
    role = str(value or "client")
    if role in _VALID_ROLES:
        return role  # type: ignore[return-value]
    return "client"


def _role_allowed(user_role: Role, allowed_roles: tuple[Role, ...]) -> bool:
    return user_role in allowed_roles or (user_role == "super_admin" and "admin" in allowed_roles)


async def _verify_supabase_token(token: str) -> str:
    settings = get_settings()
    headers = {
        "Authorization": f"Bearer {token}",
        "apikey": settings.supabase_anon_key or settings.supabase_key or "",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{settings.supabase_url}/auth/v1/user", headers=headers)

    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Supabase session is invalid or expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    data = response.json()
    user_id = data.get("id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Supabase session does not include a user id.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return str(user_id)


async def _fetch_supabase_user_profile(user_id: str) -> dict[str, Any]:
    from core.database import get_supabase

    supabase = await get_supabase()
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database is not configured.",
        )

    result = await (
        supabase.table("users")
        .select("id, role, tenant_id, is_active")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )
    profile = result.data[0] if isinstance(result.data, list) and result.data else result.data
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User profile was not found.",
        )
    if profile.get("is_active") is False:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin account is inactive.",
        )
    return profile
