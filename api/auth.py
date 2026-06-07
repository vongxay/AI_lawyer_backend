"""
api/auth.py
===========
Authentication bridge for the React admin app.
"""
from __future__ import annotations

from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from api.users import _fetch_user_profile, _normalise_profile
from core.config import get_settings
from core.database import get_supabase
from core.security import CurrentUser, get_admin_user

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
AdminUser = Annotated[CurrentUser, Depends(get_admin_user)]
BearerCredentials = Annotated[HTTPAuthorizationCredentials, Depends(HTTPBearer())]


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1)


class ClientProfileSyncRequest(BaseModel):
    full_name: str | None = Field(default=None, max_length=255)


@router.post("/login", summary="Sign in an admin through Supabase Auth")
async def login(payload: LoginRequest) -> dict[str, Any]:
    settings = get_settings()
    api_key = _supabase_api_key(settings)
    if not settings.supabase_url or not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase authentication is not configured.",
        )

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            f"{settings.supabase_url}/auth/v1/token?grant_type=password",
            headers={"apikey": api_key, "Content-Type": "application/json"},
            json={"email": payload.email, "password": payload.password},
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth_data = response.json()
    auth_user = auth_data.get("user") or {}
    user_id = str(auth_user.get("id") or "")
    supabase = await get_supabase()
    profile = await _fetch_user_profile(supabase, user_id) if supabase and user_id else None
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin profile was not found.",
        )

    normalised = _normalise_profile(profile)
    if normalised["role"] not in {"admin", "super_admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role is required for this application.",
        )

    return {
        "access_token": auth_data.get("access_token"),
        "refresh_token": auth_data.get("refresh_token"),
        "token_type": auth_data.get("token_type", "bearer"),
        "expires_in": auth_data.get("expires_in"),
        "user": {
            **auth_user,
            **normalised,
        },
    }


@router.post("/logout", summary="Sign out the current admin session")
async def logout(user: AdminUser, credentials: BearerCredentials) -> dict[str, str]:
    await _revoke_supabase_session(credentials.credentials)
    _ = user
    return {"status": "signed_out"}


@router.get("/me", summary="Get current admin profile")
async def me(user: AdminUser) -> dict[str, Any]:
    supabase = await get_supabase()
    profile = await _fetch_user_profile(supabase, user.sub) if supabase else None
    if profile:
        return _normalise_profile(profile)
    return {
        "id": user.sub,
        "email": "",
        "full_name": "Admin",
        "role": user.role,
        "tenant_id": user.tenant_id,
        "is_active": True,
    }


@router.post("/client/profile", summary="Ensure the current Supabase user has a client profile")
async def ensure_client_profile(
    payload: ClientProfileSyncRequest,
    credentials: BearerCredentials,
) -> dict[str, Any]:
    settings = get_settings()
    supabase = await get_supabase()
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database is not configured.",
        )

    auth_user = await _fetch_supabase_auth_user(credentials.credentials)
    user_id = str(auth_user.get("id") or "")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Supabase session does not include a user id.",
        )

    email = str(auth_user.get("email") or "").strip()
    metadata = auth_user.get("user_metadata") or auth_user.get("raw_user_meta_data") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    full_name = (
        (payload.full_name or "").strip()
        or str(metadata.get("full_name") or metadata.get("name") or "").strip()
        or email
        or "Client"
    )

    existing = await _fetch_user_profile(supabase, user_id)
    if existing:
        if existing.get("is_active") is False:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User profile is inactive.",
            )
        updates: dict[str, Any] = {}
        if email and not existing.get("email"):
            updates["email"] = email
        if full_name and (not existing.get("full_name") or payload.full_name):
            updates["full_name"] = full_name
        if not existing.get("tenant_id"):
            updates["tenant_id"] = settings.default_tenant_id
        if updates:
            await _update_user_profile(supabase, user_id, updates)
            existing = await _fetch_user_profile(supabase, user_id) or existing
        await _ensure_public_profile(supabase, user_id=user_id, tenant_id=str(existing.get("tenant_id") or settings.default_tenant_id), full_name=full_name)
        return _normalise_profile(existing)

    created = await _create_client_profile(
        supabase,
        user_id=user_id,
        tenant_id=settings.default_tenant_id,
        email=email or f"{user_id}@unknown.local",
        full_name=full_name,
    )
    await _ensure_public_profile(
        supabase,
        user_id=user_id,
        tenant_id=settings.default_tenant_id,
        full_name=full_name,
    )
    return _normalise_profile(created)


def _supabase_api_key(settings: Any) -> str | None:
    return settings.supabase_anon_key or settings.supabase_key


async def _fetch_supabase_auth_user(token: str) -> dict[str, Any]:
    settings = get_settings()
    api_key = _supabase_api_key(settings)
    if not settings.supabase_url or not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase authentication is not configured.",
        )

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            f"{settings.supabase_url}/auth/v1/user",
            headers={"Authorization": f"Bearer {token}", "apikey": api_key},
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Supabase session is invalid or expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return response.json()


async def _revoke_supabase_session(token: str) -> None:
    settings = get_settings()
    api_key = _supabase_api_key(settings)
    if not settings.supabase_url or not api_key:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                f"{settings.supabase_url}/auth/v1/logout",
                headers={"Authorization": f"Bearer {token}", "apikey": api_key},
            )
    except Exception:
        return


async def _create_client_profile(
    supabase: Any,
    *,
    user_id: str,
    tenant_id: str,
    email: str,
    full_name: str,
) -> dict[str, Any]:
    payload = {
        "id": user_id,
        "tenant_id": tenant_id,
        "email": email,
        "full_name": full_name,
        "role": "client",
        "is_active": True,
    }
    try:
        result = await supabase.table("users").insert(payload).execute()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create client profile.",
        ) from exc
    data = result.data or []
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return payload


async def _update_user_profile(supabase: Any, user_id: str, updates: dict[str, Any]) -> None:
    try:
        await supabase.table("users").update(updates).eq("id", user_id).execute()
    except Exception:
        return


async def _ensure_public_profile(
    supabase: Any,
    *,
    user_id: str,
    tenant_id: str,
    full_name: str,
) -> None:
    try:
        await (
            supabase.table("profiles")
            .upsert(
                {"id": user_id, "tenant_id": tenant_id, "full_name": full_name},
                on_conflict="id",
            )
            .execute()
        )
    except Exception:
        return
