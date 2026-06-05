"""
api/auth.py
===========
Authentication bridge for the React admin app.
"""
from __future__ import annotations

from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.users import _fetch_user_profile, _normalise_profile
from core.config import get_settings
from core.database import get_supabase
from core.security import CurrentUser, get_admin_user

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
AdminUser = Annotated[CurrentUser, Depends(get_admin_user)]


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1)


@router.post("/login", summary="Sign in an admin through Supabase Auth")
async def login(payload: LoginRequest) -> dict[str, Any]:
    settings = get_settings()
    api_key = settings.supabase_anon_key or settings.supabase_key
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
async def logout(user: AdminUser) -> dict[str, str]:
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
