"""
api/users.py
============
User profile endpoints consumed by the React admin shell.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from core.database import get_supabase
from core.logging import get_logger
from core.security import CurrentUser, get_admin_user

router = APIRouter(prefix="/api/v1/users", tags=["users"])
log = get_logger(__name__)
AdminUser = Annotated[CurrentUser, Depends(get_admin_user)]


@router.get("/{user_id}/profile", summary="Fetch an admin user's profile")
async def get_user_profile(user_id: str, user: AdminUser) -> dict[str, Any]:
    supabase = await get_supabase()
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database is not configured.",
        )

    profile = await _fetch_user_profile(supabase, user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found.",
        )

    return _normalise_profile(profile)


async def _fetch_user_profile(supabase: Any, user_id: str) -> dict[str, Any] | None:
    selects = [
        "id, email, full_name, role, tenant_id, is_active, created_at, updated_at",
        "id, email, role, tenant_id, created_at",
        "*",
    ]

    for columns in selects:
        try:
            result = await (
                supabase.table("users")
                .select(columns)
                .eq("id", user_id)
                .limit(1)
                .execute()
            )
            data = result.data or []
            if isinstance(data, list):
                return data[0] if data else None
            return data
        except Exception as exc:
            log.warning("users.profile_select.failed", columns=columns, error=str(exc))

    return None


def _normalise_profile(row: dict[str, Any]) -> dict[str, Any]:
    email = row.get("email") or ""
    full_name = row.get("full_name") or row.get("name") or email or "Admin"
    return {
        "id": str(row.get("id") or ""),
        "email": email,
        "full_name": full_name,
        "role": row.get("role") or "client",
        "tenant_id": str(row.get("tenant_id") or ""),
        "is_active": row.get("is_active") is not False,
    }
