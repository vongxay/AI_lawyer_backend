"""
api/billing.py
==============
Minimal billing endpoints used by the user frontend.

These endpoints provide a safe development fallback while a real payment
provider integration is not wired yet.
"""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, status

from core.security import CurrentUser, require_roles

router = APIRouter(prefix="/api/v1/billing", tags=["billing"])
AuthUser = Annotated[CurrentUser, Depends(require_roles("client", "lawyer", "admin"))]


@router.get("/subscription", summary="Get current subscription")
async def get_subscription(user: AuthUser) -> dict:
    _ = user
    return {
        "plan": "free",
        "status": "active",
        "current_period_end": None,
        "cancel_at_period_end": False,
        "source": "development_fallback",
    }


@router.post("/subscribe", status_code=status.HTTP_501_NOT_IMPLEMENTED, summary="Create checkout session")
async def subscribe(payload: dict, user: AuthUser) -> dict:
    _ = user
    plan = payload.get("plan", "pro")
    return {
        "error": "BILLING_PROVIDER_NOT_CONFIGURED",
        "message": f"Checkout for plan '{plan}' is not configured yet.",
    }


@router.post("/cancel", summary="Cancel subscription")
async def cancel_subscription(user: AuthUser) -> dict:
    _ = user
    return {"success": True, "status": "noop", "message": "No paid subscription is active."}


@router.get("/portal", status_code=status.HTTP_501_NOT_IMPLEMENTED, summary="Get billing portal URL")
async def get_portal_url(user: AuthUser) -> dict:
    _ = user
    return {
        "error": "BILLING_PROVIDER_NOT_CONFIGURED",
        "message": "Billing portal is not configured yet.",
    }
