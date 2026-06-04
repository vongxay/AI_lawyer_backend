"""
api/dashboard.py
================
User dashboard endpoints for the client-facing app.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from core.database import get_supabase
from core.security import CurrentUser, require_roles

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])
AuthUser = Annotated[CurrentUser, Depends(require_roles("client", "lawyer", "admin"))]


@router.get("/stats", summary="Get user dashboard statistics")
async def get_user_dashboard_stats(user: AuthUser) -> dict[str, Any]:
    supabase = await get_supabase()
    if not supabase or not user.tenant_id:
        return _empty_stats()

    cases = await _select_rows(
        supabase,
        "legal_cases",
        user=user,
        user_column="client_id",
        columns="id, status",
    )
    sessions = await _select_rows(
        supabase,
        "case_sessions",
        user=user,
        user_column="user_id",
        columns="id, legal_case_id",
    )
    messages = await _select_rows(
        supabase,
        "messages",
        user=user,
        columns="id, role, confidence, session_id",
    )
    evidence = await _select_rows(
        supabase,
        "evidence",
        user=user,
        columns="id, legal_case_id, session_id",
    )

    case_ids = {str(row.get("id")) for row in cases if row.get("id")}
    session_ids = {str(row.get("id")) for row in sessions if row.get("id")}

    if user.role == "client":
        messages = [
            row for row in messages
            if str(row.get("session_id")) in session_ids
        ]
        evidence = [
            row for row in evidence
            if str(row.get("legal_case_id")) in case_ids
            or str(row.get("session_id")) in session_ids
        ]

    case_statuses = [_case_status(row.get("status")) for row in cases]
    assistant_confidences = [
        _to_float(row.get("confidence"))
        for row in messages
        if row.get("role") == "assistant" and row.get("confidence") is not None
    ]

    return {
        "total_cases": len(cases),
        "active_cases": case_statuses.count("active"),
        "closed_cases": case_statuses.count("closed"),
        "settled_cases": case_statuses.count("settled"),
        "total_queries": len([row for row in messages if row.get("role") == "user"]),
        "avg_confidence": _avg(assistant_confidences),
        "total_evidence": len(evidence),
        "total_sessions": len(sessions),
    }


async def _select_rows(
    supabase: Any,
    table: str,
    *,
    user: CurrentUser,
    columns: str,
    user_column: str | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    query = supabase.table(table).select(columns).eq("tenant_id", user.tenant_id).limit(limit)
    if user.role == "client" and user_column:
        query = query.eq(user_column, user.sub)

    try:
        result = await query.execute()
        return list(result.data or [])
    except Exception:
        return []


def _case_status(value: Any) -> str:
    normalized = str(value or "active").lower()
    if normalized in {"open", "active"}:
        return "active"
    if normalized in {"settled", "resolved"}:
        return "settled"
    if normalized in {"closed", "archived", "dismissed"}:
        return "closed"
    return "active"


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _to_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _empty_stats() -> dict[str, Any]:
    return {
        "total_cases": 0,
        "active_cases": 0,
        "closed_cases": 0,
        "settled_cases": 0,
        "total_queries": 0,
        "avg_confidence": 0.0,
        "total_evidence": 0,
        "total_sessions": 0,
    }
