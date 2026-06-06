"""
api/memory.py
=============
Case memory read endpoints.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import CaseCreateRequest, CaseMemoryResponse, CaseRecordResponse, TimelineEntry
from core.database import get_redis, get_supabase
from core.security import CurrentUser, require_roles
from memory.case_memory import CaseMemoryService

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])
AuthUser = Annotated[CurrentUser, Depends(require_roles("client", "lawyer", "admin"))]


async def _get_memory_service() -> CaseMemoryService:
    supabase = await get_supabase()
    redis = await get_redis()
    return CaseMemoryService(supabase=supabase, redis=redis)


@router.get("/cases", response_model=list[CaseRecordResponse], summary="List user legal cases")
async def list_cases(user: AuthUser) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase or not user.tenant_id:
        return []

    query = (
        supabase.table("legal_cases")
        .select("id, title, case_type, status, created_at, updated_at")
        .eq("tenant_id", user.tenant_id)
        .order("updated_at", desc=True)
        .limit(100)
    )
    if user.role == "client":
        query = query.eq("client_id", user.sub)

    try:
        result = await query.execute()
    except Exception:
        return []

    return [_normalise_case(row) for row in (result.data or [])]


@router.post("/cases", response_model=CaseRecordResponse, summary="Create a legal case")
async def create_case(payload: CaseCreateRequest, user: AuthUser) -> dict[str, Any]:
    supabase = await get_supabase()
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database is not configured.",
        )
    if not user.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User tenant was not found.")

    title = payload.title or f"Legal case {datetime.now(UTC).strftime('%Y-%m-%d')}"
    insert_payload = {
        "tenant_id": user.tenant_id,
        "client_id": user.sub,
        "title": title,
        "case_type": payload.case_type or "general",
        "description": payload.description,
        "jurisdiction": _db_jurisdiction(payload.jurisdiction),
    }

    try:
        result = (
            await supabase.table("legal_cases")
            .insert(insert_payload)
            .select("id, title, case_type, status, created_at, updated_at")
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    rows = result.data or []
    row = rows[0] if isinstance(rows, list) and rows else rows
    if not row:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create case.")
    return _normalise_case(row)


@router.get("/case/{case_id}", response_model=CaseMemoryResponse, summary="Get case memory summary")
async def get_case_memory(
    case_id: str,
    user: AuthUser,
) -> dict:
    svc = await _get_memory_service()
    data = await svc.get(case_id, tenant_id=user.tenant_id)
    sessions = await _load_case_sessions(case_id, user)
    history = _normalise_irac_history(data.get("irac_history"))
    timeline = _normalise_timeline(data.get("timeline"), sessions, history)
    legal_issues = _normalise_string_list(data.get("legal_issues"))
    key_facts = _normalise_key_facts(data.get("facts_summary"))

    return {
        "case_id": case_id,
        "summary": data.get("facts_summary") or "",
        "key_facts": key_facts,
        "legal_issues": legal_issues,
        "irac_history": history,
        "timeline": timeline,
        "facts_summary": data.get("facts_summary"),
        "jurisdiction": data.get("jurisdiction", "laos"),
        "status": data.get("status", "active"),
        "irac_count": len(history),
        "key_citations_count": len(data.get("key_citations", [])),
    }


@router.get(
    "/case/{case_id}/timeline",
    response_model=list[TimelineEntry],
    summary="Get chronological IRAC history for a case",
)
async def get_case_timeline(case_id: str, user: AuthUser) -> list[dict]:
    svc = await _get_memory_service()
    data = await svc.get(case_id, tenant_id=user.tenant_id)
    sessions = await _load_case_sessions(case_id, user)
    return _normalise_timeline(data.get("timeline"), sessions, _normalise_irac_history(data.get("irac_history")))


async def _load_case_sessions(case_id: str, user: CurrentUser) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase or not user.tenant_id:
        return []

    query = (
        supabase.table("case_sessions")
        .select("id, title, query_type, created_at, updated_at")
        .eq("tenant_id", user.tenant_id)
        .eq("legal_case_id", case_id)
        .order("updated_at", desc=True)
        .limit(20)
    )
    if user.role == "client":
        query = query.eq("user_id", user.sub)

    try:
        result = await query.execute()
        return list(result.data or [])
    except Exception:
        return []


def _normalise_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "title": str(row.get("title") or "Legal case"),
        "type": str(row.get("case_type") or row.get("type") or "general"),
        "status": _case_status(row.get("status")),
        "created_at": _iso(row.get("created_at")),
        "last_accessed": _iso(row.get("updated_at") or row.get("created_at")),
    }


def _case_status(value: Any) -> str:
    normalized = str(value or "active").lower()
    if normalized in {"open", "active"}:
        return "active"
    if normalized in {"closed", "archived", "dismissed"}:
        return "closed"
    if normalized in {"settled", "resolved"}:
        return "settled"
    return "active"


def _db_jurisdiction(value: str | None) -> str:
    normalized = str(value or "laos").strip().lower().replace("_", "-")
    if normalized in {"laos", "lao", "la", "lao-pdr"}:
        return "laos"
    if normalized in {"thailand", "thai", "th"}:
        return "thailand"
    return normalized or "laos"


def _normalise_irac_history(value: Any) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for index, item in enumerate(value if isinstance(value, list) else []):
        if not isinstance(item, dict):
            continue

        conclusion = item.get("conclusion")
        if not isinstance(conclusion, dict):
            conclusion = item.get("irac_conclusion") if isinstance(item.get("irac_conclusion"), dict) else {}

        recommendation = (
            conclusion.get("recommendation")
            or item.get("conclusion")
            or item.get("recommendation")
            or ""
        )
        history.append({
            "id": str(item.get("id") or item.get("session_id") or f"irac-{index}"),
            "query": str(item.get("query") or item.get("question") or ""),
            "conclusion": str(recommendation),
            "confidence": _to_float(item.get("confidence")),
            "created_at": _event_date(item.get("created_at") or item.get("date") or item.get("ts")),
        })
    return history


def _normalise_timeline(
    timeline_value: Any,
    sessions: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, item in enumerate(timeline_value if isinstance(timeline_value, list) else []):
        if not isinstance(item, dict):
            continue
        events.append({
            "id": str(item.get("id") or f"timeline-{index}"),
            "event": str(item.get("event") or item.get("title") or item.get("question") or "Case event"),
            "date": _event_date(item.get("date") or item.get("created_at") or item.get("ts")),
            "type": _timeline_type(item.get("type")),
        })

    for session in sessions:
        query_type = str(session.get("query_type") or "query")
        events.append({
            "id": str(session.get("id") or f"session-{len(events)}"),
            "event": str(session.get("title") or "Legal chat"),
            "date": _event_date(session.get("updated_at") or session.get("created_at")),
            "type": _timeline_type(query_type),
        })

    for item in history:
        if item["id"] in {event["id"] for event in events}:
            continue
        events.append({
            "id": item["id"],
            "event": item["query"] or "IRAC analysis",
            "date": item["created_at"],
            "type": "query",
        })

    return events


def _timeline_type(value: Any) -> str:
    normalized = str(value or "query").lower()
    if "evidence" in normalized:
        return "evidence"
    if "document" in normalized or "draft" in normalized:
        return "document"
    if normalized in {"milestone", "deadline", "hearing"}:
        return "milestone"
    return "query"


def _normalise_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalise_key_facts(summary: Any) -> list[str]:
    text = str(summary or "").strip()
    if not text:
        return []
    parts = [part.strip(" -") for part in text.replace("\r", "\n").split("\n") if part.strip(" -")]
    return parts[:5] if len(parts) > 1 else [text]


def _event_date(value: Any) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    return _iso(value) or datetime.now(UTC).isoformat()


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _to_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
