"""
api/memory.py
=============
Case memory read endpoints.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from backend.api.schemas import CaseMemoryResponse, TimelineEntry
from backend.core.database import get_redis, get_supabase
from backend.core.security import CurrentUser, require_roles
from backend.memory.case_memory import CaseMemoryService

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])
AuthUser = Annotated[CurrentUser, Depends(require_roles("client", "lawyer", "admin"))]


async def _get_memory_service() -> CaseMemoryService:
    supabase = await get_supabase()
    redis = await get_redis()
    return CaseMemoryService(supabase=supabase, redis=redis)


@router.get("/case/{case_id}", response_model=CaseMemoryResponse, summary="Get case memory summary")
async def get_case_memory(
    case_id: str,
    user: AuthUser,
) -> dict:
    svc = await _get_memory_service()
    data = await svc.get(case_id)
    return {
        "case_id": case_id,
        "facts_summary": data.get("facts_summary"),
        "jurisdiction": data.get("jurisdiction", "TH"),
        "status": data.get("status", "active"),
        "irac_count": len(data.get("irac_history", [])),
        "key_citations_count": len(data.get("key_citations", [])),
    }


@router.get(
    "/case/{case_id}/timeline",
    response_model=list[TimelineEntry],
    summary="Get chronological IRAC history for a case",
)
async def get_case_timeline(case_id: str, user: AuthUser) -> list[dict]:
    svc = await _get_memory_service()
    data = await svc.get(case_id)
    history = data.get("irac_history", [])
    return [{"ts": h.get("ts", 0), "question": h.get("question")} for h in history]
