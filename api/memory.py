from __future__ import annotations

from fastapi import APIRouter

from backend.memory.case_memory import CaseMemoryService


router = APIRouter(prefix="/api/v1/memory", tags=["memory"])
_mem = CaseMemoryService()


@router.get("/case/{case_id}")
async def get_case_memory(case_id: str) -> dict:
    data = await _mem.get(case_id)
    return {
        "case_id": case_id,
        "facts_summary": data.get("facts_summary"),
        "jurisdiction": data.get("jurisdiction", "TH"),
        "status": data.get("status", "active"),
    }


@router.get("/case/{case_id}/timeline")
async def get_case_timeline(case_id: str) -> list[dict]:
    data = await _mem.get(case_id)
    history = data.get("irac_history", [])
    return [{"ts": h.get("ts"), "question": h.get("question")} for h in history]

