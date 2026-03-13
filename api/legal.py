"""
api/legal.py
============
Core legal query endpoints.

Routes:
    POST /api/v1/legal/query           — Full IRAC response
    POST /api/v1/legal/query/stream    — SSE streaming response
    POST /api/v1/legal/draft           — Draft legal document
    POST /api/v1/legal/citations/verify — Verify citation list
    GET  /api/v1/legal/graph/{case_no} — Precedent graph for a case
"""
from __future__ import annotations

import json
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.api.dependencies import WorkflowDep
from backend.api.schemas import (
    DraftRequest,
    LegalQueryRequest,
    LegalQueryResponse,
    VerifyCitationsRequest,
)
from backend.core.database import get_supabase
from backend.core.security import CurrentUser, get_optional_user, require_roles

router = APIRouter(prefix="/api/v1/legal", tags=["legal"])

OptionalUser = Annotated[CurrentUser | None, Depends(get_optional_user)]
AuthUser = Annotated[CurrentUser, Depends(require_roles("client", "lawyer", "admin"))]


@router.post("/query", response_model=LegalQueryResponse, summary="Legal query — full IRAC response")
async def legal_query(
    payload: LegalQueryRequest,
    workflow: WorkflowDep,
    user: OptionalUser,
) -> dict:
    result = await workflow.orchestrate(
        question=payload.question,
        case_id=payload.case_id,
        jurisdiction=payload.jurisdiction,
        user_id=user.sub if user else None,
        tenant_id=user.tenant_id if user else None,
    )
    return result.response


@router.post(
    "/query/stream",
    summary="Legal query — SSE streaming (tokens as they arrive)",
    response_class=StreamingResponse,
)
async def legal_query_stream(
    payload: LegalQueryRequest,
    workflow: WorkflowDep,
    user: OptionalUser,
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        # Phase 1: Status tokens (keep client alive during processing)
        for msg in ["กำลังค้นข้อมูลกฎหมาย...", "กำลังวิเคราะห์ตาม IRAC...", "กำลังตรวจสอบ citation..."]:
            yield _sse({"type": "status", "message": msg})

        # Phase 2: Full result
        result = await workflow.orchestrate(
            question=payload.question,
            case_id=payload.case_id,
            jurisdiction=payload.jurisdiction,
            user_id=user.sub if user else None,
            tenant_id=user.tenant_id if user else None,
        )
        yield _sse({"type": "result", "data": result.response})
        yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/draft", summary="Draft a legal document")
async def draft_document(
    payload: DraftRequest,
    user: OptionalUser,
) -> dict:
    # Future: invoke DocumentAgent in draft mode
    return {
        "content": f"[DRAFT — {payload.document_type or 'document'}]\n\n{payload.prompt}",
        "language": payload.language,
        "jurisdiction": payload.jurisdiction,
        "disclaimer": "This is a draft template. Review with a qualified lawyer before use.",
    }


@router.post("/citations/verify", summary="Verify a list of legal citations")
async def verify_citations(
    payload: VerifyCitationsRequest,
    workflow: WorkflowDep,
) -> dict:
    result = await workflow._verification_agent.run(
        citations=[c.model_dump() for c in payload.citations]
    )
    return result.data


@router.get("/graph/{case_no}", summary="Get precedent chain for a case")
async def precedent_graph(case_no: str) -> dict:
    supabase = await get_supabase()
    if not supabase:
        return {"case_no": case_no, "nodes": [], "edges": [], "note": "Database not configured"}

    try:
        result = await supabase.table("cases") \
            .select("id, case_no, court, year") \
            .eq("case_no", case_no) \
            .single() \
            .execute()

        if not result.data:
            return {"case_no": case_no, "nodes": [], "edges": []}

        case_id = result.data["id"]
        chain = await supabase.rpc("get_precedent_chain", {
            "start_case_id": case_id, "max_depth": 3
        }).execute()

        return {
            "case_no": case_no,
            "nodes": result.data,
            "edges": chain.data or [],
        }
    except Exception:
        return {"case_no": case_no, "nodes": [], "edges": []}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
