from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from backend.api.schemas import DraftRequest, LegalQueryRequest, LegalQueryResponse, VerifyCitationsRequest
from backend.orchestrator.workflow_manager import WorkflowManager


router = APIRouter(prefix="/api/v1/legal", tags=["legal"])
_workflow = WorkflowManager()


@router.post("/query", response_model=LegalQueryResponse)
async def legal_query(payload: LegalQueryRequest) -> dict:
    result = await _workflow.orchestrate(question=payload.question, case_id=payload.case_id)
    return result.response


def _sse_event(data: str) -> str:
    return f"data: {data}\n\n"


@router.post("/query/stream")
async def legal_query_stream_post(payload: LegalQueryRequest) -> StreamingResponse:
    return await _legal_query_stream_impl(q=payload.question, case_id=payload.case_id)


@router.get("/query/stream")
async def legal_query_stream_get(q: str = Query(min_length=1), case_id: str | None = None) -> StreamingResponse:
    return await _legal_query_stream_impl(q=q, case_id=case_id)


async def _legal_query_stream_impl(*, q: str, case_id: str | None) -> StreamingResponse:
    async def gen() -> AsyncIterator[str]:
        # Stream "thinking" tokens first (stub), then send final JSON as one event, then [DONE]
        for chunk in ["กำลังค้นข้อมูล...", "กำลังวิเคราะห์ตาม IRAC...", "กำลังตรวจสอบ citation..."]:
            yield _sse_event(chunk)
        result = await _workflow.orchestrate(question=q, case_id=case_id)
        yield _sse_event(json.dumps(result.response, ensure_ascii=False))
        yield _sse_event("[DONE]")

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/draft")
async def draft_document(payload: DraftRequest) -> dict:
    _ = payload.jurisdiction
    return {"content": f"STUB DRAFT:\n\n{payload.prompt}", "format": "text"}


@router.post("/citations/verify")
async def verify_citations(payload: VerifyCitationsRequest) -> dict:
    # Accept list of citations and run verifier agent stub
    verification = await _workflow.verification_agent.verify([c.model_dump() for c in payload.citations])
    return verification


@router.get("/graph/{case_no}")
async def precedent_graph(case_no: str) -> dict:
    return {"case_no": case_no, "nodes": [], "edges": []}

