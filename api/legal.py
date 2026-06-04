"""
api/legal.py
============
Core legal query endpoints.
"""
from __future__ import annotations

import asyncio
import json
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

from api.dependencies import WorkflowDep
from api.schemas import (
    DraftRequest,
    LegalQueryRequest,
    LegalQueryResponse,
    VerifyCitationsRequest,
)
from agents.evidence_agent import EvidenceFile
from core.config import get_settings
from core.database import get_supabase
from core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from core.logging import get_logger
from core.security import CurrentUser, require_roles
from services.llm_service import LlmService, Message

router = APIRouter(prefix="/api/v1/legal", tags=["legal"])
log = get_logger(__name__)

AuthUser = Annotated[CurrentUser, Depends(require_roles("client", "lawyer", "admin"))]


@router.post("/query", response_model=LegalQueryResponse, summary="Legal query - full IRAC response")
async def legal_query(
    payload: LegalQueryRequest,
    workflow: WorkflowDep,
    user: AuthUser,
) -> dict:
    result = await workflow.orchestrate(
        question=payload.question,
        case_id=payload.case_id,
        jurisdiction=payload.jurisdiction,
        user_id=user.sub,
        tenant_id=user.tenant_id,
    )
    return result.response


@router.post(
    "/query/with-files",
    response_model=LegalQueryResponse,
    summary="Legal query with uploaded evidence files",
)
async def legal_query_with_files(
    workflow: WorkflowDep,
    user: AuthUser,
    question: str = Form(..., min_length=3, max_length=5000),
    case_id: str | None = Form(default=None),
    jurisdiction: str | None = Form(default=None),
    files: list[UploadFile] = File(default_factory=list),
) -> dict:
    settings = get_settings()
    evidence_files: list[EvidenceFile] = []

    for upload in files:
        if upload.content_type not in settings.allowed_mime_types:
            raise UnsupportedFileTypeError(
                f"'{upload.filename}' has unsupported type '{upload.content_type}'",
                details={"allowed": sorted(settings.allowed_mime_types)},
            )

        content = await upload.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            raise FileTooLargeError(f"'{upload.filename}' ({size_mb:.1f}MB) exceeds limit")

        evidence_files.append(
            EvidenceFile(
                filename=upload.filename or "unnamed",
                content_type=upload.content_type or "application/octet-stream",
                content=content,
            )
        )

    result = await workflow.orchestrate(
        question=question.strip(),
        case_id=case_id,
        jurisdiction=jurisdiction,
        evidence_files=evidence_files or None,
        user_id=user.sub,
        tenant_id=user.tenant_id,
    )
    return result.response


@router.post(
    "/query/stream",
    summary="Legal query - SSE structured stream",
    response_class=StreamingResponse,
)
async def legal_query_stream(
    request: Request,
    payload: LegalQueryRequest,
    workflow: WorkflowDep,
    user: AuthUser,
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        try:
            yield _sse("meta", {"status": "started"})

            for msg in (
                "กำลังค้นข้อมูลกฎหมาย...",
                "กำลังวิเคราะห์ตาม IRAC...",
                "กำลังตรวจสอบ citation...",
            ):
                if await request.is_disconnected():
                    return
                yield _sse("token", {"token": msg + "\n"})

            result = await workflow.orchestrate(
                question=payload.question,
                case_id=payload.case_id,
                jurisdiction=payload.jurisdiction,
                user_id=user.sub,
                tenant_id=user.tenant_id,
            )
            response = result.response

            if payload.include_irac:
                yield _sse("irac", response.get("irac", {}))
            if payload.include_citations:
                yield _sse("citations", response.get("citations", []))

            yield _sse("confidence", {"score": response.get("confidence", result.confidence)})
            yield _sse(
                "meta",
                {
                    "agents_used": response.get("agents_used", result.agents_used),
                    "processing_time_ms": response.get("processing_time_ms", result.processing_time_ms),
                    "escalated_to_expert": response.get("escalated_to_expert", result.escalated_to_expert),
                },
            )
            yield _sse("done", {"ok": True})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("legal.stream.failed", error=str(exc))
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/draft", summary="Draft a legal document")
async def draft_document(
    payload: DraftRequest,
    user: AuthUser,
) -> dict:
    settings = get_settings()
    llm = LlmService()
    language_name = {"LA": "Lao", "TH": "Thai", "EN": "English"}.get(payload.language, payload.language)
    system_prompt = (
        "You are a careful legal drafting assistant. Draft only from the user's facts. "
        "Do not invent missing facts, parties, dates, citations, or legal authority. "
        "When information is missing, add clearly marked placeholders. "
        "Use a professional legal document structure and include a review warning."
    )
    user_prompt = (
        f"Document type: {payload.document_type or 'legal document'}\n"
        f"Jurisdiction: {payload.jurisdiction or 'unspecified'}\n"
        f"Language: {language_name}\n\n"
        f"User facts and instructions:\n{payload.prompt}"
    )
    result = await llm.generate(
        model=settings.model_document,
        system=system_prompt,
        messages=[Message(role="user", content=user_prompt)],
        max_tokens=2500,
        temperature=0.1,
    )
    content = result.text.strip()
    return {
        "content": content,
        "document": content,
        "format": "markdown",
        "language": payload.language,
        "jurisdiction": payload.jurisdiction,
        "model": result.model,
        "provider": result.provider,
        "tokens": result.total_tokens,
        "disclaimer": "Review this draft with a qualified lawyer before use.",
    }


@router.post("/citations/verify", summary="Verify a list of legal citations")
async def verify_citations(
    payload: VerifyCitationsRequest,
    workflow: WorkflowDep,
    user: AuthUser,
) -> dict:
    _ = user
    result = await workflow._verification_agent.run(
        citations=[c.model_dump() for c in payload.citations]
    )
    data = result.data
    return {
        **data,
        "verified": data.get("citations_verified", False),
        "results": data.get("citations", []),
    }


@router.get("/graph/{case_no}", summary="Get precedent chain for a case")
async def precedent_graph(case_no: str, user: AuthUser) -> dict:
    _ = user
    supabase = await get_supabase()
    if not supabase:
        return {"case_no": case_no, "nodes": [], "edges": [], "note": "Database not configured"}

    try:
        result = await (
            supabase.table("cases")
            .select("id, case_no, court, year")
            .eq("case_no", case_no)
            .single()
            .execute()
        )

        if not result.data:
            return {"case_no": case_no, "nodes": [], "edges": []}

        case_id = result.data["id"]
        chain = await supabase.rpc(
            "get_precedent_chain",
            {"start_case_id": case_id, "max_depth": 3},
        ).execute()

        return {
            "case_no": case_no,
            "nodes": result.data,
            "edges": chain.data or [],
        }
    except Exception as exc:
        log.warning("precedent_graph.failed", case_no=case_no, error=str(exc))
        return {"case_no": case_no, "nodes": [], "edges": []}


def _sse(event: str, data: dict | list) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
