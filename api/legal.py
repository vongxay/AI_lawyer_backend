"""
api/legal.py
============
Core legal query endpoints.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Annotated, Any, AsyncIterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from api.dependencies import WorkflowDep
from api.schemas import (
    ChatMessageCreate,
    ChatMessageUpdate,
    ChatSessionCreate,
    ChatSessionUpdate,
    DraftRequest,
    LegalQueryRequest,
    LegalQueryResponse,
    VerifyCitationsRequest,
)
from agents.evidence_agent import EvidenceFile
from core.config import get_settings
from core.database import get_supabase
from core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from core.jurisdiction import infer_response_language
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
        session_id=payload.session_id,
        query_mode=payload.query_mode,
        response_style=payload.response_style,
        urgency=payload.urgency,
        model_id=payload.model_id,
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
    session_id: str | None = Form(default=None),
    jurisdiction: str | None = Form(default=None),
    model_id: str | None = Form(default=None),
    query_mode: str | None = Form(default=None),
    response_style: str | None = Form(default=None),
    urgency: str | None = Form(default=None),
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
        session_id=session_id,
        query_mode=query_mode or "evidence",
        response_style=response_style or "action_plan",
        urgency=urgency,
        model_id=model_id,
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
            response_language = infer_response_language(payload.question)
            yield _sse("meta", {"status": "started", "response_language": response_language})

            for msg in _stream_progress_messages(response_language):
                if await request.is_disconnected():
                    return
                yield _sse("token", {"token": msg + "\n"})

            result = await workflow.orchestrate(
                question=payload.question,
                case_id=payload.case_id,
                jurisdiction=payload.jurisdiction,
                user_id=user.sub,
                tenant_id=user.tenant_id,
                session_id=payload.session_id,
                query_mode=payload.query_mode,
                response_style=payload.response_style,
                urgency=payload.urgency,
                model_id=payload.model_id,
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
                    "query_type": response.get("query_type"),
                    "query_mode": response.get("query_mode"),
                    "response_style": response.get("response_style"),
                    "response_language": response.get("response_language"),
                    "session_id": response.get("session_id", result.session_id),
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


@router.get("/sessions", summary="List user chat sessions")
async def list_chat_sessions(
    user: AuthUser,
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    supabase = await _require_supabase()
    _require_tenant(user)
    query = (
        supabase.table("case_sessions")
        .select("*")
        .eq("tenant_id", user.tenant_id)
        .order("updated_at", desc=True)
        .limit(limit)
    )
    if user.role == "client":
        query = query.eq("user_id", user.sub)
    result = await query.execute()
    return [_normalise_session(row) for row in (result.data or [])]


@router.post("/sessions", summary="Create a user chat session")
async def create_chat_session(payload: ChatSessionCreate, user: AuthUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    _require_tenant(user)

    metadata = dict(payload.metadata or {})
    metadata.setdefault("source", "user_chat")
    query_type = payload.query_type or str(metadata.get("query_mode") or "general")
    title = payload.title or "Legal chat"

    insert_payload = {
        "tenant_id": user.tenant_id,
        "user_id": user.sub,
        "title": title,
        "legal_case_id": payload.legal_case_id,
        "query_type": query_type,
        "metadata": metadata,
    }
    row = await _insert_row(supabase, "case_sessions", insert_payload)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat session.",
        )
    return _normalise_session(row)


@router.patch("/sessions/{session_id}", summary="Update a user chat session")
async def update_chat_session(
    session_id: str,
    payload: ChatSessionUpdate,
    user: AuthUser,
) -> dict[str, Any]:
    supabase = await _require_supabase()
    await _get_chat_session(supabase, session_id, user)

    updates: dict[str, Any] = {"updated_at": _now()}
    if payload.title is not None:
        updates["title"] = payload.title
    if payload.status is not None:
        updates["status"] = payload.status
        if payload.status == "closed":
            updates["closed_at"] = _now()
    if payload.query_type is not None:
        updates["query_type"] = payload.query_type
    if payload.metadata is not None:
        updates["metadata"] = payload.metadata

    if len(updates) == 1:
        row = await _get_chat_session(supabase, session_id, user)
        return _normalise_session(row)

    row = await _update_row(supabase, "case_sessions", session_id, updates, tenant_id=user.tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")
    return _normalise_session(row)


@router.delete("/sessions/{session_id}", summary="Delete a user chat session")
async def delete_chat_session(session_id: str, user: AuthUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    session = await _get_chat_session(supabase, session_id, user)
    tenant_id = str(session.get("tenant_id") or user.tenant_id or "")
    if not tenant_id:
        _require_tenant(user)

    await _clear_session_reference(
        supabase,
        "audit_log",
        session_id=session_id,
        tenant_id=tenant_id,
        updates={"session_id": None, "message_id": None},
    )
    await _clear_session_reference(
        supabase,
        "documents",
        session_id=session_id,
        tenant_id=tenant_id,
        updates={"session_id": None, "updated_at": _now()},
    )
    await _clear_session_reference(
        supabase,
        "evidence",
        session_id=session_id,
        tenant_id=tenant_id,
        updates={"session_id": None, "updated_at": _now()},
    )

    for table in ("feedback", "citations_log", "expert_reviews", "query_analytics", "messages"):
        await _delete_session_rows(supabase, table, session_id=session_id, tenant_id=tenant_id)

    await _delete_session_by_id(supabase, session_id, tenant_id=tenant_id)
    return {"deleted": True, "session_id": session_id}


@router.get("/sessions/{session_id}/messages", summary="List chat messages for a session")
async def list_chat_messages(session_id: str, user: AuthUser) -> list[dict[str, Any]]:
    supabase = await _require_supabase()
    await _get_chat_session(supabase, session_id, user)
    result = await (
        supabase.table("messages")
        .select("*")
        .eq("session_id", session_id)
        .eq("tenant_id", user.tenant_id)
        .order("created_at", desc=False)
        .execute()
    )
    return [_normalise_message(row) for row in (result.data or [])]


@router.post("/sessions/{session_id}/messages", summary="Save a chat message")
async def save_chat_message(
    session_id: str,
    payload: ChatMessageCreate,
    user: AuthUser,
) -> dict[str, Any]:
    supabase = await _require_supabase()
    session = await _get_chat_session(supabase, session_id, user)

    insert_payload = _message_insert_payload(payload, session_id=session_id, tenant_id=user.tenant_id)
    row = await _insert_row(supabase, "messages", insert_payload)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save chat message.",
        )

    await _touch_session_after_message(supabase, session, row)
    return _normalise_message(row)


@router.patch("/sessions/{session_id}/messages/{message_id}", summary="Update a saved chat message")
async def update_chat_message(
    session_id: str,
    message_id: str,
    payload: ChatMessageUpdate,
    user: AuthUser,
) -> dict[str, Any]:
    supabase = await _require_supabase()
    await _get_chat_session(supabase, session_id, user)

    updates = _message_update_payload(payload)
    if not updates:
        row = await _get_message(supabase, session_id, message_id, user)
        return _normalise_message(row)

    row = await _update_message(supabase, session_id, message_id, user, updates)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat message not found.")
    return _normalise_message(row)


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
        max_tokens=settings.llm_max_tokens_draft,
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
            .select("id, case_no, court, year_be")
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


async def _require_supabase() -> Any:
    supabase = await get_supabase()
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database is not configured.",
        )
    return supabase


def _require_tenant(user: CurrentUser) -> None:
    if not user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User profile does not include a tenant.",
        )


async def _insert_row(supabase: Any, table: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        result = await supabase.table(table).insert(payload).execute()
        data = result.data or []
        if isinstance(data, list):
            return data[0] if data else None
        return data
    except Exception as exc:
        log.warning("legal.chat_insert.failed", table=table, error=str(exc))
        return None


async def _update_row(
    supabase: Any,
    table: str,
    row_id: str,
    updates: dict[str, Any],
    *,
    tenant_id: str,
) -> dict[str, Any] | None:
    try:
        result = await (
            supabase.table(table)
            .update(updates)
            .eq("id", row_id)
            .eq("tenant_id", tenant_id)
            .execute()
        )
        data = result.data or []
        if isinstance(data, list):
            return data[0] if data else None
        return data
    except Exception as exc:
        log.warning("legal.chat_update.failed", table=table, id=row_id, error=str(exc))
        return None


async def _clear_session_reference(
    supabase: Any,
    table: str,
    *,
    session_id: str,
    tenant_id: str,
    updates: dict[str, Any],
) -> None:
    try:
        await (
            supabase.table(table)
            .update(updates)
            .eq("session_id", session_id)
            .eq("tenant_id", tenant_id)
            .execute()
        )
    except Exception as exc:
        log.warning(
            "legal.chat_session_reference_clear.failed",
            table=table,
            session_id=session_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat history.",
        ) from exc


async def _delete_session_rows(
    supabase: Any,
    table: str,
    *,
    session_id: str,
    tenant_id: str,
) -> None:
    try:
        await (
            supabase.table(table)
            .delete()
            .eq("session_id", session_id)
            .eq("tenant_id", tenant_id)
            .execute()
        )
    except Exception as exc:
        log.warning(
            "legal.chat_session_rows_delete.failed",
            table=table,
            session_id=session_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat history.",
        ) from exc


async def _delete_session_by_id(supabase: Any, session_id: str, *, tenant_id: str) -> None:
    try:
        await (
            supabase.table("case_sessions")
            .delete()
            .eq("id", session_id)
            .eq("tenant_id", tenant_id)
            .execute()
        )
    except Exception as exc:
        log.warning("legal.chat_session_delete.failed", session_id=session_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat history.",
        ) from exc


async def _get_chat_session(supabase: Any, session_id: str, user: CurrentUser) -> dict[str, Any]:
    _require_tenant(user)
    query = (
        supabase.table("case_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("tenant_id", user.tenant_id)
        .limit(1)
    )
    if user.role == "client":
        query = query.eq("user_id", user.sub)
    result = await query.execute()
    data = result.data or []
    row = data[0] if isinstance(data, list) and data else data
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found.")
    return row


async def _get_message(
    supabase: Any,
    session_id: str,
    message_id: str,
    user: CurrentUser,
) -> dict[str, Any]:
    result = await (
        supabase.table("messages")
        .select("*")
        .eq("id", message_id)
        .eq("session_id", session_id)
        .eq("tenant_id", user.tenant_id)
        .limit(1)
        .execute()
    )
    data = result.data or []
    row = data[0] if isinstance(data, list) and data else data
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat message not found.")
    return row


async def _update_message(
    supabase: Any,
    session_id: str,
    message_id: str,
    user: CurrentUser,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        result = await (
            supabase.table("messages")
            .update(updates)
            .eq("id", message_id)
            .eq("session_id", session_id)
            .eq("tenant_id", user.tenant_id)
            .execute()
        )
        data = result.data or []
        if isinstance(data, list):
            return data[0] if data else None
        return data
    except Exception as exc:
        log.warning("legal.chat_message_update.failed", message_id=message_id, error=str(exc))
        return None


def _message_insert_payload(
    payload: ChatMessageCreate,
    *,
    session_id: str,
    tenant_id: str,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "tenant_id": tenant_id,
        "role": _normalise_db_role(payload.role),
        "content": payload.content,
        "irac_output": payload.irac_output,
        "citations": payload.citations,
        "confidence": payload.confidence,
        "agents_used": _normalise_agent_names(payload.agents_used),
        "model_used": payload.model_used,
        "latency_ms": payload.latency_ms,
        "escalated": payload.escalated,
        "escalation_reason": payload.escalation_reason,
    }


def _message_update_payload(payload: ChatMessageUpdate) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if payload.content is not None:
        updates["content"] = payload.content
    if payload.irac_output is not None:
        updates["irac_output"] = payload.irac_output
    if payload.citations is not None:
        updates["citations"] = payload.citations
    if payload.confidence is not None:
        updates["confidence"] = payload.confidence
    if payload.agents_used is not None:
        updates["agents_used"] = _normalise_agent_names(payload.agents_used)
    if payload.model_used is not None:
        updates["model_used"] = payload.model_used
    if payload.latency_ms is not None:
        updates["latency_ms"] = payload.latency_ms
    if payload.escalated is not None:
        updates["escalated"] = payload.escalated
    if payload.escalation_reason is not None:
        updates["escalation_reason"] = payload.escalation_reason
    return updates


async def _touch_session_after_message(
    supabase: Any,
    session: dict[str, Any],
    message: dict[str, Any],
) -> None:
    session_id = str(session.get("id") or "")
    tenant_id = str(session.get("tenant_id") or "")
    if not session_id or not tenant_id:
        return

    agents = _union_agent_names(session.get("agents_used"), message.get("agents_used"))
    updates: dict[str, Any] = {
        "message_count": int(session.get("message_count") or 0) + 1,
        "updated_at": _now(),
    }
    if agents:
        updates["agents_used"] = agents
    if message.get("role") == "assistant":
        updates["last_summary"] = str(message.get("content") or "")[:500]
        if message.get("escalated"):
            updates["status"] = "escalated"

    await _update_row(supabase, "case_sessions", session_id, updates, tenant_id=tenant_id)


def _normalise_session(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "title": row.get("title"),
        "status": row.get("status"),
        "legal_case_id": row.get("legal_case_id"),
        "query_type": row.get("query_type"),
        "message_count": row.get("message_count") or 0,
        "agents_used": row.get("agents_used") or [],
        "metadata": row.get("metadata") or {},
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "closed_at": row.get("closed_at"),
    }


def _normalise_message(row: dict[str, Any]) -> dict[str, Any]:
    role = row.get("role")
    return {
        "id": row.get("id"),
        "session_id": row.get("session_id"),
        "role": "ai" if role == "assistant" else role,
        "content": row.get("content") or "",
        "irac_output": row.get("irac_output"),
        "citations": row.get("citations") or [],
        "confidence": row.get("confidence"),
        "agents_used": row.get("agents_used") or [],
        "model_used": row.get("model_used"),
        "latency_ms": row.get("latency_ms"),
        "escalated": row.get("escalated") or False,
        "escalation_reason": row.get("escalation_reason"),
        "created_at": row.get("created_at"),
    }


def _normalise_db_role(role: str) -> str:
    return "assistant" if role == "ai" else role


def _normalise_agent_names(values: list[str] | None) -> list[str]:
    mapping = {
        "research": "legal_research",
        "legal_research": "legal_research",
        "reasoning": "irac_reasoning",
        "irac": "irac_reasoning",
        "irac_reasoning": "irac_reasoning",
        "verification": "citation_verification",
        "citation": "citation_verification",
        "citation_verification": "citation_verification",
        "document": "document_analysis",
        "document_analysis": "document_analysis",
        "evidence": "evidence_analyzer",
        "evidence_analyzer": "evidence_analyzer",
        "risk": "risk_strategy",
        "risk_strategy": "risk_strategy",
        "classifier": "query_classifier",
        "query_classifier": "query_classifier",
    }
    normalised: list[str] = []
    for value in values or []:
        key = str(value).strip().lower()
        mapped = mapping.get(key)
        if mapped and mapped not in normalised:
            normalised.append(mapped)
    return normalised


def _union_agent_names(*groups: Any) -> list[str]:
    merged: list[str] = []
    for group in groups:
        if not isinstance(group, list):
            continue
        for value in _normalise_agent_names([str(item) for item in group]):
            if value not in merged:
                merged.append(value)
    return merged


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stream_progress_messages(response_language: str) -> tuple[str, str, str]:
    if response_language == "lo":
        return (
            "ກຳລັງຄົ້ນຫາແຫຼ່ງກົດໝາຍລາວ...",
            "ກຳລັງສ້າງການວິເຄາະທາງກົດໝາຍ...",
            "ກຳລັງກວດສອບ citations ແລະຄວາມໝັ້ນໃຈ...",
        )
    if response_language == "th":
        return (
            "กำลังค้นหาแหล่งกฎหมายลาว...",
            "กำลังสร้างการวิเคราะห์ทางกฎหมาย...",
            "กำลังตรวจสอบ citations และความมั่นใจ...",
        )
    return (
        "Searching Lao legal sources...",
        "Building legal analysis...",
        "Verifying citations and confidence...",
    )


def _sse(event: str, data: dict | list) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
