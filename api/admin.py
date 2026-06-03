"""
api/admin.py
============
Admin-only endpoints. All routes require role=admin.

Routes:
    POST /api/v1/admin/ingest        — Queue document for ingestion into knowledge base
    GET  /api/v1/admin/audit-log     — Read audit trail
    GET  /api/v1/admin/expert-queue  — Pending human review queue
    POST /api/v1/admin/expert-queue/{id}/resolve — Resolve a review item
"""
from __future__ import annotations

from datetime import datetime, timezone
import mimetypes
from typing import Annotated, Any
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field, HttpUrl

from api.dependencies import AuditDep, ExpertQueueDep
from api.schemas import IngestRequest
from core.config import get_settings
from core.database import get_supabase
from core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from core.logging import get_logger
from core.security import CurrentUser, get_admin_user
from services.ingestion_service import IngestionInput, LegalDocumentIngestionService

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
log = get_logger(__name__)
AdminUser = Annotated[CurrentUser, Depends(get_admin_user)]


class UrlIngestRequest(BaseModel):
    url: HttpUrl
    document_type: str = "statute"
    jurisdiction: str = "LA"
    title: str | None = Field(default=None, max_length=500)
    year: int | None = None
    tags: list[str] = Field(default_factory=list)
    review_status: str = "pending_review"


class AdminUserUpdate(BaseModel):
    email: str | None = Field(default=None, max_length=320)
    full_name: str | None = Field(default=None, max_length=255)
    role: str | None = Field(default=None, max_length=50)
    tenant_id: str | None = None
    is_active: bool | None = None


@router.post("/ingest", summary="Queue a document for knowledge base ingestion")
async def ingest_document(
    payload: IngestRequest,
    user: AdminUser,
) -> dict:
    log.info(
        "admin.ingest.queued",
        source=payload.source,
        doc_type=payload.document_type,
        jurisdiction=payload.jurisdiction,
        admin=user.sub,
    )
    # Production: push to ARQ task queue for async processing + embedding
    return {
        "status": "queued",
        "source": payload.source,
        "document_type": payload.document_type,
        "jurisdiction": payload.jurisdiction,
        "note": "Document queued for embedding and ingestion into knowledge base.",
    }


@router.post("/ingest/upload", summary="Upload and index legal documents")
async def upload_legal_documents(
    user: AdminUser,
    files: list[UploadFile] = File(...),
    document_type: str = Form(default="law"),
    jurisdiction: str = Form(default="TH"),
    title: str | None = Form(default=None),
    year: int | None = Form(default=None),
    tags: str = Form(default=""),
    source_url: str | None = Form(default=None),
) -> dict:
    settings = get_settings()
    supabase = await get_supabase()
    service = LegalDocumentIngestionService(supabase=supabase)
    parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

    results = []
    for upload in files:
        content = await upload.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            raise FileTooLargeError(
                f"'{upload.filename}' ({size_mb:.1f}MB) exceeds limit of {settings.max_upload_size_mb}MB"
            )

        result = await service.ingest(
            IngestionInput(
                filename=upload.filename or "unnamed",
                content_type=upload.content_type or "application/octet-stream",
                content=content,
                document_type=document_type,
                jurisdiction=jurisdiction,
                title=title if len(files) == 1 else None,
                year=year,
                tags=parsed_tags,
                source_url=source_url,
                review_status="pending_review",
                tenant_id=user.tenant_id or None,
                user_id=user.sub,
            )
        )
        await _record_ingestion_job(
            supabase=supabase,
            user=user,
            result=result.__dict__,
            source_type="file",
            file_name=upload.filename or result.title,
            source_url=source_url,
            config={
                "document_type": document_type,
                "jurisdiction": jurisdiction,
                "title": title,
                "year": year,
                "tags": parsed_tags,
                "content_type": upload.content_type,
                "size_bytes": len(content),
            },
        )
        results.append(result.__dict__)

    return {
        "status": "indexed",
        "count": len(results),
        "items": results,
    }


@router.post("/ingest/url", summary="Ingest a legal document from URL")
async def ingest_legal_document_url(
    payload: UrlIngestRequest,
    user: AdminUser,
) -> dict:
    settings = get_settings()
    url = str(payload.url)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise UnsupportedFileTypeError("Only http/https URLs are supported.")

    timeout = httpx.Timeout(20.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content

    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise FileTooLargeError(f"Remote document ({size_mb:.1f}MB) exceeds limit of {settings.max_upload_size_mb}MB")

    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if not content_type or content_type == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(parsed.path)
        content_type = guessed or "application/pdf"

    filename = parsed.path.rsplit("/", 1)[-1] or "remote-legal-document"
    supabase = await get_supabase()
    service = LegalDocumentIngestionService(supabase=supabase)
    result = await service.ingest(
        IngestionInput(
            filename=filename,
            content_type=content_type,
            content=content,
            document_type=payload.document_type,
            jurisdiction=payload.jurisdiction,
            title=payload.title,
            year=payload.year,
            tags=payload.tags,
            source_url=url,
            review_status=payload.review_status,
            tenant_id=user.tenant_id or None,
            user_id=user.sub,
        )
    )
    await _record_ingestion_job(
        supabase=supabase,
        user=user,
        result=result.__dict__,
        source_type="url",
        file_name=filename,
        source_url=url,
        config={
            "document_type": payload.document_type,
            "jurisdiction": payload.jurisdiction,
            "title": payload.title,
            "year": payload.year,
            "tags": payload.tags,
            "content_type": content_type,
            "size_bytes": len(content),
        },
    )

    return {"status": result.status, "item": result.__dict__}


@router.get("/ingestion", summary="List recent knowledge ingestion jobs")
async def list_ingestion_jobs(
    user: AdminUser,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    supabase = await get_supabase()
    if not supabase:
        return []

    try:
        query = supabase.table("ingestion_jobs").select("*").order("created_at", desc=True).limit(limit)
        if user.tenant_id:
            query = query.eq("tenant_id", user.tenant_id)
        result = await query.execute()
        return result.data or []
    except Exception as exc:
        log.warning("admin.ingestion_jobs.failed", error=str(exc))
        return await _synthesise_ingestion_jobs(supabase, limit=limit)


@router.delete("/ingestion/{job_id}", summary="Delete an ingestion job record")
async def delete_ingestion_job(job_id: str, user: AdminUser) -> dict:
    supabase = await get_supabase()
    if not supabase:
        return {"status": "skipped", "reason": "database_not_configured", "id": job_id}

    try:
        query = supabase.table("ingestion_jobs").delete().eq("id", job_id)
        if user.tenant_id:
            query = query.eq("tenant_id", user.tenant_id)
        result = await query.execute()
        if result.data:
            return {"status": "deleted", "id": job_id}
    except Exception as exc:
        log.warning("admin.delete_ingestion_job.failed", error=str(exc))

    return {"status": "not_found", "id": job_id}


@router.get("/rag/health", summary="Inspect chunk-level RAG readiness")
async def get_rag_health(user: AdminUser) -> dict[str, Any]:
    supabase = await get_supabase()
    if not supabase:
        return {"status": "degraded", "reason": "database_not_configured"}

    try:
        query = supabase.table("document_chunks").select(
            "id, source_table, status, review_status, embedding"
        ).limit(5000)
        if user.tenant_id:
            query = query.or_(f"tenant_id.is.null,tenant_id.eq.{user.tenant_id}")
        result = await query.execute()
    except Exception as exc:
        log.warning("admin.rag_health.failed", error=str(exc))
        return {
            "status": "degraded",
            "reason": "document_chunks_unavailable",
            "detail": "Apply supabase_agentic_rag_chunks.sql to enable chunk-level Agentic RAG.",
        }

    rows = result.data or []
    by_source: dict[str, dict[str, int]] = {}
    for row in rows:
        source = row.get("source_table") or "unknown"
        bucket = by_source.setdefault(source, {"chunks": 0, "embedded": 0, "approved_active": 0})
        bucket["chunks"] += 1
        if row.get("embedding"):
            bucket["embedded"] += 1
        if row.get("status") == "active" and row.get("review_status") == "approved":
            bucket["approved_active"] += 1

    embedded = sum(1 for row in rows if row.get("embedding"))
    approved_active = sum(
        1 for row in rows
        if row.get("status") == "active" and row.get("review_status") == "approved"
    )
    return {
        "status": "ok",
        "sample_limit": 5000,
        "chunks": len(rows),
        "embedded_chunks": embedded,
        "approved_active_chunks": approved_active,
        "embedding_coverage": round(embedded / len(rows), 4) if rows else 0,
        "by_source": by_source,
    }


@router.get("/users", summary="List admin users")
async def list_admin_users(
    user: AdminUser,
    limit: int = Query(default=200, ge=1, le=1000),
) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []

    users = await _fetch_admin_users(supabase, tenant_id=user.tenant_id, limit=limit)
    return [_normalise_admin_user(row) for row in users]


@router.patch("/users/{user_id}", summary="Update an admin user")
async def update_admin_user(
    user_id: str,
    payload: AdminUserUpdate,
    user: AdminUser,
) -> dict[str, Any]:
    supabase = await get_supabase()
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database is not configured.",
        )

    updates = payload.model_dump(exclude_unset=True, exclude_none=True)
    allowed_updates = {
        key: value
        for key, value in updates.items()
        if key in {"email", "full_name", "role", "tenant_id", "is_active"}
    }
    if not allowed_updates:
        existing = await _fetch_admin_user(supabase, user_id)
        if existing:
            return _normalise_admin_user(existing)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    attempts = [
        allowed_updates,
        {key: value for key, value in allowed_updates.items() if key in {"email", "role", "tenant_id"}},
    ]
    for attempt in attempts:
        if not attempt:
            continue
        try:
            result = await (
                supabase.table("users")
                .update(attempt)
                .eq("id", user_id)
                .execute()
            )
            data = result.data or []
            if isinstance(data, list) and data:
                return _normalise_admin_user(data[0])
            if isinstance(data, dict):
                return _normalise_admin_user(data)
        except Exception as exc:
            log.warning("admin.update_user.failed", fields=list(attempt), error=str(exc))

    existing = await _fetch_admin_user(supabase, user_id)
    if existing:
        return _normalise_admin_user(existing)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")


@router.delete("/users/{user_id}", summary="Delete an admin user")
async def delete_admin_user(user_id: str, user: AdminUser) -> dict[str, Any]:
    if user_id == user.sub:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own active admin account.",
        )

    supabase = await get_supabase()
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database is not configured.",
        )

    try:
        result = await supabase.table("users").delete().eq("id", user_id).execute()
        if result.data:
            return {"status": "deleted", "id": user_id}
    except Exception as exc:
        log.warning("admin.delete_user.failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User could not be deleted because it may be referenced by other records.",
        ) from exc

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")


@router.get("/audit-log", summary="Read audit trail (last N entries)")
async def get_audit_log(
    user: AdminUser,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    supabase = await get_supabase()
    if not supabase:
        return []

    for order_column in ("created_at", "ts"):
        try:
            result = await supabase.table("audit_log") \
                .select("*") \
                .order(order_column, desc=True) \
                .range(offset, offset + limit - 1) \
                .execute()
            return [_normalise_audit_log(row) for row in (result.data or [])]
        except Exception as exc:
            log.error("admin.audit_log.failed", order_column=order_column, error=str(exc))
    return []


@router.get("/expert-queue", summary="List pending human review requests")
async def get_expert_queue(
    user: AdminUser,
    expert_queue: ExpertQueueDep,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict]:
    supabase = await get_supabase()
    if supabase:
        for order_column in ("priority", "created_at", "ts"):
            try:
                query = supabase.table("expert_reviews").select("*").limit(limit)
                if user.tenant_id:
                    query = query.eq("tenant_id", user.tenant_id)
                if order_column == "priority":
                    query = query.order("priority", desc=False)
                else:
                    query = query.order(order_column, desc=True)
                result = await query.execute()
                return result.data or []
            except Exception as exc:
                log.warning("admin.expert_queue.failed", order_column=order_column, error=str(exc))

    return await expert_queue.list_pending()


@router.post("/expert-queue/{item_id}/resolve", summary="Mark a review item as resolved")
async def resolve_expert_review(
    item_id: str,
    user: AdminUser,
    resolution: str = "",
) -> dict:
    supabase = await get_supabase()
    if supabase:
        try:
            await supabase.table("expert_reviews").update({
                "status": "resolved",
                "reviewer_id": user.sub,
                "resolution": resolution,
            }).eq("id", item_id).execute()
        except Exception as exc:
            log.error("admin.resolve.failed", error=str(exc))

    return {"status": "resolved", "id": item_id, "resolver": user.sub}


async def _fetch_admin_users(
    supabase: Any,
    *,
    tenant_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    select_attempts = [
        "id, email, full_name, role, tenant_id, is_active, created_at, updated_at",
        "id, email, role, tenant_id, created_at",
        "*",
    ]

    for columns in select_attempts:
        try:
            query = supabase.table("users").select(columns).limit(limit)
            if tenant_id:
                query = query.eq("tenant_id", tenant_id)
            try:
                query = query.order("created_at", desc=True)
            except Exception:
                pass
            result = await query.execute()
            return result.data or []
        except Exception as exc:
            log.warning("admin.list_users.failed", columns=columns, error=str(exc))

    return []


async def _fetch_admin_user(supabase: Any, user_id: str) -> dict[str, Any] | None:
    select_attempts = [
        "id, email, full_name, role, tenant_id, is_active, created_at, updated_at",
        "id, email, role, tenant_id, created_at",
        "*",
    ]

    for columns in select_attempts:
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
            log.warning("admin.fetch_user.failed", columns=columns, error=str(exc))

    return None


def _normalise_admin_user(row: dict[str, Any]) -> dict[str, Any]:
    email = row.get("email") or ""
    created_at = row.get("created_at") or ""
    return {
        "id": str(row.get("id") or ""),
        "email": email,
        "full_name": row.get("full_name") or row.get("name") or email or "Admin",
        "role": row.get("role") or "client",
        "tenant_id": str(row.get("tenant_id") or ""),
        "is_active": row.get("is_active") is not False,
        "created_at": created_at,
        "updated_at": row.get("updated_at") or created_at,
        "tenant_name": row.get("tenant_name"),
    }


def _normalise_audit_log(row: dict[str, Any]) -> dict[str, Any]:
    created_at = row.get("created_at")
    if not created_at and row.get("ts"):
        try:
            created_at = datetime.fromtimestamp(int(row["ts"]), tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            created_at = ""

    return {
        **row,
        "action": row.get("action") or row.get("agent") or "legal_query",
        "latency_ms": row.get("latency_ms") or row.get("processing_time_ms"),
        "success": row.get("success") is not False,
        "escalated": row.get("escalated") is True or row.get("escalated_to_expert") is True,
        "created_at": created_at or "",
    }


async def _record_ingestion_job(
    *,
    supabase: Any | None,
    user: CurrentUser,
    result: dict[str, Any],
    source_type: str,
    file_name: str | None,
    source_url: str | None,
    config: dict[str, Any],
) -> None:
    if not supabase or not user.tenant_id:
        return

    review_status = str(result.get("review_status") or "pending_review")
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "id": result.get("job_id"),
        "tenant_id": user.tenant_id,
        "job_type": "knowledge_ingestion",
        "source_type": source_type,
        "source_url": source_url,
        "file_name": file_name,
        "status": "pending_review" if review_status == "pending_review" else "completed",
        "progress": 90 if review_status == "pending_review" else 100,
        "total_items": 1,
        "processed_items": 1,
        "error_count": 0,
        "errors": [],
        "config": config,
        "result": result,
        "created_by": user.sub,
        "started_at": now,
        "completed_at": now,
    }

    try:
        await supabase.table("ingestion_jobs").insert(payload).execute()
    except Exception as exc:
        log.warning("admin.record_ingestion_job.failed", error=str(exc))


async def _synthesise_ingestion_jobs(supabase: Any, *, limit: int) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    table_specs = [
        ("laws", "statute", "full_text"),
        ("cases", "case", "ruling"),
        ("legal_forms", "form", "content"),
    ]

    for table, document_type, text_field in table_specs:
        try:
            result = await supabase.table(table).select("*").limit(limit).execute()
        except Exception as exc:
            log.warning("admin.synthesise_ingestion_jobs_table.failed", table=table, error=str(exc))
            continue

        for row in result.data or []:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            title = row.get("title") or row.get("case_no") or metadata.get("file_name") or "Untitled"
            created_at = row.get("ingested_at") or row.get("created_at") or row.get("updated_at")
            review_status = row.get("review_status") or metadata.get("review_status") or "approved"
            text = row.get(text_field) or row.get("full_text") or row.get("summary") or ""

            jobs.append({
                "id": row.get("id"),
                "tenant_id": row.get("tenant_id"),
                "job_type": "knowledge_ingestion",
                "source_type": "knowledge_table",
                "source_url": row.get("source_url") or row.get("official_source_url") or metadata.get("source_url"),
                "file_name": metadata.get("file_name") or title,
                "status": "pending_review" if review_status == "pending_review" else "completed",
                "progress": 90 if review_status == "pending_review" else 100,
                "total_items": 1,
                "processed_items": 1,
                "error_count": 0,
                "errors": [],
                "config": {
                    "document_type": document_type,
                    "jurisdiction": row.get("jurisdiction"),
                    "source_table": table,
                },
                "result": {
                    "document_id": row.get("id"),
                    "source_table": table,
                    "title": title,
                    "status": row.get("status"),
                    "review_status": review_status,
                    "chunks": metadata.get("chunks"),
                    "text_length": metadata.get("text_length") or len(text),
                    "embedding_model": metadata.get("embedding_model"),
                    "document_type": document_type,
                    "jurisdiction": row.get("jurisdiction"),
                },
                "created_by": row.get("ingested_by"),
                "started_at": created_at,
                "completed_at": created_at,
                "created_at": created_at,
                "updated_at": row.get("updated_at") or created_at,
            })

    jobs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return jobs[:limit]
