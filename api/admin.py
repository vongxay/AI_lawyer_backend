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
from html.parser import HTMLParser
import mimetypes
import re
import uuid
from typing import Annotated, Any
from urllib.parse import unquote, urljoin, urlparse

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

        try:
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
            result_payload = result.__dict__
        except UnsupportedFileTypeError as exc:
            result_payload = _failed_upload_result(
                exc=exc,
                file_name=upload.filename or "unnamed",
                content_type=upload.content_type or "application/octet-stream",
                document_type=document_type,
                jurisdiction=jurisdiction,
                title=title if len(files) == 1 else None,
            )
            log.warning(
                "admin.ingest.upload_file_failed",
                file_name=upload.filename,
                content_type=upload.content_type,
                error=exc.message,
                details=exc.details,
            )

        await _record_ingestion_job(
            supabase=supabase,
            user=user,
            result=result_payload,
            source_type="file",
            file_name=upload.filename or result_payload["title"],
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
        results.append(result_payload)

    failed_count = len([item for item in results if item.get("status") == "failed"])
    processed_count = len(results) - failed_count
    response_status = "indexed"
    if failed_count and processed_count:
        response_status = "completed_with_errors"
    elif failed_count:
        response_status = "failed"

    return {
        "status": response_status,
        "count": len(results),
        "processed_count": processed_count,
        "failed_count": failed_count,
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
    supabase = await get_supabase()
    service = LegalDocumentIngestionService(supabase=supabase)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content

        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            raise FileTooLargeError(
                f"Remote document ({size_mb:.1f}MB) exceeds limit of {settings.max_upload_size_mb}MB"
            )

        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if not content_type or content_type == "application/octet-stream":
            guessed, _ = mimetypes.guess_type(parsed.path)
            content_type = guessed or "application/pdf"

        if content_type == "text/html":
            html = content.decode(response.encoding or "utf-8", errors="replace")
            document_links = _extract_document_links(html, str(response.url), limit=settings.url_ingest_max_documents)
            if not document_links:
                raise UnsupportedFileTypeError(
                    "The URL returned an HTML page, but no downloadable legal document links were found.",
                    details={
                        "content_type": content_type,
                        "url": url,
                        "hint": "Use a direct PDF/DOCX URL, or a page that contains document download links.",
                    },
                )

            results: list[dict[str, Any]] = []
            for link in document_links:
                try:
                    result_payload = await _ingest_remote_document(
                        client=client,
                        service=service,
                        supabase=supabase,
                        user=user,
                        url=link["url"],
                        source_page_url=url,
                        document_type=payload.document_type,
                        jurisdiction=payload.jurisdiction,
                        title=payload.title if len(document_links) == 1 else None,
                        year=payload.year,
                        tags=payload.tags,
                        review_status=payload.review_status,
                        settings=settings,
                    )
                except Exception as exc:
                    result_payload = _failed_remote_document_result(
                        exc=exc,
                        url=link["url"],
                        document_type=payload.document_type,
                        jurisdiction=payload.jurisdiction,
                        title=link.get("title"),
                    )
                    log.warning("admin.ingest.url_link_failed", url=link["url"], error=str(exc))
                results.append(result_payload)

            return {
                **_multi_ingestion_response(results),
                "source": "html_index",
                "source_url": url,
                "discovered_count": len(document_links),
                "item": results[0] if results else None,
            }

        result_payload = await _ingest_remote_document(
            client=client,
            service=service,
            supabase=supabase,
            user=user,
            url=url,
            source_page_url=None,
            document_type=payload.document_type,
            jurisdiction=payload.jurisdiction,
            title=payload.title,
            year=payload.year,
            tags=payload.tags,
            review_status=payload.review_status,
            settings=settings,
            prefetched_content=content,
            prefetched_content_type=content_type,
            prefetched_final_url=str(response.url),
        )

        return {
            **_multi_ingestion_response([result_payload]),
            "source": "direct_url",
            "item": result_payload,
        }


async def _ingest_remote_document(
    *,
    client: httpx.AsyncClient,
    service: LegalDocumentIngestionService,
    supabase: Any | None,
    user: CurrentUser,
    url: str,
    source_page_url: str | None,
    document_type: str,
    jurisdiction: str,
    title: str | None,
    year: int | None,
    tags: list[str],
    review_status: str,
    settings: Any,
    prefetched_content: bytes | None = None,
    prefetched_content_type: str | None = None,
    prefetched_final_url: str | None = None,
) -> dict[str, Any]:
    if prefetched_content is None:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content
        final_url = str(response.url)
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    else:
        content = prefetched_content
        final_url = prefetched_final_url or url
        content_type = prefetched_content_type or "application/octet-stream"

    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise FileTooLargeError(f"Remote document ({size_mb:.1f}MB) exceeds limit of {settings.max_upload_size_mb}MB")

    parsed_final = urlparse(final_url)
    if not content_type or content_type == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(parsed_final.path)
        content_type = guessed or "application/pdf"

    filename = _filename_from_url(final_url)
    try:
        result = await service.ingest(
            IngestionInput(
                filename=filename,
                content_type=content_type,
                content=content,
                document_type=document_type,
                jurisdiction=jurisdiction,
                title=title,
                year=year,
                tags=tags,
                source_url=final_url,
                review_status=review_status,
                tenant_id=user.tenant_id or None,
                user_id=user.sub,
            )
        )
        result_payload = result.__dict__
    except UnsupportedFileTypeError as exc:
        result_payload = _failed_upload_result(
            exc=exc,
            file_name=filename,
            content_type=content_type,
            document_type=document_type,
            jurisdiction=jurisdiction,
            title=title,
        )
        log.warning(
            "admin.ingest.url_document_failed",
            url=url,
            final_url=final_url,
            content_type=content_type,
            error=exc.message,
            details=exc.details,
        )

    await _record_ingestion_job(
        supabase=supabase,
        user=user,
        result=result_payload,
        source_type="url",
        file_name=filename or result_payload["title"],
        source_url=final_url,
        config={
            "document_type": document_type,
            "jurisdiction": jurisdiction,
            "title": title,
            "year": year,
            "tags": tags,
            "content_type": content_type,
            "size_bytes": len(content),
            "source_page_url": source_page_url,
            "requested_url": url,
            "final_url": final_url,
        },
    )
    return result_payload


def _multi_ingestion_response(results: list[dict[str, Any]]) -> dict[str, Any]:
    failed_count = len([item for item in results if item.get("status") == "failed"])
    processed_count = len(results) - failed_count
    response_status = "indexed"
    if failed_count and processed_count:
        response_status = "completed_with_errors"
    elif failed_count:
        response_status = "failed"
    return {
        "status": response_status,
        "count": len(results),
        "processed_count": processed_count,
        "failed_count": failed_count,
        "items": results,
    }


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
    result_status = str(result.get("status") or "")
    result_error = result.get("error")
    errors = []
    if result_error:
        errors.append({
            "message": str(result_error),
            "details": result.get("details") or {},
        })
    failed = result_status == "failed" or bool(errors)
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "id": result.get("job_id"),
        "tenant_id": user.tenant_id,
        "job_type": "knowledge_ingestion",
        "source_type": source_type,
        "source_url": source_url,
        "file_name": file_name,
        "status": "failed" if failed else ("pending_review" if review_status == "pending_review" else "completed"),
        "progress": 100 if failed else (90 if review_status == "pending_review" else 100),
        "total_items": 1,
        "processed_items": 0 if failed else 1,
        "error_count": len(errors),
        "errors": errors,
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


class _DocumentLinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._active_href: str | None = None
        self._active_text: list[str] = []
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        self._active_href = urljoin(self._base_url, href)
        self._active_text = []

    def handle_data(self, data: str) -> None:
        if self._active_href:
            self._active_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._active_href:
            return
        self.links.append({
            "url": self._active_href,
            "title": re.sub(r"\s+", " ", " ".join(self._active_text)).strip(),
        })
        self._active_href = None
        self._active_text = []


def _extract_document_links(html: str, base_url: str, *, limit: int) -> list[dict[str, str]]:
    parser = _DocumentLinkParser(base_url)
    parser.feed(html)

    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for link in parser.links:
        url = link["url"]
        if not _looks_like_document_url(url, link.get("title", "")):
            continue
        key = url.casefold()
        if key in seen:
            continue
        seen.add(key)
        links.append(link)
        if len(links) >= max(1, limit):
            break
    return links


def _looks_like_document_url(url: str, text: str = "") -> bool:
    parsed = urlparse(url)
    path = unquote(parsed.path).casefold()
    haystack = f"{path} {text.casefold()}"
    if "/documents.search" in path:
        return False
    if "/document/view/" in path or "/storage/document/" in path:
        return True
    return any(haystack.endswith(ext) or f".{ext}" in haystack for ext in ("pdf", "docx", "doc", "txt", "md"))


def _filename_from_url(url: str, fallback: str = "remote-legal-document") -> str:
    filename = unquote(urlparse(url).path.rsplit("/", 1)[-1]).strip()
    return filename or fallback


def _failed_remote_document_result(
    *,
    exc: Exception,
    url: str,
    document_type: str,
    jurisdiction: str,
    title: str | None,
) -> dict[str, Any]:
    file_name = _filename_from_url(url)
    message = str(exc) or exc.__class__.__name__
    return {
        "job_id": str(uuid.uuid4()),
        "document_id": None,
        "source_table": _source_table_for_document_type(document_type),
        "title": title or _title_from_filename(file_name),
        "status": "failed",
        "chunks": 0,
        "chunks_indexed": 0,
        "chunks_embedded": 0,
        "text_length": 0,
        "embedding_model": None,
        "review_status": "failed",
        "document_type": document_type,
        "jurisdiction": jurisdiction,
        "warnings": [f"Could not ingest remote document from {url}."],
        "error": message,
        "details": {
            "url": url,
            "exception_type": exc.__class__.__name__,
        },
    }


def _failed_upload_result(
    *,
    exc: UnsupportedFileTypeError,
    file_name: str,
    content_type: str,
    document_type: str,
    jurisdiction: str,
    title: str | None,
) -> dict[str, Any]:
    return {
        "job_id": str(uuid.uuid4()),
        "document_id": None,
        "source_table": _source_table_for_document_type(document_type),
        "title": title or _title_from_filename(file_name),
        "status": "failed",
        "chunks": 0,
        "chunks_indexed": 0,
        "chunks_embedded": 0,
        "text_length": 0,
        "embedding_model": None,
        "review_status": "failed",
        "document_type": document_type,
        "jurisdiction": jurisdiction,
        "warnings": [_upload_failure_hint(exc, content_type=content_type)],
        "error": exc.message,
        "details": exc.details,
    }


def _upload_failure_hint(exc: UnsupportedFileTypeError, *, content_type: str) -> str:
    if content_type == "application/pdf" and "PDF text extraction failed" in exc.message:
        return (
            "This PDF could not be indexed with reliable Lao legal text. Use a Unicode text-searchable PDF, "
            "or install/configure Tesseract with Lao/Thai language data and re-upload."
        )
    return exc.message


def _source_table_for_document_type(document_type: str) -> str:
    normalized = document_type.strip().lower()
    if normalized in {"case", "case_law", "judgment"}:
        return "cases"
    if normalized in {"form", "template"}:
        return "legal_forms"
    return "laws"


def _title_from_filename(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return re.sub(r"\.[^.]+$", "", stem).replace("_", " ").replace("-", " ").strip() or filename


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
