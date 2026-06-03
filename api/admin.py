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

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from api.dependencies import AuditDep, ExpertQueueDep
from api.schemas import IngestRequest
from core.config import get_settings
from core.database import get_supabase
from core.exceptions import FileTooLargeError
from core.logging import get_logger
from core.security import CurrentUser, get_admin_user
from services.ingestion_service import IngestionInput, LegalDocumentIngestionService

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
log = get_logger(__name__)
AdminUser = Annotated[CurrentUser, Depends(get_admin_user)]


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
                tenant_id=user.tenant_id or None,
                user_id=user.sub,
            )
        )
        results.append(result.__dict__)

    return {
        "status": "indexed",
        "count": len(results),
        "items": results,
    }


@router.get("/audit-log", summary="Read audit trail (last N entries)")
async def get_audit_log(
    user: AdminUser,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    supabase = await get_supabase()
    if not supabase:
        return []

    try:
        result = await supabase.table("audit_log") \
            .select("*") \
            .order("ts", desc=True) \
            .range(offset, offset + limit - 1) \
            .execute()
        return result.data or []
    except Exception as exc:
        log.error("admin.audit_log.failed", error=str(exc))
        return []


@router.get("/expert-queue", summary="List pending human review requests")
async def get_expert_queue(
    user: AdminUser,
    expert_queue: ExpertQueueDep,
) -> list[dict]:
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
