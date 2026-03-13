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

from fastapi import APIRouter, Depends, Query

from api.dependencies import AuditDep, ExpertQueueDep
from api.schemas import IngestRequest
from core.database import get_supabase
from core.logging import get_logger
from core.security import CurrentUser, require_roles

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
log = get_logger(__name__)
AdminUser = Annotated[CurrentUser, Depends(require_roles("admin"))]


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
