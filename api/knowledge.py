"""
api/knowledge.py
================
Knowledge-base management endpoints for the admin UI.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from core.database import get_supabase
from core.logging import get_logger
from core.security import CurrentUser, get_admin_user
from services.ingestion_service import IngestionInput, LegalDocumentIngestionService

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])
log = get_logger(__name__)
AdminUser = Annotated[CurrentUser, Depends(get_admin_user)]


class CreateKnowledgeDocument(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    type: str = "statute"
    jurisdiction: str = "laos"
    year: int | None = None
    fullText: str = Field(default="", max_length=500_000)
    tags: list[str] = Field(default_factory=list)


class UpdateKnowledgeDocument(BaseModel):
    title: str | None = None
    jurisdiction: str | None = None
    year: int | None = None
    status: str | None = None
    tags: list[str] | None = None
    reviewStatus: str | None = None
    reviewNotes: str | None = None


class ReviewDecision(BaseModel):
    action: str = Field(pattern="^(approve|reject)$")
    notes: str | None = Field(default=None, max_length=2000)


@router.get("/documents", summary="List legal knowledge documents")
async def list_documents(
    user: AdminUser,
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict:
    supabase = await get_supabase()
    if not supabase:
        return {"items": [], "total": 0}

    docs: list[dict[str, Any]] = []
    table_specs = [
        ("laws", "statute"),
        ("cases", "case"),
        ("legal_forms", "form"),
    ]

    for table, fallback_type in table_specs:
        try:
            result = await supabase.table(table).select("*").limit(limit).execute()
            for row in result.data or []:
                docs.append(_map_document(table, fallback_type, row))
        except Exception as exc:
            log.warning("knowledge.list_table.failed", table=table, error=str(exc))

    docs.sort(key=lambda item: item.get("_sort", ""), reverse=True)
    for doc in docs:
        doc.pop("_sort", None)
    return {"items": docs[:limit], "total": len(docs)}


@router.get("/documents/review", summary="List documents pending knowledge review")
async def list_pending_review_documents(
    user: AdminUser,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    supabase = await get_supabase()
    if not supabase:
        return {"items": [], "total": 0}

    items: list[dict[str, Any]] = []
    for table, fallback_type in [("laws", "statute"), ("cases", "case"), ("legal_forms", "form")]:
        try:
            result = await (
                supabase.table(table)
                .select("*")
                .eq("review_status", "pending_review")
                .limit(limit)
                .execute()
            )
            for row in result.data or []:
                items.append(_map_document(table, fallback_type, row))
        except Exception as exc:
            log.warning("knowledge.review_list_table.failed", table=table, error=str(exc))

    return {"items": items[:limit], "total": len(items)}


@router.post("/documents", summary="Create a legal knowledge document")
async def create_document(
    payload: CreateKnowledgeDocument,
    user: AdminUser,
) -> dict:
    supabase = await get_supabase()
    service = LegalDocumentIngestionService(supabase=supabase)
    result = await service.ingest(
        IngestionInput(
            filename=f"{payload.title}.txt",
            content_type="text/plain",
            content=(payload.fullText or payload.title).encode("utf-8"),
            document_type=payload.type,
            jurisdiction=payload.jurisdiction,
            title=payload.title,
            year=payload.year,
            tags=payload.tags,
            tenant_id=user.tenant_id or None,
            user_id=user.sub,
            allow_short_text=True,
        )
    )
    return {"id": result.document_id, **result.__dict__}


@router.put("/documents/{document_id}", summary="Update a legal knowledge document")
async def update_document(
    document_id: str,
    payload: UpdateKnowledgeDocument,
    user: AdminUser,
) -> dict:
    supabase = await get_supabase()
    if not supabase:
        return {"status": "skipped", "reason": "database_not_configured"}

    updates = payload.model_dump(exclude_unset=True)
    for table in ("laws", "cases", "legal_forms"):
        mapped = _map_update(table, updates)
        if not mapped:
            continue
        try:
            result = await supabase.table(table).update(mapped).eq("id", document_id).execute()
            if result.data:
                await _update_document_chunks(supabase, table, document_id, mapped)
                return {"status": "updated", "id": document_id, "source_table": table}
        except Exception as exc:
            log.warning("knowledge.update_table.failed", table=table, error=str(exc))

    return {"status": "not_found", "id": document_id}


@router.delete("/documents/{document_id}", summary="Delete a legal knowledge document")
async def delete_document(document_id: str, user: AdminUser) -> dict:
    supabase = await get_supabase()
    if not supabase:
        return {"status": "skipped", "reason": "database_not_configured"}

    for table in ("laws", "cases", "legal_forms"):
        try:
            result = await supabase.table(table).delete().eq("id", document_id).execute()
            if result.data:
                await _delete_document_chunks(supabase, table, document_id)
                return {"status": "deleted", "id": document_id, "source_table": table}
        except Exception as exc:
            log.warning("knowledge.delete_table.failed", table=table, error=str(exc))

    return {"status": "not_found", "id": document_id}


@router.post("/documents/{document_id}/review", summary="Approve or reject a knowledge document")
async def review_document(
    document_id: str,
    payload: ReviewDecision,
    user: AdminUser,
) -> dict:
    supabase = await get_supabase()
    if not supabase:
        return {"status": "skipped", "reason": "database_not_configured"}

    review_status = "approved" if payload.action == "approve" else "rejected"
    reviewed_at = datetime.now(timezone.utc).isoformat()
    for table in ("laws", "cases", "legal_forms"):
        try:
            updates: dict[str, Any] = {
                "review_status": review_status,
                "reviewed_by": user.sub,
                "reviewed_at": reviewed_at,
                "review_notes": payload.notes,
            }
            if table == "legal_forms":
                updates["is_active"] = payload.action == "approve"
            else:
                updates["status"] = "active" if payload.action == "approve" else "pending"

            result = await (
                supabase.table(table)
                .update(updates)
                .eq("id", document_id)
                .execute()
            )
            if result.data:
                await _update_document_chunks(supabase, table, document_id, updates)
                return {"status": review_status, "id": document_id, "source_table": table}
        except Exception as exc:
            log.warning("knowledge.review_table.failed", table=table, error=str(exc))

    return {"status": "not_found", "id": document_id}


async def _update_document_chunks(
    supabase: Any,
    source_table: str,
    document_id: str,
    source_updates: dict[str, Any],
) -> None:
    updates: dict[str, Any] = {}
    if "review_status" in source_updates:
        updates["review_status"] = source_updates["review_status"]
    if "status" in source_updates:
        updates["status"] = source_updates["status"]
    if "is_active" in source_updates:
        updates["status"] = "active" if source_updates["is_active"] else "archived"
    if "title" in source_updates:
        updates["title"] = source_updates["title"]
    if "case_no" in source_updates:
        updates["title"] = source_updates["case_no"]
    if "jurisdiction" in source_updates:
        updates["jurisdiction"] = str(source_updates["jurisdiction"])

    if not updates:
        return

    try:
        await (
            supabase.table("document_chunks")
            .update(updates)
            .eq("source_table", source_table)
            .eq("source_id", document_id)
            .execute()
        )
    except Exception as exc:
        log.warning("knowledge.update_chunks.failed", source_table=source_table, error=str(exc))


async def _delete_document_chunks(supabase: Any, source_table: str, document_id: str) -> None:
    try:
        await (
            supabase.table("document_chunks")
            .delete()
            .eq("source_table", source_table)
            .eq("source_id", document_id)
            .execute()
        )
    except Exception as exc:
        log.warning("knowledge.delete_chunks.failed", source_table=source_table, error=str(exc))


def _map_document(table: str, fallback_type: str, row: dict[str, Any]) -> dict[str, Any]:
    if table == "cases":
        title = row.get("case_no") or row.get("title") or "Untitled case"
        text = row.get("full_text") or row.get("ruling") or row.get("summary") or ""
        year = row.get("year_be") or row.get("year") or row.get("case_no_year") or 0
    elif table == "legal_forms":
        title = row.get("title") or "Untitled form"
        text = row.get("content") or ""
        year = 0
    else:
        title = row.get("title") or "Untitled law"
        text = row.get("full_text") or ""
        year = row.get("year_be") or row.get("year") or 0

    review_status = row.get("review_status") or (row.get("metadata") or {}).get("review_status") or "approved"
    status = "pending" if review_status == "pending_review" else row.get("status") or (
        "active" if row.get("is_active", True) else "archived"
    )
    return {
        "id": row.get("id"),
        "title": title,
        "type": fallback_type,
        "sourceTable": table,
        "jurisdiction": _short_jurisdiction(str(row.get("jurisdiction") or "laos")),
        "year": year,
        "status": _ui_status(str(status)),
        "lastUpdated": row.get("updated_at") or row.get("created_at") or row.get("ingested_at") or "",
        "tags": row.get("tags") or [],
        "hasEmbedding": bool(row.get("embedding")),
        "textLength": len(text),
        "reviewStatus": review_status,
        "sourceAuthority": row.get("source_authority") or (row.get("metadata") or {}).get("source_authority"),
        "officialSourceUrl": row.get("official_source_url") or row.get("source_url") or (row.get("metadata") or {}).get("official_source_url"),
        "lawNo": row.get("law_no") or (row.get("metadata") or {}).get("law_no"),
        "article": row.get("article") or row.get("section_number") or row.get("section") or (row.get("metadata") or {}).get("article"),
        "effectiveDate": row.get("effective_date") or (row.get("metadata") or {}).get("effective_date"),
        "_sort": row.get("updated_at") or row.get("created_at") or "",
    }


def _map_update(table: str, updates: dict[str, Any]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    if "title" in updates:
        mapped["case_no" if table == "cases" else "title"] = updates["title"]
    if "jurisdiction" in updates:
        mapped["jurisdiction"] = _db_jurisdiction(str(updates["jurisdiction"]))
    if "year" in updates:
        mapped["year_be" if table != "legal_forms" else "version"] = updates["year"]
    if "tags" in updates and table != "legal_forms":
        mapped["tags"] = updates["tags"]
    if "status" in updates:
        if table == "legal_forms":
            mapped["is_active"] = updates["status"] != "archived"
        else:
            if updates["status"] == "indexed":
                mapped["status"] = "active"
            elif updates["status"] == "archived":
                mapped["status"] = "repealed"
            else:
                mapped["status"] = "pending"
    if "reviewStatus" in updates:
        mapped["review_status"] = updates["reviewStatus"]
    if "reviewNotes" in updates:
        mapped["review_notes"] = updates["reviewNotes"]
    return mapped


def _short_jurisdiction(value: str) -> str:
    normalized = value.upper()
    if normalized in {"THAILAND", "TH"}:
        return "TH"
    if normalized in {"LAOS", "LAO", "LA"}:
        return "LA"
    if normalized in {"INTERNATIONAL", "INTL"}:
        return "INTL"
    return normalized


def _db_jurisdiction(value: str) -> str:
    normalized = value.upper()
    if normalized == "TH":
        return "thailand"
    if normalized == "LA":
        return "laos"
    if normalized == "INTL":
        return "international"
    return value.lower()


def _ui_status(status: str) -> str:
    normalized = status.lower()
    if normalized in {"active", "indexed"}:
        return "indexed"
    if normalized in {"repealed", "archived"}:
        return "archived"
    if normalized in {"processing"}:
        return "processing"
    return "draft"
