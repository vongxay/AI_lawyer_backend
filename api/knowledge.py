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
from services.ingestion_service import (
    IngestionInput,
    LegalDocumentIngestionService,
    _article_from_section,
    _article_number_as_int,
)

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])
log = get_logger(__name__)
AdminUser = Annotated[CurrentUser, Depends(get_admin_user)]


class CreateKnowledgeDocument(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    type: str = "statute"
    lawCategory: str | None = None
    jurisdiction: str = "laos"
    year: int | None = None
    fullText: str = Field(default="", max_length=500_000)
    tags: list[str] = Field(default_factory=list)


class UpdateKnowledgeDocument(BaseModel):
    title: str | None = None
    lawCategory: str | None = None
    jurisdiction: str | None = None
    year: int | None = None
    status: str | None = None
    tags: list[str] | None = None
    reviewStatus: str | None = None
    reviewNotes: str | None = None


class ReviewDecision(BaseModel):
    action: str = Field(pattern="^(approve|reject)$")
    notes: str | None = Field(default=None, max_length=2000)
    jobId: str | None = None


CHUNK_AUDIT_COLUMNS = (
    "id, source_table, source_id, document_type, law_category, jurisdiction, title, "
    "chunk_index, section_ref, chapter_ref, law_no, article, language, token_count, "
    "status, review_status, metadata, created_at, updated_at"
)
CHUNK_AUDIT_LEGACY_COLUMNS = (
    "id, source_table, source_id, document_type, jurisdiction, title, chunk_index, "
    "section_ref, token_count, status, review_status, metadata, created_at, updated_at"
)
CHUNK_PREVIEW_COLUMNS = "id, chunk_index, content"


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


@router.get("/documents/{document_id}/chunks", summary="Audit chunks for a knowledge document")
async def get_document_chunks_audit(
    document_id: str,
    user: AdminUser,
    source_table: str = Query(default="laws", pattern="^(laws|cases|legal_forms)$"),
    limit: int = Query(default=200, ge=1, le=1000),
    audit_limit: int = Query(default=5000, ge=1, le=10000),
) -> dict:
    supabase = await get_supabase()
    if not supabase:
        return _empty_chunk_audit_response(document_id, source_table)

    rows = await _fetch_document_chunk_rows(
        supabase,
        source_table=source_table,
        document_id=document_id,
        audit_limit=audit_limit,
    )
    preview_rows = await _fetch_document_chunk_previews(
        supabase,
        source_table=source_table,
        document_id=document_id,
        limit=limit,
    )
    preview_by_key = {_chunk_key(row): str(row.get("content") or "") for row in preview_rows}
    items = [_map_chunk_audit_item(row, preview_by_key) for row in rows]
    document_structure = await _fetch_document_structure(supabase, source_table, document_id)
    qa = _assess_chunk_audit(
        items,
        source_table=source_table,
        audit_limit=audit_limit,
        document_structure=document_structure,
    )

    return {
        "documentId": document_id,
        "sourceTable": source_table,
        "total": len(rows),
        "returned": min(len(rows), limit),
        "auditLimit": audit_limit,
        "isTruncated": len(rows) >= audit_limit,
        "documentStructure": document_structure,
        "qa": qa,
        "items": items[:limit],
    }


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
            law_category=payload.lawCategory,
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
                await _sync_ingestion_job_review_status(
                    supabase,
                    document_id=document_id,
                    source_table=table,
                    review_status=review_status,
                    reviewed_at=reviewed_at,
                    notes=payload.notes,
                    job_id=payload.jobId,
                    user=user,
                )
                return {"status": review_status, "id": document_id, "source_table": table}
        except Exception as exc:
            log.warning("knowledge.review_table.failed", table=table, error=str(exc))

    return {"status": "not_found", "id": document_id}


async def _fetch_document_chunk_rows(
    supabase: Any,
    *,
    source_table: str,
    document_id: str,
    audit_limit: int,
) -> list[dict[str, Any]]:
    for columns in (CHUNK_AUDIT_COLUMNS, CHUNK_AUDIT_LEGACY_COLUMNS):
        try:
            result = await (
                supabase.table("document_chunks")
                .select(columns)
                .eq("source_table", source_table)
                .eq("source_id", document_id)
                .order("chunk_index")
                .limit(audit_limit)
                .execute()
            )
            return result.data or []
        except Exception as exc:
            log.warning(
                "knowledge.chunk_audit.select_failed",
                source_table=source_table,
                legacy=columns == CHUNK_AUDIT_LEGACY_COLUMNS,
                error=str(exc),
            )
    return []


async def _fetch_document_chunk_previews(
    supabase: Any,
    *,
    source_table: str,
    document_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    try:
        result = await (
            supabase.table("document_chunks")
            .select(CHUNK_PREVIEW_COLUMNS)
            .eq("source_table", source_table)
            .eq("source_id", document_id)
            .order("chunk_index")
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as exc:
        log.warning("knowledge.chunk_audit.preview_failed", source_table=source_table, error=str(exc))
        return []


async def _fetch_document_structure(
    supabase: Any,
    source_table: str,
    document_id: str,
) -> dict[str, Any] | None:
    try:
        result = await supabase.table(source_table).select("metadata").eq("id", document_id).limit(1).execute()
        row = (result.data or [None])[0]
        metadata = row.get("metadata") if isinstance(row, dict) else None
        if isinstance(metadata, dict) and isinstance(metadata.get("legal_structure"), dict):
            return metadata["legal_structure"]
    except Exception as exc:
        log.warning("knowledge.chunk_audit.structure_failed", source_table=source_table, error=str(exc))
    return None


def _map_chunk_audit_item(
    row: dict[str, Any],
    preview_by_key: dict[str, str],
) -> dict[str, Any]:
    metadata = _safe_dict(row.get("metadata"))
    section_ref = _str_or_none(row.get("section_ref"))
    stored_article = _str_or_none(row.get("article") or metadata.get("article"))
    detected_article = _article_from_section(section_ref)
    article_matches = _article_match_status(stored_article, detected_article)
    content = preview_by_key.get(_chunk_key(row))
    chunk_quality = _safe_dict(metadata.get("chunk_text_quality"))
    quality_warnings = chunk_quality.get("warnings") if isinstance(chunk_quality.get("warnings"), list) else []
    warnings = list(quality_warnings)

    if article_matches is False:
        if detected_article and stored_article:
            warnings.append("Article tag does not match section_ref")
        elif detected_article:
            warnings.append("Article detected in section_ref but missing from chunk metadata")
        else:
            warnings.append("Chunk has article metadata but no article heading was detected")

    return {
        "id": str(row.get("id") or ""),
        "chunkIndex": row.get("chunk_index"),
        "sectionRef": section_ref,
        "article": stored_article,
        "detectedArticle": detected_article,
        "articleSource": _str_or_none(metadata.get("article_source")),
        "articleMatches": article_matches,
        "chapterRef": _str_or_none(row.get("chapter_ref")),
        "lawNo": _str_or_none(row.get("law_no")),
        "language": _str_or_none(row.get("language") or metadata.get("language")),
        "tokenCount": row.get("token_count") or 0,
        "status": _str_or_none(row.get("status")),
        "reviewStatus": _str_or_none(row.get("review_status")),
        "qualityScore": chunk_quality.get("score") if isinstance(chunk_quality.get("score"), (int, float)) else None,
        "qualityLanguage": _str_or_none(chunk_quality.get("language")),
        "warnings": warnings,
        "contentPreview": _content_preview(content),
        "contentLength": len(content) if content is not None else None,
    }


def _assess_chunk_audit(
    items: list[dict[str, Any]],
    *,
    source_table: str,
    audit_limit: int,
    document_structure: dict[str, Any] | None,
) -> dict[str, Any]:
    total = len(items)
    article_numbers: list[int] = []
    section_refs: list[str] = []
    out_of_order = 0
    previous_article: int | None = None

    for item in items:
        section_ref = str(item.get("sectionRef") or "").strip()
        if section_ref and section_ref != "Preamble":
            section_refs.append(section_ref)

        article = _article_number_as_int(item.get("detectedArticle") or item.get("article"))
        if article is None:
            continue
        article_numbers.append(article)
        if previous_article is not None and article < previous_article:
            out_of_order += 1
        previous_article = article

    unique_articles = _ordered_unique_ints(article_numbers)
    max_article = max(unique_articles) if unique_articles else None
    missing_articles = _missing_articles(unique_articles, max_article) if source_table == "laws" else []
    duplicate_sections = _duplicates(section_refs)
    article_mismatch_count = sum(1 for item in items if item.get("articleMatches") is False)
    article_tagged_count = sum(1 for item in items if item.get("article") or item.get("detectedArticle"))
    status_counts = _count_values(item.get("status") for item in items)
    review_status_counts = _count_values(item.get("reviewStatus") for item in items)
    recommendations = _chunk_audit_recommendations(
        total=total,
        source_table=source_table,
        audit_limit=audit_limit,
        article_mismatch_count=article_mismatch_count,
        missing_articles=missing_articles,
        duplicate_sections=duplicate_sections,
        out_of_order=out_of_order,
        document_structure=document_structure,
        article_count=len(unique_articles),
    )
    correctness = _chunk_correctness(
        total=total,
        source_table=source_table,
        article_mismatch_count=article_mismatch_count,
        missing_articles=missing_articles,
        duplicate_sections=duplicate_sections,
        out_of_order=out_of_order,
    )

    return {
        "totalChunks": total,
        "articleTaggedChunks": article_tagged_count,
        "articleCount": len(unique_articles),
        "maxArticle": max_article,
        "missingArticles": missing_articles[:250],
        "missingArticleCount": len(missing_articles),
        "duplicateSections": duplicate_sections[:100],
        "duplicateSectionCount": len(duplicate_sections),
        "outOfOrderCount": out_of_order,
        "articleMismatchCount": article_mismatch_count,
        "statusCounts": status_counts,
        "reviewStatusCounts": review_status_counts,
        "recommendations": recommendations,
        "correctness": correctness,
    }


def _empty_chunk_audit_response(document_id: str, source_table: str) -> dict[str, Any]:
    return {
        "documentId": document_id,
        "sourceTable": source_table,
        "total": 0,
        "returned": 0,
        "auditLimit": 0,
        "isTruncated": False,
        "documentStructure": None,
        "qa": {
            "totalChunks": 0,
            "articleTaggedChunks": 0,
            "articleCount": 0,
            "maxArticle": None,
            "missingArticles": [],
            "missingArticleCount": 0,
            "duplicateSections": [],
            "duplicateSectionCount": 0,
            "outOfOrderCount": 0,
            "articleMismatchCount": 0,
            "statusCounts": {},
            "reviewStatusCounts": {},
            "recommendations": ["Database is not configured."],
            "correctness": {
                "status": "empty",
                "label": "No chunks",
                "severity": "warning",
                "isCorrect": False,
            },
        },
        "items": [],
    }


def _chunk_audit_recommendations(
    *,
    total: int,
    source_table: str,
    audit_limit: int,
    article_mismatch_count: int,
    missing_articles: list[str],
    duplicate_sections: list[str],
    out_of_order: int,
    document_structure: dict[str, Any] | None,
    article_count: int,
) -> list[str]:
    recommendations: list[str] = []
    if total == 0:
        recommendations.append("No chunks were found for this document in document_chunks.")
        return recommendations
    if total >= audit_limit:
        recommendations.append("Chunk audit reached the audit limit; increase audit_limit for a full large-document check.")
    if article_mismatch_count:
        recommendations.append(f"{article_mismatch_count} chunks have article metadata that should be reviewed.")
    if source_table == "laws" and missing_articles:
        recommendations.append(f"{len(missing_articles)} article numbers appear to be missing in the chunk sequence.")
    if duplicate_sections:
        recommendations.append(f"{len(duplicate_sections)} duplicate section references were detected.")
    if out_of_order:
        recommendations.append(f"{out_of_order} chunks appear out of article order.")
    if document_structure:
        expected = document_structure.get("article_count")
        if isinstance(expected, int) and expected != article_count:
            recommendations.append(
                f"Document metadata expected {expected} articles, but chunks show {article_count} unique articles."
            )
    if not recommendations:
        recommendations.append("Chunk article mapping looks consistent with the stored section references.")
    return recommendations


def _chunk_correctness(
    *,
    total: int,
    source_table: str,
    article_mismatch_count: int,
    missing_articles: list[str],
    duplicate_sections: list[str],
    out_of_order: int,
) -> dict[str, Any]:
    if total == 0:
        return {"status": "empty", "label": "No chunks", "severity": "warning", "isCorrect": False}

    needs_review = bool(
        article_mismatch_count
        or duplicate_sections
        or out_of_order
        or (source_table == "laws" and missing_articles)
    )
    if needs_review:
        return {
            "status": "needs_review",
            "label": "Needs review",
            "severity": "error",
            "isCorrect": False,
        }
    return {"status": "ok", "label": "Looks aligned", "severity": "success", "isCorrect": True}


def _article_match_status(stored_article: str | None, detected_article: str | None) -> bool | None:
    if not stored_article and not detected_article:
        return None
    if not stored_article or not detected_article:
        return False
    stored_number = _article_number_as_int(stored_article)
    detected_number = _article_number_as_int(detected_article)
    if stored_number is not None and detected_number is not None:
        return stored_number == detected_number
    return stored_article.strip().casefold() == detected_article.strip().casefold()


def _chunk_key(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("chunk_index") or "")


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _content_preview(content: str | None, max_length: int = 900) -> str | None:
    if content is None:
        return None
    compact = " ".join(content.split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[:max_length].rstrip()}..."


def _ordered_unique_ints(values: list[int]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _missing_articles(article_numbers: list[int], max_article: int | None) -> list[str]:
    if not max_article or max_article <= 1:
        return []
    present = set(article_numbers)
    return [str(article) for article in range(1, max_article + 1) if article not in present]


def _duplicates(values: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    duplicates: list[str] = []
    for value in values:
        counts[value] = counts.get(value, 0) + 1
        if counts[value] == 2:
            duplicates.append(value)
    return duplicates


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


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
    if "law_category" in source_updates:
        updates["law_category"] = str(source_updates["law_category"])

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


async def _sync_ingestion_job_review_status(
    supabase: Any,
    *,
    document_id: str,
    source_table: str,
    review_status: str,
    reviewed_at: str,
    notes: str | None,
    job_id: str | None,
    user: CurrentUser,
) -> None:
    """Keep admin ingestion history in step with knowledge review decisions."""
    try:
        query = (
            supabase.table("ingestion_jobs")
            .select("id, tenant_id, status, progress, config, result, errors")
            .limit(500)
        )
        if user.tenant_id:
            query = query.or_(f"tenant_id.is.null,tenant_id.eq.{user.tenant_id}")
        rows_result = await query.execute()
    except Exception as exc:
        log.warning("knowledge.review_ingestion_jobs.select_failed", error=str(exc))
        return

    rows = rows_result.data or []
    matched = 0
    for row in rows:
        current_result = row.get("result") if isinstance(row.get("result"), dict) else {}
        current_config = row.get("config") if isinstance(row.get("config"), dict) else {}
        result_document_id = str(current_result.get("document_id") or "")
        result_source_table = str(current_result.get("source_table") or current_config.get("source_table") or "")
        row_id = str(row.get("id") or "")
        if not row_id:
            continue

        if job_id:
            is_match = row_id == job_id or result_document_id == document_id
        else:
            is_match = result_document_id == document_id
        if not is_match:
            continue
        if result_source_table and result_source_table != source_table:
            continue

        next_result = {
            **current_result,
            "document_id": document_id,
            "source_table": source_table,
            "review_status": review_status,
            "reviewed_at": reviewed_at,
            "reviewed_by": user.sub,
        }
        if notes:
            next_result["review_notes"] = notes

        next_config = {
            **current_config,
            "review_status": review_status,
        }

        if review_status == "approved":
            next_job_status = "completed"
            next_result["status"] = "indexed"
            errors = row.get("errors") or []
        else:
            next_job_status = "rejected"
            next_result["status"] = "rejected"
            next_result.setdefault("error", "Rejected during knowledge review")
            errors = row.get("errors") or []

        updates = {
            "status": next_job_status,
            "progress": 100,
            "config": next_config,
            "result": next_result,
            "errors": errors,
            "updated_at": reviewed_at,
            "completed_at": reviewed_at,
        }

        try:
            result = await supabase.table("ingestion_jobs").update(updates).eq("id", row_id).execute()
            if result.data is not None:
                matched += 1
        except Exception as exc:
            log.warning("knowledge.review_ingestion_jobs.update_failed", job_id=row_id, error=str(exc))

    if matched == 0:
        log.info("knowledge.review_ingestion_jobs.no_match", document_id=document_id, source_table=source_table)


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
        "lawCategory": row.get("law_category") or (row.get("metadata") or {}).get("law_category"),
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
    if "lawCategory" in updates:
        mapped["law_category"] = updates["lawCategory"]
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
