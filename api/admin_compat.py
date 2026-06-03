"""
api/admin_compat.py
===================
Compatibility endpoints for the React admin services.

These routes keep the admin UI on FastAPI as the integration boundary while
reusing Supabase tables as the current source of truth.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from core.database import get_supabase
from core.logging import get_logger
from core.security import CurrentUser, get_admin_user

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
log = get_logger(__name__)
AdminUser = Annotated[CurrentUser, Depends(get_admin_user)]


@router.get("/sessions", summary="List case sessions")
async def list_sessions(user: AdminUser, limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []

    rows = await _select_rows(supabase, "case_sessions", limit=limit, tenant_id=user.tenant_id)
    users = await _lookup_users(supabase, [row.get("user_id") for row in rows])
    return [_normalise_session(row, users) for row in rows]


@router.patch("/sessions/{session_id}", summary="Update a case session status")
async def update_session_status(session_id: str, payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    new_status = str(payload.get("status") or "").strip()
    if not new_status:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="status is required.")

    updates: dict[str, Any] = {"status": new_status}
    if new_status == "closed":
        updates["closed_at"] = _now()

    row = await _update_row(supabase, "case_sessions", session_id, updates, tenant_id=user.tenant_id)
    if not row and "closed_at" in updates:
        row = await _update_row(supabase, "case_sessions", session_id, {"status": new_status}, tenant_id=user.tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return row


@router.get("/models", summary="List AI models")
async def list_models(user: AdminUser) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []
    rows = await _select_rows(
        supabase,
        "ai_models",
        limit=500,
        tenant_id=user.tenant_id,
        tenant_optional=True,
        order_by="is_default",
    )
    return [_normalise_model(row) for row in rows]


@router.post("/models", summary="Create an AI model")
async def create_model(payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    insert = dict(payload)
    insert.setdefault("model_type", "llm")
    insert.setdefault("is_active", True)
    insert.setdefault("is_default", False)
    insert.setdefault("config", {})
    if user.tenant_id and "tenant_id" not in insert:
        insert["tenant_id"] = user.tenant_id

    row = await _insert_row(supabase, "ai_models", insert)
    if not row:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Model could not be created.")
    return _normalise_model(row)


@router.patch("/models/{model_id}", summary="Update an AI model")
async def update_model(model_id: str, payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    row = await _update_row(supabase, "ai_models", model_id, dict(payload), tenant_id=user.tenant_id, tenant_optional=True)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")
    return _normalise_model(row)


@router.delete("/models/{model_id}", summary="Delete an AI model")
async def delete_model(model_id: str, user: AdminUser) -> dict[str, str]:
    supabase = await _require_supabase()
    deleted = await _delete_row(supabase, "ai_models", model_id, tenant_id=user.tenant_id, tenant_optional=True)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found.")
    return {"status": "deleted", "id": model_id}


@router.get("/citations", summary="List citation verification logs")
async def list_citations(user: AdminUser, limit: int = Query(default=200, ge=1, le=1000)) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []
    return await _select_rows(supabase, "citations_log", limit=limit, tenant_id=user.tenant_id)


@router.get("/citations/stats", summary="Get citation verification stats")
async def get_citation_stats(user: AdminUser) -> dict[str, Any]:
    supabase = await get_supabase()
    if not supabase:
        return _citation_stats([])
    rows = await _select_rows(supabase, "citations_log", limit=1000, tenant_id=user.tenant_id, order_by=None)
    return _citation_stats(rows)


@router.get("/case-graph", summary="List case citation edges")
async def list_case_graph(user: AdminUser, limit: int = Query(default=200, ge=1, le=1000)) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []
    rows = await _select_rows(supabase, "case_citations", limit=limit, tenant_id=None)
    case_numbers = await _lookup_case_numbers(
        supabase,
        [row.get("source_case_id") for row in rows] + [row.get("cited_case_id") for row in rows],
    )
    return [_normalise_case_citation(row, case_numbers) for row in rows]


@router.get("/cases", summary="List legal cases for graph management")
async def list_cases(user: AdminUser, limit: int = Query(default=200, ge=1, le=1000)) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []
    rows = await _select_rows(supabase, "cases", limit=limit, tenant_id=user.tenant_id, tenant_optional=True, order_by="year_be")
    return [_normalise_case(row) for row in rows]


@router.get("/case-graph/{case_id}/chain", summary="Get a precedent chain")
async def get_case_chain(case_id: str, user: AdminUser, depth: int = Query(default=3, ge=1, le=10)) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []
    try:
        result = await supabase.rpc("get_precedent_chain", {"start_case_id": case_id, "max_depth": depth}).execute()
        return result.data or []
    except Exception as exc:
        log.warning("admin.case_chain.failed", error=str(exc))
        return []


@router.post("/case-graph/citations", summary="Create a case citation edge")
async def create_case_citation(payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    row = await _insert_row(supabase, "case_citations", dict(payload))
    if not row:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Citation edge could not be created.")
    case_numbers = await _lookup_case_numbers(supabase, [row.get("source_case_id"), row.get("cited_case_id")])
    return _normalise_case_citation(row, case_numbers)


@router.delete("/case-graph/citations/{citation_id}", summary="Delete a case citation edge")
async def delete_case_citation(citation_id: str, user: AdminUser) -> dict[str, str]:
    supabase = await _require_supabase()
    deleted = await _delete_row(supabase, "case_citations", citation_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Citation edge not found.")
    return {"status": "deleted", "id": citation_id}


@router.get("/feedback", summary="List user feedback")
async def list_feedback(user: AdminUser, limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []
    rows = await _select_rows(supabase, "feedback", limit=limit, tenant_id=user.tenant_id)
    users = await _lookup_users(supabase, [row.get("user_id") for row in rows])
    return [_normalise_feedback(row, users) for row in rows]


@router.patch("/feedback/{feedback_id}", summary="Mark feedback processed")
async def update_feedback(feedback_id: str, payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    updates = dict(payload)
    if updates.get("is_processed") is True:
        updates.setdefault("processed_at", _now())
    row = await _update_row(supabase, "feedback", feedback_id, updates, tenant_id=user.tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found.")
    return row


@router.get("/documents", summary="List uploaded documents")
async def list_uploaded_documents(user: AdminUser, limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []
    return await _select_rows(supabase, "documents", limit=limit, tenant_id=user.tenant_id)


@router.delete("/documents/{document_id}", summary="Delete an uploaded document")
async def delete_uploaded_document(document_id: str, user: AdminUser) -> dict[str, str]:
    supabase = await _require_supabase()
    deleted = await _delete_row(supabase, "documents", document_id, tenant_id=user.tenant_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return {"status": "deleted", "id": document_id}


@router.get("/evidence", summary="List uploaded evidence")
async def list_uploaded_evidence(user: AdminUser, limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []
    return await _select_rows(supabase, "evidence", limit=limit, tenant_id=user.tenant_id)


@router.delete("/evidence/{evidence_id}", summary="Delete uploaded evidence")
async def delete_uploaded_evidence(evidence_id: str, user: AdminUser) -> dict[str, str]:
    supabase = await _require_supabase()
    deleted = await _delete_row(supabase, "evidence", evidence_id, tenant_id=user.tenant_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found.")
    return {"status": "deleted", "id": evidence_id}


@router.get("/billing/invoices", summary="List billing invoices")
async def list_billing_invoices(user: AdminUser, limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []
    rows = await _select_rows(supabase, "billing_invoices", limit=limit, tenant_id=user.tenant_id)
    tenants = await _lookup_tenants(supabase, [row.get("tenant_id") for row in rows])
    return [{**row, "tenant_name": tenants.get(str(row.get("tenant_id") or ""), {}).get("name")} for row in rows]


@router.get("/billing/plans", summary="List subscription plans")
async def list_billing_plans(user: AdminUser) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []
    return await _select_rows(supabase, "subscription_plans", limit=500, order_by="sort_order", desc=False)


@router.patch("/billing/invoices/{invoice_id}", summary="Update a billing invoice")
async def update_billing_invoice(invoice_id: str, payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    updates = dict(payload)
    if updates.get("status") == "paid":
        updates.setdefault("paid_at", _now())
    row = await _update_row(supabase, "billing_invoices", invoice_id, updates, tenant_id=user.tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invoice not found.")
    return row


@router.get("/cache", summary="List cache entries")
async def list_cache_entries(user: AdminUser, limit: int = Query(default=200, ge=1, le=1000)) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []
    return await _select_rows(
        supabase,
        "cache_entries",
        limit=limit,
        tenant_id=user.tenant_id,
        tenant_optional=True,
        order_by="hit_count",
    )


@router.delete("/cache/{cache_id}", summary="Invalidate a cache entry")
async def invalidate_cache_entry(cache_id: str, user: AdminUser) -> dict[str, str]:
    supabase = await _require_supabase()
    row = await _update_row(
        supabase,
        "cache_entries",
        cache_id,
        {"is_valid": False},
        tenant_id=user.tenant_id,
        tenant_optional=True,
    )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cache entry not found.")
    return {"status": "invalidated", "id": cache_id}


@router.post("/cache/purge", summary="Purge expired cache entries")
async def purge_cache(user: AdminUser) -> dict[str, int]:
    supabase = await get_supabase()
    if not supabase:
        return {"count": 0}
    try:
        result = await supabase.rpc("cleanup_expired_cache").execute()
        return {"count": int(result.data or 0)}
    except Exception as exc:
        log.warning("admin.cache_purge.failed", error=str(exc))
        return {"count": 0}


@router.get("/audit-log/stats", summary="Get audit log stats")
async def get_audit_stats(user: AdminUser) -> dict[str, int]:
    supabase = await get_supabase()
    if not supabase:
        return {"total24h": 0, "securityEvents": 0, "uniqueUsers": 0, "systemActions": 0}

    since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    rows = await _select_rows(supabase, "audit_log", limit=1000, tenant_id=user.tenant_id, order_by="created_at")
    recent = [row for row in rows if str(row.get("created_at") or "") >= since]
    unique_users = {row.get("user_id") for row in recent if row.get("user_id")}
    security_events = [
        row for row in recent
        if row.get("action") in {"flag_hallucination", "role_change", "delete_document"}
    ]
    return {
        "total24h": len(recent),
        "securityEvents": len(security_events),
        "uniqueUsers": len(unique_users),
        "systemActions": len([row for row in recent if not row.get("user_id")]),
    }


@router.patch("/expert-queue/{item_id}/assign", summary="Assign an expert review")
async def assign_expert_review(item_id: str, payload: dict[str, Any], user: AdminUser) -> dict[str, str]:
    supabase = await _require_supabase()
    reviewer_id = payload.get("reviewer_id")
    if not reviewer_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="reviewer_id is required.")

    updates = {"reviewer_id": reviewer_id, "assigned_at": _now(), "status": "in_review"}
    row = await _update_row(supabase, "expert_reviews", item_id, updates, tenant_id=user.tenant_id)
    if not row:
        row = await _update_row(
            supabase,
            "expert_reviews",
            item_id,
            {"reviewer_id": reviewer_id, "assigned_at": _now()},
            tenant_id=user.tenant_id,
        )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review item not found.")
    return {"status": "assigned", "id": item_id}


@router.patch("/expert-queue/{item_id}/resolve", summary="Resolve an expert review")
async def patch_resolve_expert_review(item_id: str, payload: dict[str, Any], user: AdminUser) -> dict[str, str]:
    supabase = await _require_supabase()
    updates = {
        "resolution": payload.get("resolution") or "",
        "status": "resolved",
        "reviewed_at": _now(),
    }
    row = await _update_row(supabase, "expert_reviews", item_id, updates, tenant_id=user.tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review item not found.")
    return {"status": "resolved", "id": item_id}


async def _require_supabase() -> Any:
    supabase = await get_supabase()
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase database is not configured.",
        )
    return supabase


async def _select_rows(
    supabase: Any,
    table: str,
    *,
    columns: str = "*",
    limit: int = 100,
    tenant_id: str | None = None,
    tenant_optional: bool = False,
    order_by: str | None = "created_at",
    desc: bool = True,
) -> list[dict[str, Any]]:
    tenant_modes = [True] if tenant_id else [False]
    if tenant_id and tenant_optional:
        tenant_modes = [True, False]

    for use_tenant in tenant_modes:
        for use_order in ([True, False] if order_by else [False]):
            try:
                query = supabase.table(table).select(columns).limit(limit)
                if use_tenant and tenant_id:
                    query = query.eq("tenant_id", tenant_id)
                if use_order and order_by:
                    query = query.order(order_by, desc=desc)
                result = await query.execute()
                return result.data or []
            except Exception as exc:
                log.warning(
                    "admin.compat_select.failed",
                    table=table,
                    use_tenant=use_tenant,
                    order_by=order_by if use_order else None,
                    error=str(exc),
                )
    return []


async def _insert_row(supabase: Any, table: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        result = await supabase.table(table).insert(payload).execute()
        data = result.data or []
        if isinstance(data, list):
            return data[0] if data else None
        return data
    except Exception as exc:
        log.warning("admin.compat_insert.failed", table=table, error=str(exc))
        return None


async def _update_row(
    supabase: Any,
    table: str,
    row_id: str,
    updates: dict[str, Any],
    *,
    tenant_id: str | None = None,
    tenant_optional: bool = False,
) -> dict[str, Any] | None:
    tenant_modes = [True] if tenant_id else [False]
    if tenant_id and tenant_optional:
        tenant_modes = [True, False]

    for use_tenant in tenant_modes:
        try:
            query = supabase.table(table).update(updates).eq("id", row_id)
            if use_tenant and tenant_id:
                query = query.eq("tenant_id", tenant_id)
            result = await query.execute()
            data = result.data or []
            if isinstance(data, list):
                return data[0] if data else None
            return data
        except Exception as exc:
            log.warning("admin.compat_update.failed", table=table, id=row_id, error=str(exc))
    return None


async def _delete_row(
    supabase: Any,
    table: str,
    row_id: str,
    *,
    tenant_id: str | None = None,
    tenant_optional: bool = False,
) -> bool:
    tenant_modes = [True] if tenant_id else [False]
    if tenant_id and tenant_optional:
        tenant_modes = [True, False]

    for use_tenant in tenant_modes:
        try:
            query = supabase.table(table).delete().eq("id", row_id)
            if use_tenant and tenant_id:
                query = query.eq("tenant_id", tenant_id)
            result = await query.execute()
            return bool(result.data)
        except Exception as exc:
            log.warning("admin.compat_delete.failed", table=table, id=row_id, error=str(exc))
    return False


async def _lookup_users(supabase: Any, user_ids: list[Any]) -> dict[str, dict[str, Any]]:
    ids = sorted({str(user_id) for user_id in user_ids if user_id})
    if not ids:
        return {}
    for columns in ("id, email, full_name", "id, email", "*"):
        try:
            result = await supabase.table("users").select(columns).in_("id", ids).execute()
            return {str(row.get("id")): row for row in (result.data or [])}
        except Exception as exc:
            log.warning("admin.lookup_users.failed", columns=columns, error=str(exc))
    return {}


async def _lookup_case_numbers(supabase: Any, case_ids: list[Any]) -> dict[str, str]:
    ids = sorted({str(case_id) for case_id in case_ids if case_id})
    if not ids:
        return {}
    try:
        result = await supabase.table("cases").select("id, case_no").in_("id", ids).execute()
        return {str(row.get("id")): str(row.get("case_no") or "") for row in (result.data or [])}
    except Exception as exc:
        log.warning("admin.lookup_cases.failed", error=str(exc))
        return {}


async def _lookup_tenants(supabase: Any, tenant_ids: list[Any]) -> dict[str, dict[str, Any]]:
    ids = sorted({str(tenant_id) for tenant_id in tenant_ids if tenant_id})
    if not ids:
        return {}
    try:
        result = await supabase.table("tenants").select("id, name").in_("id", ids).execute()
        return {str(row.get("id")): row for row in (result.data or [])}
    except Exception as exc:
        log.warning("admin.lookup_tenants.failed", error=str(exc))
        return {}


def _normalise_session(row: dict[str, Any], users: dict[str, dict[str, Any]]) -> dict[str, Any]:
    user_row = users.get(str(row.get("user_id") or ""), {})
    messages = row.get("messages") if isinstance(row.get("messages"), list) else []
    return {
        **row,
        "legal_case_id": row.get("legal_case_id") or row.get("case_id"),
        "message_count": row.get("message_count") or len(messages),
        "total_tokens": row.get("total_tokens") or 0,
        "total_cost_usd": _to_float(row.get("total_cost_usd")),
        "last_summary": row.get("last_summary") or row.get("facts_summary"),
        "user_email": user_row.get("email"),
        "user_name": user_row.get("full_name") or user_row.get("email"),
    }


def _normalise_model(row: dict[str, Any]) -> dict[str, Any]:
    created_at = row.get("created_at") or ""
    return {
        "id": str(row.get("id") or ""),
        "model_name": row.get("model_name") or "",
        "display_name": row.get("display_name") or row.get("model_name") or "",
        "provider": row.get("provider") or "",
        "model_type": row.get("model_type") or "llm",
        "is_active": row.get("is_active") is not False,
        "is_default": row.get("is_default") is True,
        "config": row.get("config") or {},
        "capabilities": row.get("capabilities"),
        "cost_per_1k_input": row.get("cost_per_1k_input"),
        "cost_per_1k_output": row.get("cost_per_1k_output"),
        "max_context_tokens": row.get("max_context_tokens"),
        "tenant_id": row.get("tenant_id"),
        "created_at": created_at,
        "updated_at": row.get("updated_at") or created_at,
    }


def _normalise_case_citation(row: dict[str, Any], case_numbers: dict[str, str]) -> dict[str, Any]:
    source_id = str(row.get("source_case_id") or "")
    cited_id = str(row.get("cited_case_id") or "")
    return {
        **row,
        "source_case_no": case_numbers.get(source_id),
        "cited_case_no": case_numbers.get(cited_id),
    }


def _normalise_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "case_no": row.get("case_no") or row.get("title") or "",
        "court": row.get("court") or "",
        "year_be": row.get("year_be") or row.get("year") or row.get("case_no_year"),
        "summary": row.get("summary"),
        "ruling": row.get("ruling") or row.get("full_text") or "",
        "outcome": row.get("outcome") or "",
        "ratio_decidendi": row.get("ratio_decidendi"),
    }


def _normalise_feedback(row: dict[str, Any], users: dict[str, dict[str, Any]]) -> dict[str, Any]:
    user_row = users.get(str(row.get("user_id") or ""), {})
    return {
        **row,
        "user_email": user_row.get("email"),
    }


def _citation_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    verified = len([row for row in rows if str(row.get("status") or "").lower() == "verified"])
    rejected = len([row for row in rows if str(row.get("status") or "").lower() == "rejected"])
    unverified = total - verified - rejected
    return {
        "total": total,
        "verified": verified,
        "rejected": rejected,
        "unverified": max(0, unverified),
        "accuracy": (verified / total * 100) if total else 0,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
