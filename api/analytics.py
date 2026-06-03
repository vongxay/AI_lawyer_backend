"""
api/analytics.py
================
Admin dashboard analytics endpoints.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from core.database import get_supabase
from core.logging import get_logger
from core.security import CurrentUser, get_admin_user

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])
log = get_logger(__name__)
AdminUser = Annotated[CurrentUser, Depends(get_admin_user)]


@router.get("/dashboard", summary="Get admin dashboard statistics")
async def get_dashboard_stats(user: AdminUser) -> dict[str, Any]:
    supabase = await get_supabase()
    if not supabase:
        return _empty_dashboard()

    total_documents = sum([
        await _count_table(supabase, "laws", tenant_id=user.tenant_id, tenant_optional=True),
        await _count_table(supabase, "cases", tenant_id=user.tenant_id, tenant_optional=True),
        await _count_table(supabase, "legal_forms", tenant_id=user.tenant_id, tenant_optional=True),
    ])

    active_models = await _count_table(
        supabase,
        "ai_models",
        filters={"is_active": True},
        tenant_id=user.tenant_id,
        tenant_optional=True,
    )
    active_users = await _count_active_users(supabase, tenant_id=user.tenant_id)
    expert_queue_count = await _count_table(
        supabase,
        "expert_reviews",
        filters={"status": "pending"},
        tenant_id=user.tenant_id,
    )

    return {
        "totalDocuments": total_documents,
        "activeModels": active_models,
        "activeUsers": active_users,
        "expertQueueCount": expert_queue_count,
        "sessionStats": await _get_session_stats(supabase, tenant_id=user.tenant_id),
    }


async def _count_active_users(supabase: Any, *, tenant_id: str | None) -> int:
    query = supabase.table("users").select("id").eq("is_active", True).limit(1000)
    if tenant_id:
        query = query.eq("tenant_id", tenant_id)

    try:
        result = await query.execute()
        return len(result.data or [])
    except Exception as exc:
        log.warning("analytics.active_users_count.failed", error=str(exc))
        return await _count_table(supabase, "users", tenant_id=tenant_id)


async def _count_table(
    supabase: Any,
    table: str,
    *,
    filters: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    tenant_optional: bool = False,
    limit: int = 1000,
) -> int:
    query = supabase.table(table).select("id").limit(limit)
    for key, value in (filters or {}).items():
        query = query.eq(key, value)
    if tenant_id:
        query = query.eq("tenant_id", tenant_id)

    try:
        result = await query.execute()
        return len(result.data or [])
    except Exception as exc:
        log.warning("analytics.count.failed", table=table, error=str(exc))

    if tenant_id and tenant_optional:
        try:
            query = supabase.table(table).select("id").limit(limit)
            for key, value in (filters or {}).items():
                query = query.eq(key, value)
            result = await query.execute()
            return len(result.data or [])
        except Exception as exc:
            log.warning("analytics.count_without_tenant.failed", table=table, error=str(exc))

    return 0


async def _get_session_stats(supabase: Any, *, tenant_id: str | None) -> dict[str, Any] | None:
    if tenant_id:
        try:
            result = await supabase.rpc(
                "get_session_stats",
                {"p_tenant_id": tenant_id, "p_days": 30},
            ).execute()
            data = result.data or []
            if isinstance(data, list) and data:
                return _normalise_session_stats(data[0])
            if isinstance(data, dict):
                return _normalise_session_stats(data)
        except Exception as exc:
            log.warning("analytics.session_stats_rpc.failed", error=str(exc))

    return await _session_stats_from_audit_log(supabase, tenant_id=tenant_id)


async def _session_stats_from_audit_log(supabase: Any, *, tenant_id: str | None) -> dict[str, Any] | None:
    query = supabase.table("audit_log").select("*").limit(500)
    if tenant_id:
        query = query.eq("tenant_id", tenant_id)

    try:
        result = await query.execute()
    except Exception as exc:
        log.warning("analytics.audit_log_stats.failed", error=str(exc))
        return None

    rows = result.data or []
    if not rows:
        return None

    confidences = [_to_float(row.get("confidence")) for row in rows if row.get("confidence") is not None]
    latencies = [
        _to_float(row.get("latency_ms") or row.get("processing_time_ms"))
        for row in rows
        if row.get("latency_ms") is not None or row.get("processing_time_ms") is not None
    ]
    escalations = [
        row
        for row in rows
        if row.get("escalated") is True or row.get("escalated_to_expert") is True
    ]
    citation_rejections = sum(int(row.get("citation_rejection_count") or 0) for row in rows)
    costs = [_to_float(row.get("cost_usd")) for row in rows if row.get("cost_usd") is not None]
    session_ids = {row.get("session_id") for row in rows if row.get("session_id")}

    return {
        "total_sessions": len(session_ids) or len(rows),
        "total_queries": len(rows),
        "avg_confidence": _avg(confidences),
        "avg_latency_ms": _avg(latencies),
        "escalation_rate": round(len(escalations) / len(rows), 4),
        "citation_accuracy": round(max(0.0, 1.0 - (citation_rejections / max(len(rows), 1))), 4),
        "total_cost_usd": round(sum(costs), 6),
    }


def _normalise_session_stats(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_sessions": int(row.get("total_sessions") or 0),
        "total_queries": int(row.get("total_queries") or 0),
        "avg_confidence": _to_float(row.get("avg_confidence")),
        "avg_latency_ms": _to_float(row.get("avg_latency_ms")),
        "escalation_rate": _to_float(row.get("escalation_rate")),
        "citation_accuracy": _to_float(row.get("citation_accuracy")),
        "total_cost_usd": _to_float(row.get("total_cost_usd")),
    }


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _to_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _empty_dashboard() -> dict[str, Any]:
    return {
        "totalDocuments": 0,
        "activeModels": 0,
        "activeUsers": 0,
        "expertQueueCount": 0,
        "sessionStats": None,
    }
