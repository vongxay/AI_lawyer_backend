"""
api/admin_compat.py
===================
Compatibility endpoints for the React admin services.

These routes keep the admin UI on FastAPI as the integration boundary while
reusing Supabase tables as the current source of truth.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from core.config import get_settings
from core.database import get_supabase, ping_redis, ping_supabase
from core.logging import get_logger
from core.security import CurrentUser, get_admin_user

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
log = get_logger(__name__)
AdminUser = Annotated[CurrentUser, Depends(get_admin_user)]


@router.get("/health", summary="Admin-scoped backend health check")
async def get_admin_health(user: AdminUser) -> dict[str, Any]:
    settings = get_settings()
    started = time.perf_counter()
    redis_ok = await ping_redis()
    redis_latency = int((time.perf_counter() - started) * 1000)
    redis_available = _redis_available(redis_ok, settings)
    started = time.perf_counter()
    supabase_ok = await ping_supabase()
    supabase_latency = int((time.perf_counter() - started) * 1000)

    configured_llm = _provider_configured(settings.model_reasoning)
    status_value = "ok" if supabase_ok and configured_llm and redis_available else "degraded"
    return {
        "status": status_value,
        "version": settings.app_version,
        "environment": settings.app_env,
        "services": {
            "fastapi": {"ok": True, "latency_ms": 0},
            "redis": {
                "ok": redis_available,
                "connected": redis_ok,
                "mode": _redis_mode(redis_ok, settings),
                "latency_ms": redis_latency,
            },
            "supabase": {"ok": supabase_ok, "latency_ms": supabase_latency},
            "anthropic": {"ok": bool(settings.anthropic_api_key)},
            "openai": {"ok": bool(settings.openai_api_key)},
        },
    }


@router.get("/ops/agents", summary="Get AI agent runtime overview")
async def get_agent_monitor(user: AdminUser) -> dict[str, Any]:
    supabase = await get_supabase()
    audit_rows = await _fetch_audit_rows(supabase, user=user, limit=1000)
    agents = _build_agent_statuses(audit_rows)
    timeline = _agent_calls_timeline(audit_rows)
    status_distribution = _status_distribution(agents, key="status", labels={
        "running": "Running",
        "idle": "Idle",
        "error": "Error",
    })
    return {
        "agents": agents,
        "callsTimeline": timeline,
        "agentLoadData": [
            {"name": item["name"].split(" ")[0], "calls": item["last24hCalls"]}
            for item in agents
        ],
        "statusDistribution": status_distribution,
    }


@router.get("/ops/system", summary="Get system and infrastructure monitor data")
async def get_system_monitor(user: AdminUser) -> dict[str, Any]:
    settings = get_settings()
    supabase = await get_supabase()
    audit_rows = await _fetch_audit_rows(supabase, user=user, limit=1000)
    expert_rows = await _select_rows(supabase, "expert_reviews", limit=500, tenant_id=user.tenant_id) if supabase else []

    started = time.perf_counter()
    redis_ok = await ping_redis()
    redis_latency = int((time.perf_counter() - started) * 1000)
    redis_available = _redis_available(redis_ok, settings)
    started = time.perf_counter()
    supabase_ok = await ping_supabase()
    supabase_latency = int((time.perf_counter() - started) * 1000)

    services = [
        _service_status("FastAPI", True, 0, "Backend API is accepting authenticated admin requests.", "API"),
        _service_status("Supabase", supabase_ok, supabase_latency, "Database and Auth project connectivity.", "Database"),
        _service_status(
            "Redis / Local Cache",
            redis_available,
            redis_latency,
            "Redis connectivity, or in-memory cache fallback when Redis is optional.",
            "Cache",
        ),
        _service_status("Anthropic", bool(settings.anthropic_api_key), None, "Claude provider configured for LLM agents.", "AI"),
        _service_status(
            "OpenAI Embeddings",
            bool(settings.openai_api_key),
            None,
            "Embedding provider for vector RAG. Keyword retrieval is used when unavailable.",
            "AI",
        ),
    ]
    agents = _build_agent_statuses(audit_rows)
    avg_latency = _average([_to_float(row.get("latency_ms") or row.get("processing_time_ms")) for row in audit_rows])
    error_rows = [row for row in audit_rows if row.get("success") is False or row.get("error_message")]
    queue_depth = len([row for row in expert_rows if str(row.get("status") or "pending") in {"pending", "in_review"}])

    return {
        "services": services,
        "agentMetrics": [
            {
                "name": item["name"],
                "model": item["model"],
                "avgLatency": item["avgLatency"],
                "p95Latency": item.get("p95Latency", item["avgLatency"]),
                "errorRate": item.get("errorRate", "0%"),
                "invocations": item["last24hCalls"],
                "status": "healthy" if item["status"] == "running" else ("degraded" if item["status"] == "idle" else "down"),
            }
            for item in agents
        ],
        "summary": {
            "status": "Operational" if not any(s["status"] == "down" for s in services) else "Degraded",
            "avgApiLatencyMs": round(avg_latency or max(supabase_latency, redis_latency), 1),
            "queueDepth": queue_depth,
            "errorRate": round(len(error_rows) / max(len(audit_rows), 1), 4),
            "storageUsedGb": None,
        },
        "uptimeData": _last_24h_buckets(audit_rows, value_key=None, value_name="uptime", default=100),
        "latencyData": _latency_buckets(audit_rows, supabase_latency=supabase_latency, redis_latency=redis_latency),
        "resourceUsage": [
            {"name": "Database", "value": 1 if supabase_ok else 0, "color": "hsl(var(--primary))"},
            {"name": "Cache", "value": 1 if redis_available else 0, "color": "hsl(var(--sky-info))"},
            {"name": "LLM", "value": 1 if settings.anthropic_api_key else 0, "color": "hsl(var(--amber-warning))"},
        ],
    }


@router.get("/ops/security", summary="Get security posture and event overview")
async def get_security_overview(user: AdminUser) -> dict[str, Any]:
    settings = get_settings()
    supabase = await get_supabase()
    rows = await _fetch_audit_rows(supabase, user=user, limit=1000)
    events = _security_events(rows)
    unresolved = len([event for event in events if not event["resolved"]])
    high_risk = len([event for event in events if event["severity"] in {"critical", "high"} and not event["resolved"]])
    score = max(0, 100 - high_risk * 10 - unresolved * 2)
    return {
        "score": score,
        "events": events,
        "threatTrend": _security_trend(events),
        "severityDistribution": _status_distribution(events, key="severity", labels={
            "critical": "Critical",
            "high": "High",
            "medium": "Medium",
            "low": "Low",
        }),
        "policies": _security_policies(settings),
        "stats": {
            "activeThreats": unresolved,
            "failedLogins24h": len([event for event in events if event["type"] == "failed_login"]),
            "mfaAdoption": None,
        },
    }


@router.get("/ops/notifications", summary="Get operational admin notifications")
async def get_notifications(user: AdminUser) -> dict[str, Any]:
    settings = get_settings()
    supabase = await get_supabase()
    audit_rows = await _fetch_audit_rows(supabase, user=user, limit=500)
    expert_rows = await _select_rows(supabase, "expert_reviews", limit=100, tenant_id=user.tenant_id) if supabase else []
    rag_health = await _rag_health_snapshot(supabase, tenant_id=user.tenant_id) if supabase else {}
    notifications = _build_notifications(settings, audit_rows, expert_rows, rag_health)
    return {
        "items": notifications,
        "settings": {
            "critical": True,
            "warning": True,
            "info": True,
            "success": False,
            "email": True,
            "slack": False,
        },
    }


@router.get("/ops/roles", summary="Get role definitions and assigned user counts")
async def get_roles_overview(user: AdminUser) -> dict[str, Any]:
    supabase = await get_supabase()
    users = await _select_rows(supabase, "users", limit=1000, tenant_id=user.tenant_id) if supabase else []
    roles = _role_definitions(users)
    return {"items": roles, "scopes": _scope_categories()}


@router.get("/ops/settings", summary="Get sanitized application settings")
async def get_settings_overview(user: AdminUser) -> dict[str, Any]:
    return {"sections": _settings_sections()}


@router.patch("/ops/settings/{setting_key}", summary="Validate an admin setting update")
async def update_setting_preview(setting_key: str, payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    value = payload.get("value")
    return {
        "status": "accepted",
        "key": setting_key,
        "value": _mask_secret(str(value)) if _looks_secret_key(setting_key) else value,
        "note": "Runtime environment settings are read from deployment configuration; persist this change in your secret manager or .env.",
    }


@router.post("/ops/actions/{action}", summary="Run supported admin maintenance actions")
async def run_ops_action(action: str, user: AdminUser) -> dict[str, Any]:
    if action == "purge-cache":
        supabase = await get_supabase()
        if supabase:
            try:
                result = await supabase.rpc("cleanup_expired_cache").execute()
                return {"status": "completed", "count": int(result.data or 0)}
            except Exception as exc:
                log.warning("admin.ops_action.cache_purge.failed", error=str(exc))
        return {"status": "skipped", "count": 0}
    return {
        "status": "queued",
        "action": action,
        "note": "This action requires an operations worker or deployment-level automation.",
    }


@router.get("/ops/pii", summary="Get PII redaction monitoring overview")
async def get_pii_overview(user: AdminUser) -> dict[str, Any]:
    supabase = await get_supabase()
    rows = await _fetch_audit_rows(supabase, user=user, limit=1000)
    records = _pii_records(rows)
    rules = _pii_rules(records)
    return {
        "records": records,
        "rules": rules,
        "detectionTrend": _pii_trend(records),
        "typeDistribution": _pii_type_distribution(records),
        "stats": {
            "detected24h": len(records),
            "autoMaskedRate": _auto_mask_rate(records),
            "pendingReview": len([row for row in records if row["status"] == "flagged"]),
            "compliance": 98.2 if records else 100,
        },
    }


@router.get("/prompt-versions", summary="List prompt versions")
async def list_prompt_versions(
    user: AdminUser,
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    supabase = await get_supabase()
    if not supabase:
        return {"items": [], "agents": _default_prompt_agents()}

    rows = await _select_rows(
        supabase,
        "prompt_versions",
        limit=limit,
        tenant_id=user.tenant_id,
        order_by="version",
    )
    users = await _lookup_users(supabase, [row.get("created_by") for row in rows])
    agents = await _prompt_agent_options(supabase, rows, tenant_id=user.tenant_id)
    active_by_agent = {
        str(row.get("agent_name") or ""): int(row.get("version") or 0)
        for row in rows
        if row.get("is_active") is True
    }
    return {
        "items": [_normalise_prompt_version(row, users, active_by_agent) for row in rows],
        "agents": agents,
    }


@router.post("/prompt-versions", summary="Create a prompt version")
async def create_prompt_version(payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    tenant_id = payload.get("tenant_id") or user.tenant_id
    if not tenant_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="tenant_id is required.")

    agent_name = str(payload.get("agent_name") or payload.get("agent") or "").strip()
    prompt_text = str(payload.get("prompt_text") or payload.get("promptPreview") or payload.get("system_prompt") or "").strip()
    if not agent_name or not prompt_text:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="agent and prompt text are required.")

    version = _prompt_version_number(payload.get("version"), fallback=await _next_prompt_version(supabase, tenant_id, agent_name))
    is_active = payload.get("is_active") is True or str(payload.get("status") or "").lower() == "active"
    if is_active:
        await _deactivate_prompt_versions(supabase, tenant_id, agent_name)

    insert = {
        "tenant_id": tenant_id,
        "agent_name": agent_name,
        "version": version,
        "prompt_text": prompt_text,
        "system_prompt": payload.get("system_prompt") or prompt_text,
        "temperature": _to_float(payload.get("temperature") if payload.get("temperature") is not None else 0.7),
        "max_tokens": int(payload.get("max_tokens") or payload.get("maxTokens") or 4096),
        "is_active": is_active,
        "created_by": user.sub,
        "notes": payload.get("notes") or payload.get("changes") or "",
        "test_results": payload.get("test_results") or payload.get("testResults") or {},
    }
    row = await _insert_row(supabase, "prompt_versions", insert)
    if not row:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Prompt version could not be created.")
    users = await _lookup_users(supabase, [row.get("created_by")])
    return _normalise_prompt_version(row, users, {agent_name: version if is_active else 0})


@router.patch("/prompt-versions/{version_id}", summary="Update a prompt version")
async def update_prompt_version(version_id: str, payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    current = await _get_row(supabase, "prompt_versions", version_id, tenant_id=user.tenant_id)
    if not current:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt version not found.")

    agent_name = str(payload.get("agent_name") or payload.get("agent") or current.get("agent_name") or "").strip()
    is_active = payload.get("is_active") is True or str(payload.get("status") or "").lower() == "active"
    if is_active:
        await _deactivate_prompt_versions(supabase, str(current.get("tenant_id") or user.tenant_id), agent_name)

    updates: dict[str, Any] = {"updated_at": _now()}
    if agent_name:
        updates["agent_name"] = agent_name
    if payload.get("version") is not None:
        updates["version"] = _prompt_version_number(payload.get("version"), fallback=int(current.get("version") or 1))
    if payload.get("prompt_text") is not None or payload.get("promptPreview") is not None or payload.get("system_prompt") is not None:
        prompt_text = str(payload.get("prompt_text") or payload.get("promptPreview") or payload.get("system_prompt") or "").strip()
        if not prompt_text:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="prompt text cannot be empty.")
        updates["prompt_text"] = prompt_text
        updates["system_prompt"] = payload.get("system_prompt") or prompt_text
    if payload.get("notes") is not None or payload.get("changes") is not None:
        updates["notes"] = payload.get("notes") if payload.get("notes") is not None else payload.get("changes")
    if payload.get("temperature") is not None:
        updates["temperature"] = _to_float(payload.get("temperature"))
    if payload.get("max_tokens") is not None or payload.get("maxTokens") is not None:
        updates["max_tokens"] = int(payload.get("max_tokens") or payload.get("maxTokens"))
    if payload.get("test_results") is not None or payload.get("testResults") is not None:
        updates["test_results"] = payload.get("test_results") or payload.get("testResults") or {}
    if payload.get("status") is not None or payload.get("is_active") is not None:
        updates["is_active"] = is_active

    row = await _update_row(supabase, "prompt_versions", version_id, updates, tenant_id=user.tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt version not found.")
    users = await _lookup_users(supabase, [row.get("created_by")])
    return _normalise_prompt_version(row, users, {str(row.get("agent_name") or ""): int(row.get("version") or 0) if row.get("is_active") else 0})


@router.patch("/prompt-versions/{version_id}/restore", summary="Activate a prompt version")
async def restore_prompt_version(version_id: str, user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    current = await _get_row(supabase, "prompt_versions", version_id, tenant_id=user.tenant_id)
    if not current:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt version not found.")

    tenant_id = str(current.get("tenant_id") or user.tenant_id)
    agent_name = str(current.get("agent_name") or "")
    await _deactivate_prompt_versions(supabase, tenant_id, agent_name)
    row = await _update_row(supabase, "prompt_versions", version_id, {"is_active": True, "updated_at": _now()}, tenant_id=user.tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt version not found.")
    users = await _lookup_users(supabase, [row.get("created_by")])
    return _normalise_prompt_version(row, users, {agent_name: int(row.get("version") or 0)})


@router.delete("/prompt-versions/{version_id}", summary="Delete a prompt version")
async def delete_prompt_version(version_id: str, user: AdminUser) -> dict[str, str]:
    supabase = await _require_supabase()
    deleted = await _delete_row(supabase, "prompt_versions", version_id, tenant_id=user.tenant_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prompt version not found.")
    return {"status": "deleted", "id": version_id}


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


@router.post("/cases", summary="Create a legal case")
async def create_case(payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    insert = _case_payload(payload, user=user)
    row = await _insert_row(supabase, "cases", insert)
    if not row:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Case could not be created.")
    return _normalise_case(row)


@router.patch("/cases/{case_id}", summary="Update a legal case")
async def update_case(case_id: str, payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    row = await _update_row(supabase, "cases", case_id, _case_payload(payload, user=user, partial=True), tenant_id=user.tenant_id, tenant_optional=True)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found.")
    return _normalise_case(row)


@router.delete("/cases/{case_id}", summary="Delete a legal case")
async def delete_case(case_id: str, user: AdminUser) -> dict[str, str]:
    supabase = await _require_supabase()
    await _delete_case_citations(supabase, case_id)
    deleted = await _delete_row(supabase, "cases", case_id, tenant_id=user.tenant_id, tenant_optional=True)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found.")
    return {"status": "deleted", "id": case_id}


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


@router.post("/documents", summary="Register uploaded document metadata")
async def create_uploaded_document(payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    tenant_id = payload.get("tenant_id") or user.tenant_id
    if not tenant_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="tenant_id is required.")
    row = await _insert_row(supabase, "documents", _document_payload(payload, user=user, tenant_id=str(tenant_id)))
    if not row:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Document could not be created.")
    return row


@router.post("/documents/upload", summary="Upload a document file into private storage")
async def upload_admin_document(
    user: AdminUser,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    doc_type: str | None = Form(default=None, alias="type"),
    source: str | None = Form(default=None),
    confidential: bool = Form(default=False),
    legal_case_id: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    tenant_id: str | None = Form(default=None),
) -> dict[str, Any]:
    supabase = await _require_supabase()
    resolved_tenant_id = tenant_id or user.tenant_id
    if not resolved_tenant_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="tenant_id is required.")

    uploaded = await _store_admin_upload(
        supabase,
        file=file,
        kind="document",
        tenant_id=str(resolved_tenant_id),
        title=title,
    )
    payload = _document_payload(
        {
            "title": title or uploaded["file_name"],
            "file_path": uploaded["file_path"],
            "file_type": uploaded["content_type"],
            "file_size_bytes": uploaded["file_size_bytes"],
            "type": doc_type,
            "source": source,
            "is_privileged": confidential,
            "legal_case_id": legal_case_id,
            "session_id": session_id,
        },
        user=user,
        tenant_id=str(resolved_tenant_id),
    )
    payload["file_name"] = uploaded["file_name"]
    payload["checksum"] = uploaded["checksum"]
    row = await _insert_row(supabase, "documents", payload)
    if not row:
        await _remove_storage_object(supabase, bucket=_storage_bucket("document"), path=uploaded["file_path"])
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Document could not be created.")
    return row


@router.get("/documents/{document_id}/signed-url", summary="Create a signed download URL for a private document")
async def get_document_signed_url(document_id: str, user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    row = await _get_row(supabase, "documents", document_id, tenant_id=user.tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return await _create_storage_signed_url(supabase, bucket=_storage_bucket("document"), path=str(row.get("file_path") or ""))


@router.patch("/documents/{document_id}", summary="Update uploaded document metadata")
async def update_uploaded_document(document_id: str, payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    row = await _update_row(supabase, "documents", document_id, _document_payload(payload, user=user, tenant_id=user.tenant_id, partial=True), tenant_id=user.tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return row


@router.delete("/documents/{document_id}", summary="Delete an uploaded document")
async def delete_uploaded_document(document_id: str, user: AdminUser) -> dict[str, str]:
    supabase = await _require_supabase()
    row = await _get_row(supabase, "documents", document_id, tenant_id=user.tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    deleted = await _delete_row(supabase, "documents", document_id, tenant_id=user.tenant_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    await _remove_storage_object(supabase, bucket=_storage_bucket("document"), path=str(row.get("file_path") or ""))
    return {"status": "deleted", "id": document_id}


@router.get("/evidence", summary="List uploaded evidence")
async def list_uploaded_evidence(user: AdminUser, limit: int = Query(default=100, ge=1, le=1000)) -> list[dict[str, Any]]:
    supabase = await get_supabase()
    if not supabase:
        return []
    return await _select_rows(supabase, "evidence", limit=limit, tenant_id=user.tenant_id)


@router.post("/evidence", summary="Register uploaded evidence metadata")
async def create_uploaded_evidence(payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    tenant_id = payload.get("tenant_id") or user.tenant_id
    if not tenant_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="tenant_id is required.")
    row = await _insert_row(supabase, "evidence", _evidence_payload(payload, user=user, tenant_id=str(tenant_id)))
    if not row:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Evidence could not be created.")
    return row


@router.post("/evidence/upload", summary="Upload evidence into private storage")
async def upload_admin_evidence(
    user: AdminUser,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    evidence_type: str | None = Form(default=None, alias="type"),
    source: str | None = Form(default=None),
    is_original: bool = Form(default=True),
    legal_case_id: str | None = Form(default=None),
    session_id: str | None = Form(default=None),
    tenant_id: str | None = Form(default=None),
) -> dict[str, Any]:
    supabase = await _require_supabase()
    resolved_tenant_id = tenant_id or user.tenant_id
    if not resolved_tenant_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="tenant_id is required.")

    uploaded = await _store_admin_upload(
        supabase,
        file=file,
        kind="evidence",
        tenant_id=str(resolved_tenant_id),
        title=title,
    )
    payload = _evidence_payload(
        {
            "title": title or uploaded["file_name"],
            "file_path": uploaded["file_path"],
            "mimeType": uploaded["content_type"],
            "file_size_bytes": uploaded["file_size_bytes"],
            "type": evidence_type,
            "source": source,
            "is_original": is_original,
            "legal_case_id": legal_case_id,
            "session_id": session_id,
        },
        user=user,
        tenant_id=str(resolved_tenant_id),
    )
    payload["file_name"] = uploaded["file_name"]
    row = await _insert_row(supabase, "evidence", payload)
    if not row:
        await _remove_storage_object(supabase, bucket=_storage_bucket("evidence"), path=uploaded["file_path"])
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Evidence could not be created.")
    return row


@router.get("/evidence/{evidence_id}/signed-url", summary="Create a signed download URL for private evidence")
async def get_evidence_signed_url(evidence_id: str, user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    row = await _get_row(supabase, "evidence", evidence_id, tenant_id=user.tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found.")
    return await _create_storage_signed_url(supabase, bucket=_storage_bucket("evidence"), path=str(row.get("file_path") or ""))


@router.patch("/evidence/{evidence_id}", summary="Update uploaded evidence metadata")
async def update_uploaded_evidence(evidence_id: str, payload: dict[str, Any], user: AdminUser) -> dict[str, Any]:
    supabase = await _require_supabase()
    row = await _update_row(supabase, "evidence", evidence_id, _evidence_payload(payload, user=user, tenant_id=user.tenant_id, partial=True), tenant_id=user.tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found.")
    return row


@router.delete("/evidence/{evidence_id}", summary="Delete uploaded evidence")
async def delete_uploaded_evidence(evidence_id: str, user: AdminUser) -> dict[str, str]:
    supabase = await _require_supabase()
    row = await _get_row(supabase, "evidence", evidence_id, tenant_id=user.tenant_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found.")
    deleted = await _delete_row(supabase, "evidence", evidence_id, tenant_id=user.tenant_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence not found.")
    await _remove_storage_object(supabase, bucket=_storage_bucket("evidence"), path=str(row.get("file_path") or ""))
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


async def _get_row(
    supabase: Any,
    table: str,
    row_id: str,
    *,
    tenant_id: str | None = None,
) -> dict[str, Any] | None:
    for use_tenant in ([True, False] if tenant_id else [False]):
        try:
            query = supabase.table(table).select("*").eq("id", row_id).limit(1)
            if use_tenant and tenant_id:
                query = query.eq("tenant_id", tenant_id)
            result = await query.execute()
            data = result.data or []
            if isinstance(data, list):
                return data[0] if data else None
            return data
        except Exception as exc:
            log.warning("admin.compat_get.failed", table=table, id=row_id, error=str(exc))
    return None


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


def _normalise_prompt_version(
    row: dict[str, Any],
    users: dict[str, dict[str, Any]],
    active_by_agent: dict[str, int],
) -> dict[str, Any]:
    agent_name = str(row.get("agent_name") or "")
    version_number = int(row.get("version") or 1)
    created_by = str(row.get("created_by") or "")
    author = users.get(created_by, {}).get("full_name") or users.get(created_by, {}).get("email") or "Admin"
    prompt_text = row.get("prompt_text") or row.get("system_prompt") or ""
    is_active = row.get("is_active") is True
    test_results = row.get("test_results") if isinstance(row.get("test_results"), dict) else {}
    return {
        "id": str(row.get("id") or ""),
        "agent": agent_name,
        "agent_name": agent_name,
        "version": f"v{version_number}",
        "versionNumber": version_number,
        "author": author,
        "createdBy": created_by,
        "date": _prompt_date(row.get("updated_at") or row.get("created_at")),
        "changes": row.get("notes") or ("Active production prompt" if is_active else "Draft prompt"),
        "status": _prompt_status(row, active_by_agent),
        "promptPreview": prompt_text,
        "prompt_text": prompt_text,
        "system_prompt": row.get("system_prompt") or prompt_text,
        "temperature": _to_float(row.get("temperature")),
        "maxTokens": int(row.get("max_tokens") or 4096),
        "tokens": _estimate_tokens(prompt_text),
        "testScore": _prompt_test_score(test_results),
        "testResults": test_results,
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


async def _prompt_agent_options(
    supabase: Any,
    prompt_rows: list[dict[str, Any]],
    *,
    tenant_id: str | None,
) -> list[str]:
    names = {str(row.get("agent_name") or "").strip() for row in prompt_rows if row.get("agent_name")}
    config_rows = await _select_rows(
        supabase,
        "agent_configs",
        columns="agent_name, display_name",
        limit=500,
        tenant_id=tenant_id,
        tenant_optional=True,
        order_by="agent_name",
        desc=False,
    )
    for row in config_rows:
        names.add(str(row.get("agent_name") or row.get("display_name") or "").strip())
    names.update(_default_prompt_agents())
    return sorted([name for name in names if name])


def _default_prompt_agents() -> list[str]:
    return ["research", "reasoning", "citation", "evidence", "intake"]


async def _next_prompt_version(supabase: Any, tenant_id: str, agent_name: str) -> int:
    rows = await _select_rows(
        supabase,
        "prompt_versions",
        columns="agent_name, version",
        limit=100,
        tenant_id=tenant_id,
        order_by="version",
    )
    versions = [
        int(row.get("version") or 0)
        for row in rows
        if str(row.get("agent_name") or "") == agent_name
    ]
    return max(versions or [0]) + 1


async def _deactivate_prompt_versions(supabase: Any, tenant_id: str, agent_name: str) -> None:
    try:
        await (
            supabase.table("prompt_versions")
            .update({"is_active": False, "updated_at": _now()})
            .eq("tenant_id", tenant_id)
            .eq("agent_name", agent_name)
            .execute()
        )
    except Exception as exc:
        log.warning("admin.prompt_deactivate.failed", agent=agent_name, error=str(exc))


def _prompt_version_number(value: Any, *, fallback: int = 1) -> int:
    if value is None or value == "":
        return max(1, fallback)
    if isinstance(value, int):
        return max(1, value)
    text = str(value).strip().lower().lstrip("v")
    if not text:
        return max(1, fallback)
    try:
        return max(1, int(float(text)))
    except ValueError:
        return max(1, fallback)


def _prompt_status(row: dict[str, Any], active_by_agent: dict[str, int]) -> str:
    if row.get("is_active") is True:
        return "active"
    agent_name = str(row.get("agent_name") or "")
    if active_by_agent.get(agent_name):
        return "previous"
    return "draft"


def _prompt_test_score(test_results: dict[str, Any]) -> float:
    for key in ("score", "avg_score", "pass_rate", "quality_score"):
        if key in test_results:
            value = _to_float(test_results.get(key))
            return round(value * 100, 1) if 0 < value <= 1 else round(value, 1)
    return 0


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text.split()) * 1.3))


def _prompt_date(value: Any) -> str:
    parsed = _parse_dt(value)
    if not parsed:
        return ""
    return parsed.strftime("%Y-%m-%d %H:%M")


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


def _case_payload(payload: dict[str, Any], *, user: CurrentUser, partial: bool = False) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    case_no = payload.get("case_no") or payload.get("caseNo")
    if case_no is not None:
        updates["case_no"] = str(case_no).strip()
    if payload.get("court") is not None:
        updates["court"] = str(payload.get("court") or "").strip()
    year = payload.get("year_be") if payload.get("year_be") is not None else payload.get("year")
    if year is not None:
        updates["year_be"] = int(year or 0)
        updates["year_ce"] = int(year or 0)
    summary = payload.get("summary")
    if summary is not None:
        updates["summary"] = str(summary)
        updates.setdefault("ruling", str(payload.get("ruling") or summary or "-"))
    if payload.get("ruling") is not None:
        updates["ruling"] = str(payload.get("ruling") or "-")
    if payload.get("outcome") is not None:
        updates["outcome"] = _case_outcome(payload.get("outcome"))
        updates["outcome_notes"] = str(payload.get("outcome"))
    if payload.get("ratio_decidendi") is not None:
        updates["ratio_decidendi"] = payload.get("ratio_decidendi")
    if payload.get("source_url") is not None:
        updates["source_url"] = payload.get("source_url")
    updates["updated_at"] = _now()

    if partial:
        return {key: value for key, value in updates.items() if value is not None}

    updates.setdefault("case_no", str(payload.get("caseNo") or "Untitled case").strip())
    updates.setdefault("court", str(payload.get("court") or "Court").strip())
    updates.setdefault("jurisdiction", "laos")
    updates.setdefault("ruling", str(payload.get("ruling") or payload.get("summary") or "-"))
    updates.setdefault("summary", payload.get("summary") or "")
    updates.setdefault("outcome", _case_outcome(payload.get("outcome")))
    updates.setdefault("doc_type", "case_law")
    updates.setdefault("status", "active")
    if user.tenant_id:
        updates.setdefault("tenant_id", user.tenant_id)
    if user.sub:
        updates.setdefault("ingested_by", user.sub)
    return updates


def _document_payload(payload: dict[str, Any], *, user: CurrentUser, tenant_id: str, partial: bool = False) -> dict[str, Any]:
    file_name = payload.get("file_name") or payload.get("title")
    updates: dict[str, Any] = {}
    if file_name is not None:
        updates["file_name"] = _safe_file_name(str(file_name))
    if payload.get("file_path") is not None:
        updates["file_path"] = str(payload.get("file_path"))
    if payload.get("file_type") is not None or payload.get("mimeType") is not None:
        updates["file_type"] = str(payload.get("file_type") or payload.get("mimeType") or "application/octet-stream")
    if payload.get("file_size_bytes") is not None or payload.get("fileSizeBytes") is not None:
        updates["file_size_bytes"] = int(payload.get("file_size_bytes") or payload.get("fileSizeBytes") or 0)
    if payload.get("doc_category") is not None or payload.get("type") is not None:
        updates["doc_category"] = _document_category(payload.get("doc_category") or payload.get("type"))
    if payload.get("is_privileged") is not None or payload.get("confidential") is not None:
        updates["is_privileged"] = payload.get("is_privileged") if payload.get("is_privileged") is not None else payload.get("confidential") is True
    if payload.get("legal_case_id") is not None:
        updates["legal_case_id"] = payload.get("legal_case_id")
    if payload.get("session_id") is not None:
        updates["session_id"] = payload.get("session_id")
    updates["updated_at"] = _now()

    if partial:
        return updates

    updates.setdefault("tenant_id", tenant_id)
    updates.setdefault("uploaded_by", user.sub)
    updates.setdefault("file_name", _safe_file_name(str(file_name or "document")))
    updates.setdefault("file_path", _metadata_file_path(tenant_id, updates["file_name"]))
    updates.setdefault("file_type", str(payload.get("mimeType") or "application/octet-stream"))
    updates.setdefault("doc_category", _document_category(payload.get("type") or payload.get("doc_category")))
    updates.setdefault("is_analyzed", False)
    updates.setdefault("is_privileged", False)
    return updates


def _evidence_payload(payload: dict[str, Any], *, user: CurrentUser, tenant_id: str, partial: bool = False) -> dict[str, Any]:
    file_name = payload.get("file_name") or payload.get("title")
    updates: dict[str, Any] = {}
    if file_name is not None:
        updates["file_name"] = _safe_file_name(str(file_name))
    if payload.get("file_path") is not None:
        updates["file_path"] = str(payload.get("file_path"))
    if payload.get("evidence_type") is not None or payload.get("type") is not None or payload.get("mimeType") is not None:
        updates["evidence_type"] = _evidence_type(payload.get("evidence_type") or payload.get("type"), payload.get("mimeType"))
    if payload.get("file_size_bytes") is not None or payload.get("fileSizeBytes") is not None:
        updates["file_size_bytes"] = int(payload.get("file_size_bytes") or payload.get("fileSizeBytes") or 0)
    if payload.get("legal_case_id") is not None:
        updates["legal_case_id"] = payload.get("legal_case_id")
    if payload.get("session_id") is not None:
        updates["session_id"] = payload.get("session_id")
    if payload.get("is_original") is not None:
        updates["is_original"] = payload.get("is_original") is True
    updates["updated_at"] = _now()

    if partial:
        return updates

    updates.setdefault("tenant_id", tenant_id)
    updates.setdefault("uploaded_by", user.sub)
    updates.setdefault("file_name", _safe_file_name(str(file_name or "evidence")))
    updates.setdefault("file_path", _metadata_file_path(tenant_id, updates["file_name"]))
    updates.setdefault("evidence_type", _evidence_type(payload.get("type"), payload.get("mimeType")))
    updates.setdefault("is_original", True)
    updates.setdefault("is_processed", False)
    return updates


async def _delete_case_citations(supabase: Any, case_id: str) -> None:
    for column in ("source_case_id", "cited_case_id"):
        try:
            await supabase.table("case_citations").delete().eq(column, case_id).execute()
        except Exception as exc:
            log.warning("admin.case_citation_cleanup.failed", column=column, case_id=case_id, error=str(exc))


def _case_outcome(value: Any) -> str:
    text = str(value or "").lower()
    if "defendant" in text:
        return "defendant_won"
    if "dismiss" in text:
        return "dismissed"
    if "settle" in text:
        return "settled"
    if "partial" in text:
        return "partial"
    if "plaintiff" in text:
        return "plaintiff_won"
    return "unknown"


def _document_category(value: Any) -> str:
    text = str(value or "contract").lower()
    if text == "case_law":
        return "case_law"
    if text in {"statute", "regulation", "form", "contract", "evidence"}:
        return text
    return "contract"


def _evidence_type(value: Any, mime_type: Any = None) -> str:
    text = f"{value or ''} {mime_type or ''}".lower()
    if "audio" in text:
        return "audio"
    if "video" in text:
        return "video"
    if "email" in text:
        return "email"
    if "image" in text or any(ext in text for ext in (".png", ".jpg", ".jpeg")):
        return "image"
    if "word" in text or ".doc" in text:
        return "document_word"
    if "contract" in text:
        return "contract"
    return "document_pdf"


def _safe_file_name(value: str) -> str:
    stripped = value.strip() or "upload"
    return stripped.replace("/", "_").replace("\\", "_")


def _metadata_file_path(tenant_id: str, file_name: str) -> str:
    return f"admin-metadata/{tenant_id}/{int(time.time())}-{_safe_file_name(file_name)}"


async def _store_admin_upload(
    supabase: Any,
    *,
    file: UploadFile,
    kind: str,
    tenant_id: str,
    title: str | None,
) -> dict[str, Any]:
    settings = get_settings()
    original_name = _safe_file_name(file.filename or title or kind)
    content_type = _normalise_upload_mime(file.content_type, original_name)
    allowed = _storage_allowed_mime(kind)
    if content_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported {kind} file type: {content_type}.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds {settings.max_upload_size_mb}MB limit.",
        )

    stored_name = _safe_file_name(title or original_name)
    object_path = _storage_file_path(tenant_id, kind, original_name)
    bucket = _storage_bucket(kind)
    try:
        await supabase.storage.from_(bucket).upload(
            object_path,
            content,
            {
                "content-type": content_type,
                "cache-control": "3600",
                "upsert": "false",
            },
        )
    except Exception as exc:
        log.warning("admin.storage_upload.failed", bucket=bucket, path=object_path, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="File could not be uploaded to Supabase Storage.",
        ) from exc

    return {
        "file_name": stored_name,
        "file_path": object_path,
        "content_type": content_type,
        "file_size_bytes": len(content),
        "checksum": hashlib.sha256(content).hexdigest(),
    }


async def _create_storage_signed_url(supabase: Any, *, bucket: str, path: str) -> dict[str, Any]:
    settings = get_settings()
    if not path or path.startswith("admin-metadata/"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stored file is not available for download.")
    try:
        result = await supabase.storage.from_(bucket).create_signed_url(path, settings.storage_signed_url_ttl_seconds)
    except Exception as exc:
        log.warning("admin.storage_signed_url.failed", bucket=bucket, path=path, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Signed download URL could not be created.",
        ) from exc

    signed_url = (
        result.get("signedURL")
        or result.get("signedUrl")
        or result.get("signed_url")
        or result.get("url")
    )
    if not signed_url:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Signed download URL was not returned.")
    return {
        "signed_url": signed_url,
        "signedUrl": signed_url,
        "expires_in": settings.storage_signed_url_ttl_seconds,
        "expiresIn": settings.storage_signed_url_ttl_seconds,
        "path": path,
        "bucket": bucket,
    }


async def _remove_storage_object(supabase: Any, *, bucket: str, path: str) -> None:
    if not path or path.startswith("admin-metadata/"):
        return
    try:
        await supabase.storage.from_(bucket).remove([path])
    except Exception as exc:
        log.warning("admin.storage_remove.failed", bucket=bucket, path=path, error=str(exc))


def _storage_bucket(kind: str) -> str:
    settings = get_settings()
    return settings.supabase_evidence_bucket if kind == "evidence" else settings.supabase_documents_bucket


def _storage_file_path(tenant_id: str, kind: str, file_name: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    safe_tenant = _safe_path_segment(tenant_id)
    safe_name = _safe_path_segment(file_name)
    return f"{safe_tenant}/{kind}/{day}/{uuid.uuid4().hex}-{safe_name}"


def _safe_path_segment(value: str) -> str:
    return _safe_file_name(value).replace(" ", "_")


def _normalise_upload_mime(content_type: str | None, file_name: str) -> str:
    mime = (content_type or "").split(";")[0].strip().lower()
    suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    if mime in {"", "application/octet-stream"}:
        mime = {
            "pdf": "application/pdf",
            "doc": "application/msword",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "txt": "text/plain",
            "csv": "text/csv",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "mp4": "video/mp4",
            "zip": "application/zip",
        }.get(suffix, mime or "application/octet-stream")
    if mime == "audio/x-wav":
        return "audio/wav"
    if mime == "application/x-zip-compressed":
        return "application/zip"
    return mime


def _storage_allowed_mime(kind: str) -> set[str]:
    if kind == "evidence":
        return {
            "application/pdf",
            "image/jpeg",
            "image/png",
            "image/webp",
            "audio/mpeg",
            "audio/wav",
            "video/mp4",
            "application/zip",
        }
    return {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/csv",
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


def _provider_configured(model: str | None) -> bool:
    settings = get_settings()
    name = str(model or "").lower()
    if name.startswith("claude"):
        return bool(settings.anthropic_api_key)
    if name.startswith(("gpt", "o1", "o3", "o4")):
        return bool(settings.openai_api_key)
    return True


def _redis_required(settings: Any) -> bool:
    return bool(settings.redis_required or settings.is_production())


def _redis_available(redis_ok: bool, settings: Any) -> bool:
    return redis_ok or not _redis_required(settings)


def _redis_mode(redis_ok: bool, settings: Any) -> str:
    if redis_ok:
        return "redis"
    return "unavailable" if _redis_required(settings) else "memory_fallback"


async def _fetch_audit_rows(
    supabase: Any | None,
    *,
    user: CurrentUser,
    limit: int,
) -> list[dict[str, Any]]:
    if not supabase:
        return []

    for order_column in ("created_at", "ts"):
        try:
            query = supabase.table("audit_log").select("*").limit(limit)
            if user.tenant_id:
                query = query.eq("tenant_id", user.tenant_id)
            result = await query.order(order_column, desc=True).execute()
            return result.data or []
        except Exception as exc:
            log.warning("admin.audit_fetch.failed", order_column=order_column, error=str(exc))
    return []


def _build_agent_statuses(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settings = get_settings()
    specs = [
        ("research", "Legal Research Agent", "researchAgent", settings.model_research),
        ("reasoning", "IRAC Reasoning Agent", "reasoningAgent", settings.model_reasoning),
        ("verification", "Citation Verification", "citationAgent", settings.model_verification),
        ("document", "Document Analysis", "documentAgent", settings.model_document),
        ("evidence", "Evidence Analyzer", "evidenceAgent", settings.model_evidence),
        ("risk", "Risk & Strategy", "riskAgent", settings.model_risk),
        ("pii", "PII Redaction", "piiAgent", "rule-based"),
    ]
    now = datetime.now(timezone.utc)
    items: list[dict[str, Any]] = []
    for agent_id, name, role, model in specs:
        rows = [
            row for row in audit_rows
            if agent_id in _agent_tokens(row) or (agent_id == "reasoning" and "irac" in _agent_tokens(row))
        ]
        recent_rows = [
            row for row in rows
            if (_parse_dt(row.get("created_at")) or now) >= now - timedelta(days=1)
        ]
        latencies = [
            _to_float(row.get("latency_ms") or row.get("processing_time_ms"))
            for row in rows
            if row.get("latency_ms") is not None or row.get("processing_time_ms") is not None
        ]
        errors = [row for row in rows if row.get("success") is False or row.get("error_message")]
        configured = model == "rule-based" or _provider_configured(model)
        status_value = "error" if not configured else ("running" if recent_rows else "idle")
        success_rate = 100.0 if not rows else round((1 - len(errors) / max(len(rows), 1)) * 100, 1)
        avg_latency_ms = _average(latencies)
        p95_ms = _percentile(latencies, 0.95)
        items.append({
            "id": agent_id,
            "name": name,
            "role": role,
            "status": status_value,
            "activeRequests": 0,
            "avgLatency": f"{round(avg_latency_ms / 1000, 2)}s" if avg_latency_ms else "-",
            "p95Latency": f"{round(p95_ms / 1000, 2)}s" if p95_ms else "-",
            "successRate": success_rate,
            "last24hCalls": len(recent_rows),
            "lastError": None if configured else f"{_provider_name(model)} API key is not configured.",
            "model": model,
            "memoryUsage": "-",
            "uptime": "runtime",
            "errorRate": f"{round((len(errors) / max(len(rows), 1)) * 100, 1)}%",
        })
    return items


def _agent_tokens(row: dict[str, Any]) -> set[str]:
    values: list[str] = []
    raw = row.get("agents_used") or row.get("agents_invoked")
    if isinstance(raw, list):
        values.extend(str(item) for item in raw)
    elif raw:
        values.append(str(raw))
    for key in ("agent", "action", "model_used"):
        if row.get(key):
            values.append(str(row[key]))
    joined = " ".join(values).lower().replace("_agent", "").replace(" agent", "")
    tokens = {part for part in re_split_nonword(joined) if part}
    if "legal" in tokens and "research" in tokens:
        tokens.add("research")
    if "citation" in tokens:
        tokens.add("verification")
    if "irac" in tokens:
        tokens.add("reasoning")
    if "strategy" in tokens:
        tokens.add("risk")
    return tokens


def re_split_nonword(value: str) -> list[str]:
    import re
    return re.split(r"[^a-z0-9]+", value)


def _agent_calls_timeline(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, int]] = {}
    now = datetime.now(timezone.utc)
    for hour in range(0, 24, 4):
        label = f"{hour:02d}"
        buckets[label] = {"time": label, "Research": 0, "IRAC": 0, "Citation": 0, "Others": 0}
    for row in audit_rows:
        created = _parse_dt(row.get("created_at"))
        if not created or created < now - timedelta(days=1):
            continue
        bucket = f"{(created.hour // 4) * 4:02d}"
        tokens = _agent_tokens(row)
        if "research" in tokens:
            buckets[bucket]["Research"] += 1
        elif "reasoning" in tokens or "irac" in tokens:
            buckets[bucket]["IRAC"] += 1
        elif "verification" in tokens or "citation" in tokens:
            buckets[bucket]["Citation"] += 1
        else:
            buckets[bucket]["Others"] += 1
    return list(buckets.values())


def _status_distribution(
    rows: list[dict[str, Any]],
    *,
    key: str,
    labels: dict[str, str],
) -> list[dict[str, Any]]:
    colors = {
        "running": "hsl(43, 74%, 49%)",
        "idle": "hsl(220, 15%, 70%)",
        "error": "hsl(0, 72%, 51%)",
        "critical": "hsl(var(--destructive))",
        "high": "hsl(var(--amber-warning))",
        "medium": "hsl(var(--sky-info))",
        "low": "hsl(var(--muted-foreground))",
    }
    return [
        {"name": label, "value": len([row for row in rows if row.get(key) == raw]), "color": colors.get(raw, "hsl(var(--muted-foreground))")}
        for raw, label in labels.items()
    ]


def _service_status(
    name: str,
    ok: bool,
    latency_ms: int | None,
    details: str,
    category: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "healthy" if ok else "down",
        "latency": f"{latency_ms}ms" if latency_ms is not None else "-",
        "latencyMs": latency_ms,
        "uptime": "runtime",
        "lastCheck": "just now",
        "details": details,
        "category": category,
    }


def _last_24h_buckets(
    rows: list[dict[str, Any]],
    *,
    value_key: str | None,
    value_name: str,
    default: float = 0,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    buckets = [
        {"time": f"{hour:02d}", value_name: default}
        for hour in range(0, 24, 4)
    ]
    if value_key is None:
        return buckets
    lookup = {bucket["time"]: [] for bucket in buckets}
    for row in rows:
        created = _parse_dt(row.get("created_at"))
        if not created or created < now - timedelta(days=1):
            continue
        lookup[f"{(created.hour // 4) * 4:02d}"].append(_to_float(row.get(value_key)))
    for bucket in buckets:
        vals = [value for value in lookup[bucket["time"]] if value]
        bucket[value_name] = round(sum(vals) / len(vals), 2) if vals else default
    return buckets


def _latency_buckets(
    rows: list[dict[str, Any]],
    *,
    supabase_latency: int,
    redis_latency: int,
) -> list[dict[str, Any]]:
    base = _last_24h_buckets(rows, value_key="latency_ms", value_name="api", default=0)
    for bucket in base:
        bucket["db"] = supabase_latency
        bucket["cache"] = redis_latency
    return base


def _security_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    security_actions = {
        "flag_hallucination",
        "role_change",
        "delete_document",
        "failed_login",
        "auth_failed",
        "rate_limit_exceeded",
        "citation_rejected",
    }
    for row in rows:
        action = str(row.get("action") or row.get("event") or "").lower()
        if action not in security_actions and not row.get("error_message"):
            continue
        severity = _security_severity(action, row)
        created_at = row.get("created_at") or ""
        events.append({
            "id": str(row.get("id") or f"{action}-{len(events)}"),
            "type": action or "backend_error",
            "severity": severity,
            "description": str(row.get("error_message") or row.get("detail") or action.replace("_", " ").title()),
            "ip": str(row.get("ip_address") or row.get("ip") or "-"),
            "user": str(row.get("user_email") or row.get("user_id") or "system"),
            "timestamp": _time_ago(created_at),
            "createdAt": created_at,
            "action": "Review" if severity in {"critical", "high"} else "Monitor",
            "resolved": row.get("resolved") is True or row.get("success") is True,
        })
    return events[:100]


def _security_severity(action: str, row: dict[str, Any]) -> str:
    if action in {"role_change", "delete_document"}:
        return "high"
    if action in {"failed_login", "auth_failed", "rate_limit_exceeded"}:
        return "medium"
    if row.get("error_message"):
        return "medium"
    return "low"


def _security_trend(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    days = _last_7_day_labels()
    buckets = {day: {"day": day, "threats": 0, "blocked": 0} for day in days}
    for event in events:
        created = _parse_dt(event.get("createdAt"))
        day = _day_label(created) if created else days[-1]
        if day in buckets:
            buckets[day]["threats"] += 1
            if event.get("resolved"):
                buckets[day]["blocked"] += 1
    return list(buckets.values())


def _security_policies(settings: Any) -> list[dict[str, Any]]:
    return [
        {
            "title": "Authentication",
            "items": [
                {"label": "JWT Expiry", "value": f"{settings.jwt_access_ttl_seconds // 60} minutes", "ok": settings.jwt_access_ttl_seconds <= 3600},
                {"label": "Refresh Token", "value": f"{settings.jwt_refresh_ttl_seconds // 86400} days", "ok": settings.jwt_refresh_ttl_seconds <= 604800},
                {"label": "Admin Role Gate", "value": "FastAPI enforced", "ok": True},
                {"label": "MFA Required", "value": "Supabase setting", "ok": False},
            ],
        },
        {
            "title": "API Security",
            "items": [
                {"label": "Rate Limiting", "value": f"{settings.rate_limit_per_minute}/min", "ok": settings.rate_limit_per_minute > 0},
                {"label": "CORS Origins", "value": f"{len(settings.cors_origins)} configured", "ok": "*" not in settings.cors_origins},
                {"label": "Request Logging", "value": "Enabled", "ok": True},
                {"label": "Production Secrets", "value": settings.app_env, "ok": settings.app_env != "production" or settings.jwt_secret != "change-me-in-production-min-32-chars!!"},
            ],
        },
        {
            "title": "Data Protection",
            "items": [
                {"label": "PII Redaction", "value": "Enabled before LLM", "ok": True},
                {"label": "Service Role Exposure", "value": "Backend only", "ok": True},
                {"label": "RLS", "value": "Verify via Supabase advisors", "ok": True},
                {"label": "Embeddings", "value": "OpenAI key configured" if settings.openai_api_key else "Keyword-only fallback", "ok": bool(settings.openai_api_key)},
            ],
        },
    ]


async def _rag_health_snapshot(supabase: Any, *, tenant_id: str | None) -> dict[str, Any]:
    try:
        query = supabase.table("document_chunks").select("id, embedding, status, review_status").limit(5000)
        if tenant_id:
            query = query.or_(f"tenant_id.is.null,tenant_id.eq.{tenant_id}")
        result = await query.execute()
        rows = result.data or []
        embedded = len([row for row in rows if row.get("embedding")])
        approved = len([row for row in rows if row.get("status") == "active" and row.get("review_status") == "approved"])
        return {"chunks": len(rows), "embedded": embedded, "approved": approved}
    except Exception as exc:
        log.warning("admin.rag_snapshot.failed", error=str(exc))
        return {}


def _build_notifications(
    settings: Any,
    audit_rows: list[dict[str, Any]],
    expert_rows: list[dict[str, Any]],
    rag_health: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    pending_reviews = len([row for row in expert_rows if str(row.get("status") or "pending") in {"pending", "in_review"}])
    if pending_reviews:
        items.append(_notification("expert-queue", "warning", "Expert review queue has pending items", f"{pending_reviews} items need human review.", "Expert Queue"))
    if not settings.openai_api_key:
        items.append(_notification("embeddings", "warning", "Vector embeddings are disabled", "OpenAI embedding key is not configured, so RAG is using keyword fallback.", "RAG"))
    if rag_health and rag_health.get("chunks", 0) and not rag_health.get("embedded", 0):
        items.append(_notification("rag-coverage", "warning", "RAG chunks have no embeddings", f"{rag_health.get('approved', 0)} approved chunks are available, but embedding coverage is 0%.", "Knowledge"))
    failed = [row for row in audit_rows if row.get("success") is False or row.get("error_message")]
    if failed:
        items.append(_notification("audit-errors", "critical", "Backend errors detected", f"{len(failed)} recent audit entries contain errors.", "Audit"))
    if not items:
        items.append(_notification("system-ok", "success", "Admin backend is connected", "No critical operational notifications were found.", "System", read=True))
    return items


def _notification(
    item_id: str,
    kind: str,
    title: str,
    message: str,
    source: str,
    *,
    read: bool = False,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": kind,
        "title": title,
        "message": message,
        "source": source,
        "time": "just now",
        "read": read,
    }


def _scope_categories() -> dict[str, Any]:
    return {
        "admin": {"label": "Administration", "scopes": ["admin:read", "admin:write", "admin:delete"]},
        "model": {"label": "AI Models", "scopes": ["model:config", "model:deploy"]},
        "knowledge": {"label": "Knowledge Base", "scopes": ["knowledge:read", "knowledge:write", "knowledge:delete"]},
        "users": {"label": "User Management", "scopes": ["tenant:manage", "user:manage", "user:read"]},
        "operations": {"label": "Operations", "scopes": ["audit:read", "analytics:read", "system:read", "expert:review", "feedback:manage"]},
        "client": {"label": "Client Actions", "scopes": ["conversation:read", "legal:query", "document:upload", "evidence:upload", "memory:read"]},
    }


def _role_definitions(users: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in users:
        counts[str(row.get("role") or "client")] = counts.get(str(row.get("role") or "client"), 0) + 1
    role_specs = {
        "super_admin": ("Super Admin", "Full platform administration.", ["admin:read", "admin:write", "admin:delete", "model:config", "model:deploy", "knowledge:read", "knowledge:write", "knowledge:delete", "tenant:manage", "user:manage", "audit:read", "analytics:read", "system:read"], "bg-destructive/15 text-destructive"),
        "admin": ("Admin", "Tenant administration and operations.", ["admin:read", "admin:write", "model:config", "knowledge:read", "knowledge:write", "user:manage", "audit:read", "analytics:read", "system:read", "feedback:manage"], "bg-amber-warning/15 text-amber-warning"),
        "lawyer": ("Lawyer", "Legal professional workflow access.", ["legal:query", "conversation:read", "document:upload", "evidence:upload", "memory:read", "expert:review"], "bg-primary/15 text-primary"),
        "expert_reviewer": ("Expert Reviewer", "Human review and correction workflow.", ["expert:review", "feedback:manage", "conversation:read", "legal:query"], "bg-emerald-500/15 text-emerald-600"),
        "auditor": ("Auditor", "Read-only operational and audit access.", ["audit:read", "analytics:read", "system:read", "knowledge:read"], "bg-muted text-muted-foreground"),
        "client": ("Client", "End-user legal query access.", ["legal:query", "conversation:read", "document:upload", "evidence:upload", "memory:read", "feedback:write"], "bg-sky-info/15 text-sky-info"),
    }
    return [
        {
            "id": key,
            "name": label,
            "description": desc,
            "usersCount": counts.get(key, 0),
            "permissions": perms,
            "color": color,
            "createdAt": "",
            "isSystem": True,
        }
        for key, (label, desc, perms, color) in role_specs.items()
    ]


def _settings_sections() -> list[dict[str, Any]]:
    settings = get_settings()
    return [
        {
            "title": "Application",
            "icon": "Settings",
            "items": [
                _setting_item("APP_ENV", "Environment", "Current runtime environment.", settings.app_env, "select", False, ["development", "staging", "production"]),
                _setting_item("APP_VERSION", "Version", "Backend application version.", settings.app_version, "text", False),
                _setting_item("LOG_LEVEL", "Log Level", "Structured logging verbosity.", settings.log_level, "select", True, ["DEBUG", "INFO", "WARNING", "ERROR"]),
            ],
        },
        {
            "title": "Security",
            "icon": "Shield",
            "items": [
                _setting_item("JWT_ACCESS_TTL_SECONDS", "JWT Access TTL", "Access-token lifetime in seconds.", str(settings.jwt_access_ttl_seconds), "text", True),
                _setting_item("RATE_LIMIT_PER_MINUTE", "Rate Limit", "Requests allowed per minute.", str(settings.rate_limit_per_minute), "text", True),
                _setting_item("ALLOWED_ORIGINS", "Allowed Origins", "Browser origins allowed by CORS.", ", ".join(settings.cors_origins), "text", True),
            ],
        },
        {
            "title": "AI Providers",
            "icon": "Key",
            "items": [
                _setting_item("ANTHROPIC_API_KEY", "Anthropic API Key", "Claude provider key.", _mask_secret(settings.anthropic_api_key), "secret", True),
                _setting_item("OPENAI_API_KEY", "OpenAI API Key", "Embedding/vector provider key.", _mask_secret(settings.openai_api_key), "secret", True),
                _setting_item("MODEL_REASONING", "Reasoning Model", "Primary IRAC reasoning model.", settings.model_reasoning, "text", True),
                _setting_item("MODEL_RESEARCH", "Research Model", "Legal research model.", settings.model_research, "text", True),
            ],
        },
        {
            "title": "Knowledge & Uploads",
            "icon": "Database",
            "items": [
                _setting_item("RAG_TOP_K", "RAG Top K", "Number of retrieval candidates.", str(settings.rag_top_k), "text", True),
                _setting_item("MAX_UPLOAD_SIZE_MB", "Max Upload Size", "Maximum legal document upload size.", f"{settings.max_upload_size_mb} MB", "text", True),
                _setting_item("PDF_OCR_ENABLED", "PDF OCR", "OCR extraction for scanned PDFs.", "Enabled" if settings.pdf_ocr_enabled else "Disabled", "toggle", True),
            ],
        },
    ]


def _setting_item(
    key: str,
    label: str,
    description: str,
    value: str | None,
    item_type: str,
    editable: bool,
    options: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "description": description,
        "value": value or "",
        "type": item_type,
        "editable": editable,
        "options": options,
    }


def _mask_secret(value: str | None) -> str:
    if not value:
        return "Not configured"
    stripped = value.strip()
    if len(stripped) <= 8:
        return "Configured"
    return f"{stripped[:3]}****{stripped[-4:]}"


def _looks_secret_key(key: str) -> bool:
    return any(part in key.lower() for part in ("key", "secret", "token", "password"))


def _pii_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        action = str(row.get("action") or "").lower()
        types = row.get("pii_types") or row.get("found_types") or []
        if not types and "pii" not in action:
            continue
        if isinstance(types, str):
            types = [types]
        if not types:
            types = ["PII"]
        for pii_type in types:
            records.append({
                "id": str(row.get("id") or f"pii-{len(records)}"),
                "dataType": str(pii_type),
                "source": str(row.get("source") or row.get("action") or "legal_query"),
                "detectedIn": str(row.get("session_id") or row.get("message_id") or "-"),
                "user": str(row.get("user_id") or "system"),
                "tenant": str(row.get("tenant_id") or "-"),
                "timestamp": _time_ago(row.get("created_at")),
                "createdAt": row.get("created_at"),
                "status": "masked",
                "riskLevel": "high" if str(pii_type).upper() in {"THAI_ID", "BANK_ACCOUNT", "CREDIT_CARD"} else "medium",
                "action": "Automatically redacted before LLM call",
            })
    return records[:200]


def _pii_rules(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patterns = [
        ("THAI_ID", r"\b\d{1}-\d{4}-\d{5}-\d{2}-\d{1}\b", "Auto-mask"),
        ("PHONE_TH", r"\b0[689]\d[-\s]?\d{3}[-\s]?\d{4}\b", "Auto-mask"),
        ("EMAIL", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "Auto-mask"),
        ("BANK_ACCOUNT", r"\b\d{3}-\d{1}-\d{5}-\d{1}\b", "Auto-mask + alert"),
        ("PASSPORT", r"\b[A-Z]{1,2}\d{6,9}\b", "Auto-mask"),
        ("CREDIT_CARD", r"\b(?:\d[ \-]?){13,16}\b", "Auto-mask + alert"),
    ]
    return [
        {
            "id": pii_type.lower(),
            "type": pii_type,
            "pattern": pattern,
            "action": action,
            "detected": len([row for row in records if row["dataType"] == pii_type]),
            "enabled": True,
        }
        for pii_type, pattern, action in patterns
    ]


def _pii_trend(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    days = _last_7_day_labels()
    buckets = {day: {"day": day, "detected": 0, "masked": 0} for day in days}
    for record in records:
        created = _parse_dt(record.get("createdAt"))
        day = _day_label(created) if created else days[-1]
        if day in buckets:
            buckets[day]["detected"] += 1
            if record.get("status") == "masked":
                buckets[day]["masked"] += 1
    return list(buckets.values())


def _pii_type_distribution(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    colors = ["hsl(var(--destructive))", "hsl(var(--amber-warning))", "hsl(var(--sky-info))", "hsl(var(--primary))", "hsl(var(--muted-foreground))"]
    counts: dict[str, int] = {}
    for record in records:
        counts[record["dataType"]] = counts.get(record["dataType"], 0) + 1
    return [
        {"name": name, "value": value, "color": colors[index % len(colors)]}
        for index, (name, value) in enumerate(counts.items())
    ]


def _auto_mask_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 100.0
    masked = len([record for record in records if record["status"] == "masked"])
    return round(masked / len(records) * 100, 1)


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _time_ago(value: Any) -> str:
    created = _parse_dt(value)
    if not created:
        return "-"
    delta = datetime.now(timezone.utc) - created
    if delta.total_seconds() < 60:
        return "just now"
    if delta.total_seconds() < 3600:
        return f"{int(delta.total_seconds() // 60)} min ago"
    if delta.days < 1:
        return f"{int(delta.total_seconds() // 3600)}h ago"
    return f"{delta.days}d ago"


def _last_7_day_labels() -> list[str]:
    today = datetime.now(timezone.utc).date()
    return [
        (today - timedelta(days=6 - index)).strftime("%a")
        for index in range(7)
    ]


def _day_label(value: datetime | None) -> str:
    return (value or datetime.now(timezone.utc)).strftime("%a")


def _average(values: list[float]) -> float:
    numbers = [value for value in values if value]
    return sum(numbers) / len(numbers) if numbers else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    numbers = sorted([value for value in values if value])
    if not numbers:
        return 0.0
    index = min(len(numbers) - 1, max(0, int(round((len(numbers) - 1) * percentile))))
    return numbers[index]


def _provider_name(model: str | None) -> str:
    name = str(model or "").lower()
    if name.startswith("claude"):
        return "Anthropic"
    if name.startswith(("gpt", "o1", "o3", "o4")):
        return "OpenAI"
    return "Provider"
