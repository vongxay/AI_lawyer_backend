"""
api/dependencies.py
====================
FastAPI dependency injection container.

Single source of truth for all shared services.
Callers use `Depends(get_workflow_manager)` etc.

WorkflowManager is created once per app lifecycle (not per request)
since it holds agent instances + connection pools.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from core.database import get_redis, get_supabase
from orchestrator.workflow_manager import WorkflowManager
from services.audit_service import AuditService, ExpertQueueService

# ── Singletons (app-level, not request-level) ─────────────────────────────────

_workflow_manager: WorkflowManager | None = None


async def get_workflow_manager() -> WorkflowManager:
    global _workflow_manager
    if _workflow_manager is None:
        supabase = await get_supabase()
        redis = await get_redis()
        _workflow_manager = WorkflowManager(supabase=supabase, redis=redis)
    return _workflow_manager


async def get_audit_service() -> AuditService:
    supabase = await get_supabase()
    return AuditService(supabase=supabase)


async def get_expert_queue() -> ExpertQueueService:
    supabase = await get_supabase()
    return ExpertQueueService(supabase=supabase)


# ── Type aliases for route signatures ─────────────────────────────────────────

WorkflowDep = Annotated[WorkflowManager, Depends(get_workflow_manager)]
AuditDep = Annotated[AuditService, Depends(get_audit_service)]
ExpertQueueDep = Annotated[ExpertQueueService, Depends(get_expert_queue)]
