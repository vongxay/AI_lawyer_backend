"""
services/audit_service.py
==========================
Append-only audit trail for every agent interaction.

Every query/response cycle writes one AuditEvent.
In production this persists to Supabase `audit_log` table.
In development it logs to stdout.

Schema matches blueprint:
    audit_log(id, user_id, agent, query_hash, confidence, agents_used, ts)
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.logging import get_logger

if TYPE_CHECKING:
    from supabase import AsyncClient  # pragma: no cover

log = get_logger(__name__)

_DB_AGENT_NAMES = {
    "research": "legal_research",
    "reasoning": "irac_reasoning",
    "verification": "citation_verification",
    "document": "document_analysis",
    "evidence": "evidence_analyzer",
    "risk": "risk_strategy",
    "classifier": "query_classifier",
}


def _db_agent_name(agent: str) -> str:
    return _DB_AGENT_NAMES.get(agent, agent)


@dataclass(frozen=True)
class AuditEvent:
    user_id: str | None
    tenant_id: str | None
    agent: str
    query_hash: str
    confidence: float
    agents_used: list[str]
    processing_time_ms: int
    escalated: bool
    ts: int = field(default_factory=lambda: int(time.time()))


class AuditService:
    def __init__(self, supabase: "AsyncClient | None" = None) -> None:
        self._supabase = supabase

    @staticmethod
    def hash_query(text: str) -> str:
        """One-way hash — query content never stored in audit log."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def log_event(
        self,
        *,
        user_id: str | None,
        tenant_id: str | None,
        agent: str,
        query: str,
        confidence: float,
        agents_used: list[str],
        processing_time_ms: int = 0,
        escalated: bool = False,
    ) -> AuditEvent:
        event = AuditEvent(
            user_id=user_id,
            tenant_id=tenant_id,
            agent=agent,
            query_hash=self.hash_query(query),
            confidence=confidence,
            agents_used=agents_used,
            processing_time_ms=processing_time_ms,
            escalated=escalated,
        )

        log.info(
            "audit.event",
            user_id=user_id,
            tenant_id=tenant_id,
            agent=agent,
            query_hash=event.query_hash,
            confidence=round(confidence, 3),
            agents_used=agents_used,
            processing_time_ms=processing_time_ms,
            escalated=escalated,
        )

        if self._supabase:
            db_agents_used = [_db_agent_name(name) for name in agents_used]
            modern_payload = {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "action": "legal_query",
                "agents_used": db_agents_used,
                "model_used": agent,
                "query_hash": event.query_hash,
                "confidence": confidence,
                "latency_ms": processing_time_ms,
                "success": True,
                "escalated": escalated,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            legacy_payload = {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "agent": _db_agent_name(agent),
                "query_hash": event.query_hash,
                "confidence": confidence,
                "agents_used": db_agents_used,
                "processing_time_ms": processing_time_ms,
                "escalated_to_expert": escalated,
                "ts": event.ts,
            }
            try:
                await self._supabase.table("audit_log").insert(modern_payload).execute()
            except Exception as exc:
                try:
                    await self._supabase.table("audit_log").insert(legacy_payload).execute()
                except Exception as legacy_exc:
                    # Audit failures must never crash the main flow
                    log.error("audit.persist.failed", error=str(legacy_exc), first_error=str(exc))

        return event


class ExpertQueueService:
    """Manages the human expert review queue."""

    def __init__(self, supabase: "AsyncClient | None" = None) -> None:
        self._supabase = supabase
        self._in_memory: list[dict] = []   # fallback when Supabase unavailable

    async def enqueue(
        self,
        *,
        session_id: str,
        user_id: str | None,
        reason: str,
        confidence: float,
        query_preview: str,
        tenant_id: str | None = None,
    ) -> None:
        entry = {
            "tenant_id": tenant_id,
            "session_id": session_id,
            "user_id": user_id,
            "reason": reason,
            "confidence": confidence,
            "query_preview": query_preview[:200],
            "status": "pending",
            "ts": int(time.time()),
        }
        log.warning("expert_queue.added", reason=reason, confidence=confidence)

        if self._supabase:
            modern_entry = {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "flagged_reason": reason,
                "confidence_at_flag": confidence,
                "status": "pending",
                "priority": 5,
            }
            try:
                await self._supabase.table("expert_reviews").insert(modern_entry).execute()
            except Exception as exc:
                if "message_id" in str(exc):
                    log.warning(
                        "expert_queue.persist.skipped",
                        reason="database_requires_message_id",
                        error=str(exc),
                    )
                    self._in_memory.append(entry)
                    return
                try:
                    await self._supabase.table("expert_reviews").insert(entry).execute()
                except Exception as legacy_exc:
                    log.error("expert_queue.persist.failed", error=str(legacy_exc), first_error=str(exc))
                    self._in_memory.append(entry)
        else:
            self._in_memory.append(entry)

    async def list_pending(self) -> list[dict]:
        if self._supabase:
            for order_column in ("created_at", "ts"):
                try:
                    result = await self._supabase.table("expert_reviews") \
                        .select("*") \
                        .eq("status", "pending") \
                        .order(order_column, desc=True) \
                        .limit(100) \
                        .execute()
                    return result.data or []
                except Exception as exc:
                    log.error("expert_queue.list.failed", order_column=order_column, error=str(exc))
        return self._in_memory
