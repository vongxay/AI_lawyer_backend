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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.core.logging import get_logger

if TYPE_CHECKING:
    from supabase import AsyncClient  # pragma: no cover

log = get_logger(__name__)


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
            try:
                await self._supabase.table("audit_log").insert({
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "agent": agent,
                    "query_hash": event.query_hash,
                    "confidence": confidence,
                    "agents_used": agents_used,
                    "processing_time_ms": processing_time_ms,
                    "escalated_to_expert": escalated,
                    "ts": event.ts,
                }).execute()
            except Exception as exc:
                # Audit failures must never crash the main flow
                log.error("audit.persist.failed", error=str(exc))

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
    ) -> None:
        entry = {
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
            try:
                await self._supabase.table("expert_reviews").insert(entry).execute()
            except Exception as exc:
                log.error("expert_queue.persist.failed", error=str(exc))
                self._in_memory.append(entry)
        else:
            self._in_memory.append(entry)

    async def list_pending(self) -> list[dict]:
        if self._supabase:
            try:
                result = await self._supabase.table("expert_reviews") \
                    .select("*") \
                    .eq("status", "pending") \
                    .order("ts", desc=True) \
                    .limit(100) \
                    .execute()
                return result.data or []
            except Exception as exc:
                log.error("expert_queue.list.failed", error=str(exc))
        return self._in_memory
