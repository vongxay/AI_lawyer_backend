"""
memory/case_memory.py
=====================
Case Memory System — 3-tier architecture per blueprint.

Tier 1: Redis (TTL 24h)  — active session hot cache
Tier 2: Supabase         — persistent case memory (case_memory table + RLS)
Tier 3: In-memory dict   — development fallback (no external deps)

Key operations:
    get(case_id)         — load memory context for IRAC agent
    update(...)          — append IRAC result, refresh facts summary
    summarize_facts(...)  — LLM-powered facts consolidation

RLS is enforced at DB level — tenant_id filter is enforced by Supabase policies.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from core.config import get_settings
from core.logging import get_logger

if TYPE_CHECKING:
    import redis.asyncio as aioredis
    from supabase import AsyncClient  # pragma: no cover

log = get_logger(__name__)

_CACHE_PREFIX = "case_memory:"
_CACHE_TTL = 86_400  # 24 hours


class CaseMemoryService:
    def __init__(
        self,
        supabase: "AsyncClient | None" = None,
        redis: "aioredis.Redis | None" = None,
    ) -> None:
        self._supabase = supabase
        self._redis = redis
        # In-process fallback for dev / testing
        self._local: dict[str, dict] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    async def get(self, case_id: str | None) -> dict[str, Any]:
        """Load full case memory context. Returns {"empty": True} if not found."""
        if not case_id:
            return {"empty": True}

        # Tier 1: Redis cache
        cached = await self._redis_get(case_id)
        if cached:
            log.debug("case_memory.cache_hit", case_id=case_id)
            return cached

        # Tier 2: Supabase
        if self._supabase:
            db_data = await self._supabase_get(case_id)
            if db_data:
                await self._redis_set(case_id, db_data)
                return db_data

        # Tier 3: Local fallback
        local = self._local.get(case_id)
        if local:
            return local

        return {"empty": True}

    async def update(
        self,
        *,
        case_id: str | None,
        question: str,
        irac: dict,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Append IRAC result to case history, refresh facts summary."""
        if not case_id:
            return

        existing = await self.get(case_id)
        if existing.get("empty"):
            existing = self._new_case_record(case_id, tenant_id, user_id)

        # Append IRAC to history (keep last 20 entries)
        history: list[dict] = existing.get("irac_history", [])
        history.append({
            "ts": int(time.time()),
            "question": question[:200],
            "irac_conclusion": irac.get("irac", {}).get("conclusion", {}),
            "confidence": irac.get("confidence", 0),
        })
        existing["irac_history"] = history[-20:]

        # Merge new citations into key_citations
        new_citations = irac.get("citations", [])
        existing_citations: list = existing.get("key_citations", [])
        existing["key_citations"] = self._merge_citations(existing_citations, new_citations)

        # Refresh updated_at
        existing["updated_at"] = int(time.time())
        existing.pop("empty", None)

        # Persist
        await self._persist(case_id, existing)
        await self._redis_set(case_id, existing)
        self._local[case_id] = existing

        log.info(
            "case_memory.updated",
            case_id=case_id,
            history_len=len(existing["irac_history"]),
            citations=len(existing["key_citations"]),
        )

    async def get_context_for_prompt(self, case_id: str | None, current_question: str) -> dict:
        """
        Returns a slimmed context dict suitable for injection into IRAC prompt.
        Avoids sending full history to LLM (token cost).
        """
        memory = await self.get(case_id)
        if memory.get("empty"):
            return {}

        # Select the 3 most relevant past IRAC entries (by recency for now)
        recent = (memory.get("irac_history") or [])[-3:]
        return {
            "facts_summary": memory.get("facts_summary"),
            "relevant_irac_history": recent,
            "key_citations": (memory.get("key_citations") or [])[:10],
            "strategies_tried": memory.get("strategies", []),
            "case_status": memory.get("status", "active"),
        }

    # ── Supabase persistence ──────────────────────────────────────────────────

    async def _supabase_get(self, case_id: str) -> dict | None:
        try:
            result = await self._supabase.table("case_memory") \
                .select("*") \
                .eq("case_id", case_id) \
                .single() \
                .execute()
            return result.data
        except Exception as exc:
            log.warning("case_memory.supabase_get.failed", case_id=case_id, error=str(exc))
            return None

    async def _persist(self, case_id: str, data: dict) -> None:
        if not self._supabase:
            return
        try:
            payload = {
                "case_id": case_id,
                "tenant_id": data.get("tenant_id"),
                "client_id": data.get("client_id"),
                "facts_summary": data.get("facts_summary"),
                "irac_history": data.get("irac_history", []),
                "key_citations": data.get("key_citations", []),
                "strategies": data.get("strategies", []),
                "status": data.get("status", "active"),
                "updated_at": data.get("updated_at"),
            }
            # Upsert — insert or update
            await self._supabase.table("case_memory").upsert(payload, on_conflict="case_id").execute()
        except Exception as exc:
            log.error("case_memory.supabase_persist.failed", case_id=case_id, error=str(exc))

    # ── Redis caching ─────────────────────────────────────────────────────────

    async def _redis_get(self, case_id: str) -> dict | None:
        if not self._redis:
            return None
        try:
            raw = await self._redis.get(f"{_CACHE_PREFIX}{case_id}")
            return json.loads(raw) if raw else None
        except Exception as exc:
            log.debug("case_memory.redis_get.failed", error=str(exc))
            return None

    async def _redis_set(self, case_id: str, data: dict) -> None:
        if not self._redis:
            return
        try:
            await self._redis.setex(
                f"{_CACHE_PREFIX}{case_id}",
                _CACHE_TTL,
                json.dumps(data, ensure_ascii=False, default=str),
            )
        except Exception as exc:
            log.debug("case_memory.redis_set.failed", error=str(exc))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _new_case_record(self, case_id: str, tenant_id: str | None, user_id: str | None) -> dict:
        return {
            "case_id": case_id,
            "tenant_id": tenant_id,
            "client_id": user_id,
            "facts_summary": None,
            "irac_history": [],
            "key_citations": [],
            "strategies": [],
            "status": "active",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        }

    def _merge_citations(self, existing: list, new_citations: list) -> list:
        """Add new citations not already tracked, keep most recent 20."""
        existing_refs = {c.get("ref") for c in existing}
        unique_new = [c for c in new_citations if c.get("ref") not in existing_refs]
        merged = existing + unique_new
        return merged[-20:]
