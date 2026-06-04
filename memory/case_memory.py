"""
memory/case_memory.py
=====================
Persistent case memory with Redis, Supabase, and in-process fallbacks.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from core.logging import get_logger

if TYPE_CHECKING:
    import redis.asyncio as aioredis
    from supabase import AsyncClient  # pragma: no cover

log = get_logger(__name__)

_CACHE_PREFIX = "case_memory:"
_CACHE_TTL = 86_400


class CaseMemoryService:
    def __init__(
        self,
        supabase: "AsyncClient | None" = None,
        redis: "aioredis.Redis | None" = None,
    ) -> None:
        self._supabase = supabase
        self._redis = redis
        self._local: dict[str, dict[str, Any]] = {}

    async def get(
        self,
        case_id: str | None,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Load full case memory context. Returns {"empty": True} if not found."""
        if not case_id:
            return {"empty": True}

        cache_key = self._cache_key(case_id, tenant_id)
        cached = await self._redis_get(cache_key)
        if cached and self._record_is_allowed(cached, tenant_id=tenant_id, user_id=user_id):
            log.debug("case_memory.cache_hit", case_id=case_id, tenant_id=tenant_id)
            return cached

        if self._supabase and tenant_id:
            db_data = await self._supabase_get(case_id, tenant_id=tenant_id)
            if db_data:
                await self._redis_set(cache_key, db_data)
                return db_data

        local = self._local.get(cache_key) or self._local.get(case_id)
        if local and self._record_is_allowed(local, tenant_id=tenant_id, user_id=user_id):
            return local

        return {"empty": True}

    async def update(
        self,
        *,
        case_id: str | None,
        question: str,
        irac: dict[str, Any],
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Append IRAC result to case history, refresh citations, and persist."""
        if not case_id:
            return

        existing = await self.get(case_id, tenant_id=tenant_id)
        if existing.get("empty"):
            existing = self._new_case_record(case_id, tenant_id, user_id)

        history: list[dict[str, Any]] = list(existing.get("irac_history") or [])
        history.append({
            "ts": int(time.time()),
            "question": question[:200],
            "irac_conclusion": irac.get("irac", {}).get("conclusion", {}),
            "confidence": irac.get("confidence", 0),
        })
        existing["irac_history"] = history[-20:]

        existing["key_citations"] = self._merge_citations(
            list(existing.get("key_citations") or []),
            list(irac.get("citations") or []),
        )
        existing["tenant_id"] = tenant_id or existing.get("tenant_id")
        existing["client_id"] = user_id or existing.get("client_id")
        existing["case_id"] = case_id
        existing["legal_case_id"] = case_id
        existing["updated_at"] = int(time.time())
        existing.pop("empty", None)

        await self._persist(case_id, existing)
        cache_key = self._cache_key(case_id, existing.get("tenant_id"))
        await self._redis_set(cache_key, existing)
        self._local[cache_key] = existing

        log.info(
            "case_memory.updated",
            case_id=case_id,
            tenant_id=existing.get("tenant_id"),
            history_len=len(existing["irac_history"]),
            citations=len(existing["key_citations"]),
        )

    async def get_context_for_prompt(
        self,
        case_id: str | None,
        current_question: str,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        _ = current_question
        memory = await self.get(case_id, tenant_id=tenant_id, user_id=user_id)
        if memory.get("empty"):
            return {}

        recent = (memory.get("irac_history") or [])[-3:]
        return {
            "facts_summary": memory.get("facts_summary"),
            "relevant_irac_history": recent,
            "key_citations": (memory.get("key_citations") or [])[:10],
            "strategies_tried": memory.get("strategies", []),
            "case_status": memory.get("status", "active"),
        }

    async def _supabase_get(self, case_id: str, *, tenant_id: str) -> dict[str, Any] | None:
        for case_column in ("legal_case_id", "case_id"):
            try:
                result = await (
                    self._supabase.table("case_memory")
                    .select("*")
                    .eq(case_column, case_id)
                    .eq("tenant_id", tenant_id)
                    .limit(1)
                    .execute()
                )
                data = result.data or []
                if isinstance(data, list):
                    return data[0] if data else None
                return data
            except Exception as exc:
                log.warning(
                    "case_memory.supabase_get_variant.failed",
                    case_id=case_id,
                    tenant_id=tenant_id,
                    case_column=case_column,
                    error=str(exc),
                )
        return None

    async def _persist(self, case_id: str, data: dict[str, Any]) -> None:
        if not self._supabase:
            return

        modern_payload = {
            "legal_case_id": case_id,
            "tenant_id": data.get("tenant_id"),
            "facts_summary": data.get("facts_summary"),
            "irac_history": data.get("irac_history", []),
            "key_citations": data.get("key_citations", []),
            "strategies": data.get("strategies", []),
            "status": data.get("status", "active"),
            "updated_at": data.get("updated_at"),
        }
        legacy_payload = {
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

        last_error: Exception | None = None
        for payload, conflict_column in (
            (modern_payload, "legal_case_id"),
            (legacy_payload, "case_id"),
        ):
            try:
                await self._supabase.table("case_memory").upsert(
                    payload,
                    on_conflict=conflict_column,
                ).execute()
                return
            except Exception as exc:
                last_error = exc
                log.warning(
                    "case_memory.supabase_persist_variant.failed",
                    case_id=case_id,
                    conflict_column=conflict_column,
                    error=str(exc),
                )

        if last_error:
            log.error("case_memory.supabase_persist.failed", case_id=case_id, error=str(last_error))

    async def _redis_get(self, cache_key: str) -> dict[str, Any] | None:
        if not self._redis:
            return None
        try:
            raw = await self._redis.get(f"{_CACHE_PREFIX}{cache_key}")
            return json.loads(raw) if raw else None
        except Exception as exc:
            log.debug("case_memory.redis_get.failed", error=str(exc))
            return None

    async def _redis_set(self, cache_key: str, data: dict[str, Any]) -> None:
        if not self._redis:
            return
        try:
            await self._redis.setex(
                f"{_CACHE_PREFIX}{cache_key}",
                _CACHE_TTL,
                json.dumps(data, ensure_ascii=False, default=str),
            )
        except Exception as exc:
            log.debug("case_memory.redis_set.failed", error=str(exc))

    def _new_case_record(self, case_id: str, tenant_id: str | None, user_id: str | None) -> dict[str, Any]:
        now = int(time.time())
        return {
            "case_id": case_id,
            "legal_case_id": case_id,
            "tenant_id": tenant_id,
            "client_id": user_id,
            "facts_summary": None,
            "irac_history": [],
            "key_citations": [],
            "strategies": [],
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }

    def _merge_citations(self, existing: list[dict[str, Any]], new_citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        existing_refs = {citation.get("ref") for citation in existing}
        unique_new = [citation for citation in new_citations if citation.get("ref") not in existing_refs]
        return (existing + unique_new)[-20:]

    def _record_is_allowed(
        self,
        record: dict[str, Any],
        *,
        tenant_id: str | None,
        user_id: str | None,
    ) -> bool:
        if tenant_id and record.get("tenant_id") not in {tenant_id, None}:
            return False
        if user_id and record.get("client_id") not in {user_id, None}:
            return False
        return True

    def _cache_key(self, case_id: str, tenant_id: str | None) -> str:
        return f"{tenant_id or 'no-tenant'}:{case_id}"
