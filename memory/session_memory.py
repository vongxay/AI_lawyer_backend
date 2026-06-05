from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from core.logging import get_logger
from services.pii_service import PiiService

if TYPE_CHECKING:
    import redis.asyncio as aioredis
    from supabase import AsyncClient  # pragma: no cover

log = get_logger(__name__)

_CACHE_PREFIX = "session_memory:"
_CACHE_TTL = 900
_DEFAULT_MAX_MESSAGES = 10
_MAX_MESSAGE_CHARS = 700


class SessionMemoryService:
    """
    Short-term conversation memory for a single chat session.

    This is not legal authority. It exists to resolve follow-up questions,
    pronouns, user goals, and facts already supplied in the same chat.
    """

    def __init__(
        self,
        supabase: "AsyncClient | None" = None,
        redis: "aioredis.Redis | None" = None,
    ) -> None:
        self._supabase = supabase
        self._redis = redis
        self._pii = PiiService()
        self._local: dict[str, dict[str, Any]] = {}

    async def get(
        self,
        session_id: str | None,
        *,
        tenant_id: str | None = None,
        user_id: str | None = None,
        max_messages: int = _DEFAULT_MAX_MESSAGES,
    ) -> dict[str, Any]:
        if not session_id:
            return self._empty(None)

        cache_key = self._cache_key(session_id, tenant_id, user_id)
        cached = await self._redis_get(cache_key)
        if cached:
            return cached

        if self._supabase and tenant_id:
            if not await self._session_is_allowed(session_id, tenant_id=tenant_id, user_id=user_id):
                log.warning("session_memory.access_denied", session_id=session_id, tenant_id=tenant_id)
                return self._empty(session_id)

            messages = await self._load_supabase_messages(
                session_id,
                tenant_id=tenant_id,
                max_messages=max_messages,
            )
            memory = self._build_memory(session_id, messages)
            await self._redis_set(cache_key, memory)
            self._local[cache_key] = memory
            return memory

        local = self._local.get(cache_key)
        return local or self._empty(session_id)

    async def put(self, session_id: str, data: dict[str, Any]) -> None:
        cache_key = self._cache_key(session_id, data.get("tenant_id"), data.get("user_id"))
        self._local[cache_key] = data
        await self._redis_set(cache_key, data)

    async def _session_is_allowed(
        self,
        session_id: str,
        *,
        tenant_id: str,
        user_id: str | None,
    ) -> bool:
        if not self._supabase:
            return False

        try:
            query = (
                self._supabase.table("case_sessions")
                .select("id")
                .eq("id", session_id)
                .eq("tenant_id", tenant_id)
                .limit(1)
            )
            if user_id:
                query = query.eq("user_id", user_id)
            result = await query.execute()
            return bool(result.data)
        except Exception as exc:
            log.warning("session_memory.session_lookup.failed", session_id=session_id, error=str(exc))
            return False

    async def _load_supabase_messages(
        self,
        session_id: str,
        *,
        tenant_id: str,
        max_messages: int,
    ) -> list[dict[str, Any]]:
        try:
            result = await (
                self._supabase.table("messages")
                .select("role, content, irac_output, confidence, escalated, escalation_reason, created_at")
                .eq("session_id", session_id)
                .eq("tenant_id", tenant_id)
                .order("created_at", desc=True)
                .limit(max(1, max_messages * 2))
                .execute()
            )
        except Exception as exc:
            log.warning("session_memory.messages_load.failed", session_id=session_id, error=str(exc))
            return []

        rows = list(reversed(result.data or []))
        messages: list[dict[str, Any]] = []
        for row in rows:
            if self._is_internal_message(row):
                continue
            role = "assistant" if row.get("role") in {"assistant", "ai"} else "user" if row.get("role") == "user" else "system"
            content = self._clean_content(row)
            if not content:
                continue
            messages.append({
                "role": role,
                "content": self._pii.redact_text(content[:_MAX_MESSAGE_CHARS]),
                "created_at": row.get("created_at"),
            })
        return messages[-max_messages:]

    def _build_memory(self, session_id: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if not messages:
            return self._empty(session_id)

        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        assistant_messages = [m["content"] for m in messages if m.get("role") == "assistant"]
        summary_lines = [
            f"{self._role_label(m.get('role'))}: {m.get('content')}"
            for m in messages
        ]
        summary = "\n".join(summary_lines)
        digest = hashlib.sha256(summary.encode("utf-8")).hexdigest()

        return {
            "empty": False,
            "session_id": session_id,
            "messages": messages,
            "conversation_summary": summary,
            "current_user_state": user_messages[-1] if user_messages else "",
            "last_assistant_answer": assistant_messages[-1] if assistant_messages else "",
            "message_count": len(messages),
            "cache_key": digest,
        }

    def _clean_content(self, row: dict[str, Any]) -> str:
        content = str(row.get("content") or "").strip()
        if self._looks_like_structured_payload(content):
            irac = row.get("irac_output") if isinstance(row.get("irac_output"), dict) else {}
            content = self._content_from_irac(irac) or self._strip_json_blocks(content)
        return content.strip()

    def _content_from_irac(self, irac: dict[str, Any]) -> str:
        application = irac.get("application") if isinstance(irac.get("application"), dict) else {}
        conclusion = irac.get("conclusion") if isinstance(irac.get("conclusion"), dict) else {}
        parts = [
            str(conclusion.get("recommendation") or "").strip(),
            str(application.get("analysis") or "").strip(),
        ]
        return "\n\n".join(part for part in parts if part)

    def _is_internal_message(self, row: dict[str, Any]) -> bool:
        content = str(row.get("content") or "")
        return (
            row.get("role") == "assistant"
            and bool(row.get("escalation_reason"))
            and content.startswith("Escalated for expert review:")
        )

    def _looks_like_structured_payload(self, content: str) -> bool:
        clean = content.strip()
        return clean.startswith("{") or "```json" in clean.lower() or '"irac"' in clean

    def _strip_json_blocks(self, content: str) -> str:
        import re
        return re.sub(r"```(?:json)?[\s\S]*?```", "", content, flags=re.IGNORECASE).strip()

    def _role_label(self, role: Any) -> str:
        if role == "user":
            return "User"
        if role == "assistant":
            return "Assistant"
        return "System"

    async def _redis_get(self, cache_key: str) -> dict[str, Any] | None:
        if not self._redis:
            return None
        try:
            raw = await self._redis.get(f"{_CACHE_PREFIX}{cache_key}")
            return json.loads(raw) if raw else None
        except Exception as exc:
            log.debug("session_memory.redis_get.failed", error=str(exc))
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
            log.debug("session_memory.redis_set.failed", error=str(exc))

    def _cache_key(self, session_id: str, tenant_id: str | None, user_id: str | None) -> str:
        return f"{tenant_id or 'no-tenant'}:{user_id or 'no-user'}:{session_id}"

    def _empty(self, session_id: str | None) -> dict[str, Any]:
        return {
            "empty": True,
            "session_id": session_id,
            "messages": [],
            "conversation_summary": "",
            "current_user_state": "",
            "last_assistant_answer": "",
            "message_count": 0,
            "cache_key": "empty",
        }

