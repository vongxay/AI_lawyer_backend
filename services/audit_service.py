from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class AuditEvent:
    user_id: str | None
    agent: str
    query_hash: str
    confidence: float
    agents_used: list[str]
    ts: int


class AuditService:
    def hash_query(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def log(self, *, user_id: str | None, agent: str, query: str, confidence: float, agents_used: list[str]) -> AuditEvent:
        return AuditEvent(
            user_id=user_id,
            agent=agent,
            query_hash=self.hash_query(query),
            confidence=confidence,
            agents_used=agents_used,
            ts=int(time.time()),
        )

