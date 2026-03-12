from __future__ import annotations

import time


class CaseMemoryService:
    """
    Stub in-memory implementation.
    Replace with Supabase-backed version matching blueprint (case_memory table + RLS).
    """

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    async def get(self, case_id: str | None) -> dict:
        if not case_id:
            return {"empty": True}
        return self._store.get(case_id, {"empty": True})

    async def update(self, *, case_id: str | None, question: str, irac: dict) -> None:
        if not case_id:
            return
        existing = self._store.get(case_id, {})
        history = existing.get("irac_history", [])
        history.append({"ts": int(time.time()), "question": question, "irac": irac.get("irac")})
        self._store[case_id] = {
            **existing,
            "facts_summary": existing.get("facts_summary", "stub facts summary"),
            "irac_history": history[-20:],
            "updated_at": int(time.time()),
        }

