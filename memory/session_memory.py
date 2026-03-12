from __future__ import annotations


class SessionMemoryService:
    """
    Placeholder for Redis TTL session memory (24h) per blueprint.
    """

    async def get(self, session_id: str) -> dict:
        return {"session_id": session_id, "messages": []}

    async def put(self, session_id: str, data: dict) -> None:
        _ = (session_id, data)

