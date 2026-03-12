from __future__ import annotations


class Retriever:
    async def retrieve(self, *, query: str, jurisdiction: str | None = None, top_k: int = 10) -> list[dict]:
        _ = (jurisdiction, top_k)
        return [{"type": "law", "title": "stub", "content": f"stub context for: {query[:80]}"}]

