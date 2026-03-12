from __future__ import annotations


class Reranker:
    async def rerank(self, *, query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
        _ = (query, top_k)
        return chunks[:top_k]

