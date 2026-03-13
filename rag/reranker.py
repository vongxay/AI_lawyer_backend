"""
rag/reranker.py
===============
Cross-encoder reranker for final relevance scoring.

Production: uses a cross-encoder model (via sentence-transformers or Cohere rerank API).
Stub: simple keyword overlap scoring — adequate for development.
"""
from __future__ import annotations

from typing import Any

from backend.core.logging import get_logger

log = get_logger(__name__)


class Reranker:
    """
    Stub reranker using keyword overlap score.
    Replace `_score` with a real cross-encoder for production.

    Production options:
    - Cohere Rerank API (recommended for Thai)
    - sentence-transformers cross-encoder/ms-marco-MiniLM-L-6-v2
    - BGE-reranker-large (multilingual)
    """

    async def rerank(
        self,
        *,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []

        scored = [(chunk, self._score(query, chunk)) for chunk in chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        result = [chunk for chunk, _ in scored[:top_k]]

        log.debug("reranker.done", input=len(chunks), output=len(result))
        return result

    def _score(self, query: str, chunk: dict) -> float:
        """Keyword overlap — replace with cross-encoder in production."""
        content = (chunk.get("content") or chunk.get("title") or "").lower()
        query_terms = set(query.lower().split())
        content_terms = set(content.split())
        overlap = len(query_terms & content_terms)
        base_score = float(chunk.get("final_score", 0.5))
        keyword_boost = min(0.3, overlap * 0.05)
        return base_score + keyword_boost
