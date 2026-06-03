"""
rag/reranker.py
===============
Cross-encoder reranker for final relevance scoring.

Current implementation: deterministic keyword and authority scoring.
Production upgrade path: cross-encoder model or managed rerank API.
"""
from __future__ import annotations

from typing import Any

from core.logging import get_logger

log = get_logger(__name__)


class Reranker:
    """
    Deterministic reranker using keyword overlap plus source authority.
    Replace `_score` with a real cross-encoder when that provider is configured.

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
        authority_boost = self._authority_boost(chunk)
        return base_score + keyword_boost + authority_boost

    def _authority_boost(self, chunk: dict) -> float:
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        source_url = str(chunk.get("source_url") or metadata.get("source_url") or "").lower()
        source_table = str(chunk.get("source_table") or "").lower()
        doc_type = str(chunk.get("doc_type") or chunk.get("type") or "").lower()
        jurisdiction = str(chunk.get("jurisdiction") or metadata.get("jurisdiction") or "").lower()

        boost = 0.0
        if "laoofficialgazette.gov.la" in source_url:
            boost += 0.25
        if jurisdiction in {"laos", "la", "lao pdr"} and doc_type in {"law", "statute", "regulation", "decree"}:
            boost += 0.12
        if source_table == "laws":
            boost += 0.08
        return min(boost, 0.35)
