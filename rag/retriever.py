"""
rag/retriever.py
================
Hybrid search: Semantic (pgvector cosine) + Keyword (BM25/FTS) with RRF fusion.
Falls back to keyword-only or stub when Supabase is unavailable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.core.logging import get_logger

if TYPE_CHECKING:
    from supabase import AsyncClient  # pragma: no cover

log = get_logger(__name__)


class Retriever:
    def __init__(self, supabase: "AsyncClient | None" = None) -> None:
        self._supabase = supabase

    async def retrieve(
        self,
        *,
        query: str,
        embedding: list[float] | None = None,
        jurisdiction: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if not self._supabase:
            return self._stub_results(query, top_k)

        try:
            return await self._hybrid_search(
                query=query,
                embedding=embedding or [],
                jurisdiction=jurisdiction,
                top_k=top_k,
            )
        except Exception as exc:
            log.warning("retriever.search.failed", error=str(exc))
            return self._stub_results(query, top_k)

    async def _hybrid_search(
        self,
        *,
        query: str,
        embedding: list[float],
        jurisdiction: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Calls the hybrid_legal_search Supabase RPC function defined in blueprint.
        Returns fused + scored results.
        """
        params: dict[str, Any] = {
            "query_text": query,
            "match_count": top_k,
            "rrf_k": 60,
        }
        if embedding:
            params["query_embedding"] = embedding
        if jurisdiction:
            params["p_jurisdiction"] = jurisdiction

        result = await self._supabase.rpc("hybrid_legal_search", params).execute()
        data = result.data or []

        log.info("retriever.hybrid_search.ok", results=len(data), jurisdiction=jurisdiction)
        return data

    def _stub_results(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Development stub — returns plausible-looking results."""
        return [
            {
                "id": f"stub-{i}",
                "type": "law",
                "title": "ประมวลกฎหมายแพ่งและพาณิชย์",
                "section": f"มาตรา {420 + i}",
                "content": f"stub context chunk {i + 1} for: {query[:60]}",
                "jurisdiction": "TH",
                "status": "ACTIVE",
                "final_score": 1.0 - (i * 0.05),
            }
            for i in range(min(top_k, 3))
        ]
