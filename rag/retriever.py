"""
rag/retriever.py
================
Hybrid search: Semantic (pgvector cosine) + Keyword (BM25/FTS) with RRF fusion.
Returns no results when Supabase/search is unavailable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.jurisdiction import canonical_jurisdiction
from core.logging import get_logger

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
            log.warning("retriever.no_database", mode="empty")
            return []

        try:
            return await self._hybrid_search(
                query=query,
                embedding=embedding or [],
                jurisdiction=canonical_jurisdiction(jurisdiction),
                top_k=top_k,
            )
        except Exception as exc:
            log.warning("retriever.search.failed", error=str(exc))
            return []

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
        data = [self._normalise_row(row) for row in (result.data or [])]

        log.info("retriever.hybrid_search.ok", results=len(data), jurisdiction=jurisdiction)
        return data

    def _normalise_row(self, row: dict[str, Any]) -> dict[str, Any]:
        source_table = row.get("source_table")
        doc_type = row.get("doc_type")
        normalised_type = row.get("type") or doc_type or source_table or "doc"
        if source_table == "cases":
            normalised_type = "case"
        elif source_table == "laws":
            normalised_type = "law"
        elif source_table == "legal_forms":
            normalised_type = "form"

        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        return {
            **row,
            "type": normalised_type,
            "section": row.get("section") or row.get("section_number") or metadata.get("section"),
            "source_url": row.get("source_url") or metadata.get("source_url"),
        }
