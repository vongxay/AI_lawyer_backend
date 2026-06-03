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
                embedding=embedding,
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
        embedding: list[float] | None,
        jurisdiction: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Prefer chunk-level Agentic RAG search. Fall back to the legacy document
        RPC while deployments are migrating.
        """
        chunk_rows = await self._chunk_search(
            query=query,
            embedding=embedding,
            jurisdiction=jurisdiction,
            top_k=top_k,
        )
        if chunk_rows:
            log.info("retriever.chunk_search.ok", results=len(chunk_rows), jurisdiction=jurisdiction)
            return chunk_rows

        if not embedding:
            log.warning("retriever.no_embedding_for_legacy_search", jurisdiction=jurisdiction)
            return []

        return await self._legacy_hybrid_search(
            query=query,
            embedding=embedding,
            jurisdiction=jurisdiction,
            top_k=top_k,
        )

    async def _chunk_search(
        self,
        *,
        query: str,
        embedding: list[float] | None,
        jurisdiction: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "query_text": query,
            "match_count": top_k,
            "rrf_k": 60,
            "p_status": "active",
            "p_review_status": "approved",
        }
        if embedding:
            params["query_embedding"] = embedding
        if jurisdiction:
            params["p_jurisdiction"] = jurisdiction

        try:
            result = await self._supabase.rpc("hybrid_document_chunk_search", params).execute()
            return [self._normalise_row(row) for row in (result.data or [])]
        except Exception as exc:
            log.warning("retriever.chunk_search.failed", error=str(exc))
            return []

    async def _legacy_hybrid_search(
        self,
        *,
        query: str,
        embedding: list[float],
        jurisdiction: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "query_text": query,
            "match_count": top_k,
            "rrf_k": 60,
        }
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
            "chunk_id": row.get("chunk_id") or metadata.get("chunk_id"),
            "source_id": row.get("id") or metadata.get("source_id"),
            "source_url": row.get("source_url") or metadata.get("source_url"),
        }
