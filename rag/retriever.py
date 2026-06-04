"""
rag/retriever.py
================
Hybrid search: Semantic (pgvector cosine) + Keyword (BM25/FTS) with RRF fusion.
Returns no results when Supabase/search is unavailable.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from core.jurisdiction import canonical_jurisdiction
from core.logging import get_logger

if TYPE_CHECKING:
    from supabase import AsyncClient  # pragma: no cover

log = get_logger(__name__)


class Retriever:
    def __init__(self, supabase: "AsyncClient | None" = None) -> None:
        self._supabase = supabase
        self._chunk_search_supports_tenant_param: bool | None = False

    async def retrieve(
        self,
        *,
        query: str,
        embedding: list[float] | None = None,
        jurisdiction: str | None = None,
        tenant_id: str | None = None,
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
                tenant_id=tenant_id,
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
        tenant_id: str | None,
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
            tenant_id=tenant_id,
            top_k=top_k,
        )
        if chunk_rows:
            log.info("retriever.chunk_search.ok", results=len(chunk_rows), jurisdiction=jurisdiction)
            return chunk_rows

        keyword_rows = await self._direct_keyword_search(
            query=query,
            jurisdiction=jurisdiction,
            top_k=top_k,
        )
        if keyword_rows:
            log.info("retriever.direct_keyword.ok", results=len(keyword_rows), jurisdiction=jurisdiction)
            return keyword_rows

        if not embedding:
            log.info("retriever.keyword_only_no_results", jurisdiction=jurisdiction)
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
        tenant_id: str | None,
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
        if tenant_id and self._chunk_search_supports_tenant_param is not False:
            params["p_tenant_id"] = tenant_id

        try:
            result = await self._supabase.rpc("hybrid_document_chunk_search", params).execute()
            if "p_tenant_id" in params:
                self._chunk_search_supports_tenant_param = True
            return [self._normalise_row(row) for row in (result.data or [])]
        except Exception as exc:
            if tenant_id and "p_tenant_id" in str(exc):
                self._chunk_search_supports_tenant_param = False
                legacy_params = {key: value for key, value in params.items() if key != "p_tenant_id"}
                try:
                    result = await self._supabase.rpc("hybrid_document_chunk_search", legacy_params).execute()
                    log.info(
                        "retriever.chunk_search.legacy_signature",
                        reason="p_tenant_id_not_available_in_database_function",
                    )
                    return [self._normalise_row(row) for row in (result.data or [])]
                except Exception as legacy_exc:
                    log.warning("retriever.chunk_search.legacy_failed", error=str(legacy_exc))
                    return []

            log.warning("retriever.chunk_search.failed", error=str(exc))
            return []

    async def _direct_keyword_search(
        self,
        *,
        query: str,
        jurisdiction: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        terms = self._keyword_terms(query)
        if not terms:
            return []

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for term in terms[:8]:
            try:
                request = (
                    self._supabase.table("document_chunks")
                    .select(
                        "source_id, id, source_table, title, content, document_type, "
                        "jurisdiction, status, metadata, section_ref"
                    )
                    .eq("status", "active")
                    .eq("review_status", "approved")
                    .or_(f"title.ilike.%{term}%,content.ilike.%{term}%,section_ref.ilike.%{term}%")
                    .limit(top_k)
                )
                if jurisdiction:
                    request = request.eq("jurisdiction", jurisdiction)
                result = await request.execute()
                for row in result.data or []:
                    score = self._keyword_relevance_score(row, terms)
                    if score <= 0:
                        continue
                    key = str(row.get("id") or row.get("source_id"))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(self._normalise_row({**row, "final_score": score}))
                    if len(rows) >= top_k:
                        return rows
            except Exception as exc:
                log.debug("retriever.direct_keyword.term_failed", term=term, error=str(exc))
        return rows

    def _keyword_terms(self, query: str) -> list[str]:
        lowered = query.casefold()
        terms: list[str] = []

        for token in lowered.replace("\n", " ").split():
            cleaned = token.strip(".,;:()[]{}\"'!?")
            if len(cleaned) >= 2:
                terms.append(cleaned)

        land_markers = (
            "ທີ່ດິນ",
            "ກຳມະສິດ",
            "ສິດນຳໃຊ້",
            "ອະສັງຫາ",
            "ที่ดิน",
            "กรรมสิทธิ์",
            "land",
            "property",
            "ownership",
            "usufruct",
        )
        if any(marker in lowered for marker in land_markers):
            terms.extend([
                "ທີ່ດິນ",
                "ກຳມະສິດ",
                "ສິດນຳໃຊ້",
                "ກົດໝາຍທີ່ດິນ",
                "land",
                "property",
                "ownership",
                "usufruct",
            ])

        synonym_checks = [
            ("ເຊົ່າ", ["ເຊົ່າ", "ຄ່າເຊົ່າ", "ຜູ້ເຊົ່າ", "lease", "rent", "tenant", "เช่า", "ค่าเช่า"]),
            ("เช่า", ["เช่า", "ค่าเช่า", "ผู้เช่า", "ผู้ให้เช่า", "lease", "rent", "tenant"]),
            ("rent", ["rent", "lease", "tenant", "ค่าเช่า", "เช่า", "ເຊົ່າ"]),
            ("lease", ["lease", "rent", "tenant", "ค่าเช่า", "เช่า", "ເຊົ່າ"]),
        ]
        for marker, synonyms in synonym_checks:
            if marker in lowered:
                terms.extend(synonyms)

        unique: list[str] = []
        seen: set[str] = set()
        for term in terms:
            value = term.strip()
            if value and value not in seen:
                seen.add(value)
                unique.append(value)
        return unique

    def _keyword_relevance_score(self, row: dict[str, Any], terms: list[str]) -> float:
        haystack = " ".join(
            str(row.get(key) or "")
            for key in ("title", "content", "section_ref")
        ).casefold()
        score = 0.0
        for term in terms:
            value = term.casefold()
            if not value:
                continue

            if value.isascii():
                if re.search(rf"\b{re.escape(value)}\b", haystack):
                    score += 1.0
            elif value in haystack:
                score += 1.0

        return score

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
