"""
Hybrid retrieval for Agentic RAG.

Order of operations:
1. Chunk-level hybrid search RPC (semantic + keyword/RRF).
2. Direct keyword fallback for periods where embeddings are unavailable.
3. Legacy document-level hybrid search while older deployments migrate.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from core.jurisdiction import canonical_jurisdiction, contains_lao_script, contains_thai_script
from core.logging import get_logger
from rag.legal_text_matching import (
    extract_lao_legal_terms,
    normalise_search_text,
    table_of_contents_penalty,
    term_matches_text,
    unique_terms,
)

if TYPE_CHECKING:
    from supabase import AsyncClient  # pragma: no cover

log = get_logger(__name__)

LAO_LAND = "\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99"
LAO_ARTICLE = "\u0ea1\u0eb2\u0e94\u0e95\u0eb2"
LAO_RIGHT = "\u0eaa\u0eb4\u0e94"
LAO_LAND_USE_RIGHT = "\u0eaa\u0eb4\u0e94\u0e99\u0eb3\u0ec3\u0e8a\u0ec9"
LAO_LAND_USE_RIGHT_ALT = "\u0eaa\u0eb4\u0e94\u0e99\u0ecd\u0eb2\u0ec3\u0e8a\u0ec9"
LAO_LAND_USE_RIGHT_OCR = "\u0eaa\u0eb4\u0e94\u0e99\u0eb2\u0ecd\u0ec3\u0e8a\u0ec9"
LAO_PROTECTION = "\u0e9b\u0ebb\u0e81\u0e9b\u0ec9\u0ead\u0e87"
LAO_GUARD_RIGHT = "\u0eaa\u0eb4\u0e94\u0e9b\u0ebb\u0e81\u0e9b\u0eb1\u0e81\u0eae\u0eb1\u0e81\u0eaa\u0eb2"
LAO_USE_RIGHT = "\u0eaa\u0eb4\u0e94\u0ec3\u0e8a\u0ec9"
LAO_BENEFIT_RIGHT = "\u0eaa\u0eb4\u0e94\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a"
LAO_BENEFITS = "\u0e9c\u0ebb\u0e99\u0e9b\u0eb0\u0ec2\u0eab\u0e8d\u0e94"
LAO_TRANSFER_RIGHT = "\u0eaa\u0eb4\u0e94\u0ec2\u0ead\u0e99"
LAO_INHERIT_RIGHT = "\u0eaa\u0eb4\u0e94\u0eaa\u0eb7\u0e9a\u0e97\u0ead\u0e94"
THAI_ARTICLE = "\u0e21\u0e32\u0e15\u0e23\u0e32"


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
        chunk_rows = await self._chunk_search(
            query=query,
            embedding=embedding,
            jurisdiction=jurisdiction,
            tenant_id=tenant_id,
            top_k=top_k,
        )
        if chunk_rows:
            if self._should_supplement_keyword(query):
                keyword_rows = await self._direct_keyword_search(
                    query=query,
                    jurisdiction=jurisdiction,
                    top_k=top_k,
                )
                if keyword_rows:
                    log.info(
                        "retriever.chunk_search.keyword_supplement",
                        chunk_results=len(chunk_rows),
                        keyword_results=len(keyword_rows),
                        jurisdiction=jurisdiction,
                    )
                    return self._merge_rows(keyword_rows, chunk_rows, top_k)
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

    def _should_supplement_keyword(self, query: str) -> bool:
        terms = self._keyword_terms(query)
        return bool(self._article_targets_from_terms(terms))

    def _merge_rows(
        self,
        primary: list[dict[str, Any]],
        secondary: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for row in [*primary, *secondary]:
            key = str(row.get("chunk_id") or row.get("id") or f"{row.get('title')}|{str(row.get('content') or '')[:120]}")
            existing = merged.get(key)
            if not existing or self._row_score(row) > self._row_score(existing):
                merged[key] = row
        return sorted(merged.values(), key=self._row_score, reverse=True)[:top_k]

    def _row_score(self, row: dict[str, Any]) -> float:
        for key in ("final_score", "_rerank_score", "score"):
            try:
                return float(row.get(key))
            except (TypeError, ValueError):
                continue
        return 0.0

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
            return [self._normalise_row({**row, "retrieval_source": "chunk_rpc"}) for row in (result.data or [])]
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
                    return [
                        self._normalise_row({**row, "retrieval_source": "chunk_rpc_legacy_signature"})
                        for row in (result.data or [])
                    ]
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
        terms = self._rank_keyword_terms(self._keyword_terms(query))
        if not terms:
            return []

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for term in terms[:18]:
            safe_term = self._safe_ilike_term(term)
            if not safe_term:
                continue

            try:
                request = (
                    self._supabase.table("document_chunks")
                    .select(
                        "source_id, id, source_table, title, content, document_type, "
                        "jurisdiction, status, review_status, metadata, section_ref"
                    )
                    .eq("status", "active")
                    .eq("review_status", "approved")
                    .or_(f"title.ilike.%{safe_term}%,content.ilike.%{safe_term}%,section_ref.ilike.%{safe_term}%")
                    .limit(max(top_k, top_k * 2))
                )
                if jurisdiction:
                    request = request.eq("jurisdiction", jurisdiction)
                result = await request.execute()
                for row in result.data or []:
                    score = self._keyword_relevance_score(row, terms)
                    if score <= 0:
                        continue
                    normalised = self._normalise_row({
                        **row,
                        "final_score": score,
                        "retrieval_source": "direct_keyword",
                    })
                    key = str(normalised.get("chunk_id") or normalised.get("id"))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(normalised)
                    if len(rows) >= top_k:
                        return sorted(rows, key=self._row_score, reverse=True)[:top_k]
            except Exception as exc:
                log.debug("retriever.direct_keyword.term_failed", term=safe_term, error=str(exc))
        return sorted(rows, key=self._row_score, reverse=True)[:top_k]

    def _keyword_terms(self, query: str) -> list[str]:
        lowered = query.casefold()
        terms: list[str] = []

        for token in lowered.replace("\n", " ").split():
            cleaned = token.strip(".,;:()[]{}\"'!?")
            if len(cleaned) >= 2:
                terms.append(cleaned)

        terms.extend(extract_lao_legal_terms(query))

        for article in self._article_targets_from_text(lowered):
            terms.extend([
                f"{LAO_ARTICLE} {article}",
                f"{THAI_ARTICLE} {article}",
                f"Article {article}",
                f"Section {article}",
            ])

        if contains_lao_script(query):
            terms.extend([
                "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d",
                "\u0ea1\u0eb2\u0e94\u0e95\u0eb2",
                "\u0e94\u0eb3\u0ea5\u0eb1\u0e94",
                "\u0e84\u0eb3\u0eaa\u0eb1\u0ec8\u0e87",
            ])
        if contains_thai_script(query):
            terms.extend([
                "\u0e01\u0e0e\u0e2b\u0e21\u0e32\u0e22",
                "\u0e21\u0e32\u0e15\u0e23\u0e32",
                "\u0e1e\u0e23\u0e30\u0e23\u0e32\u0e0a\u0e1a\u0e31\u0e0d\u0e0d\u0e31\u0e15\u0e34",
            ])

        land_markers = (
            "\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99",
            "\u0e81\u0eb3\u0ea1\u0eb0\u0eaa\u0eb4\u0e94",
            "\u0eaa\u0eb4\u0e94\u0e99\u0eb3\u0ec3\u0e8a\u0ec9",
            LAO_LAND_USE_RIGHT_ALT,
            LAO_LAND_USE_RIGHT_OCR,
            "\u0ead\u0eb0\u0eaa\u0eb1\u0e87\u0eab\u0eb2",
            "\u0e17\u0e35\u0e48\u0e14\u0e34\u0e19",
            "\u0e01\u0e23\u0e23\u0e21\u0e2a\u0e34\u0e17\u0e18\u0e34\u0e4c",
            "land",
            "property",
            "ownership",
            "usufruct",
            "immovable",
        )
        if any(marker in lowered for marker in land_markers):
            terms.extend([
                "\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99",
                "\u0e81\u0eb3\u0ea1\u0eb0\u0eaa\u0eb4\u0e94",
                "\u0eaa\u0eb4\u0e94\u0e99\u0eb3\u0ec3\u0e8a\u0ec9",
                LAO_LAND_USE_RIGHT_ALT,
                LAO_LAND_USE_RIGHT_OCR,
                "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99",
                "\u0e17\u0e35\u0e48\u0e14\u0e34\u0e19",
                "land",
                "property",
                "ownership",
                "usufruct",
                "land use right",
                "immovable property",
            ])

        synonym_checks = [
            (
                "\u0e99\u0ec9\u0eb3",
                [
                    "\u0e99\u0ec9\u0eb3",
                    "\u0e99\u0ecd\u0ec9\u0eb2",
                    "\u0e99\u0eb2\u0ecd",
                    "\u0e99\u0eb2\u0ec9",
                    "\u0e9a\u0ecd\u0ea5\u0eb4\u0ec0\u0ea7\u0e99\u0e99\u0eb2\u0ecd",
                    "\u0e9a\u0ecd\u0ea5\u0eb4\u0ec0\u0ea7\u0e99\u0e99\u0eb2\u0ec9",
                    "\u0ec1\u0eab\u0ebc\u0ec8\u0e87\u0e99\u0eb2\u0ecd",
                    "\u0e97\u0eb2\u0e87\u0e99\u0eb2\u0ecd",
                    "water",
                    "water area",
                ],
            ),
            (
                "\u0e99\u0ecd\u0ec9\u0eb2",
                [
                    "\u0e99\u0ec9\u0eb3",
                    "\u0e99\u0ecd\u0ec9\u0eb2",
                    "\u0e99\u0eb2\u0ecd",
                    "\u0e99\u0eb2\u0ec9",
                    "\u0e9a\u0ecd\u0ea5\u0eb4\u0ec0\u0ea7\u0e99\u0e99\u0eb2\u0ecd",
                    "\u0e9a\u0ecd\u0ea5\u0eb4\u0ec0\u0ea7\u0e99\u0e99\u0eb2\u0ec9",
                    "water",
                    "water area",
                ],
            ),
            (
                "\u0ec0\u0e82\u0e94",
                [
                    "\u0ec0\u0e82\u0e94",
                    "\u0ec1\u0e9a\u0ec8\u0e87",
                    "\u0e88\u0eb1\u0e94\u0ec1\u0e9a\u0ec8\u0e87",
                    "\u0e9b\u0eb0\u0ec0\u0e9e\u0e94",
                    "\u0ec0\u0e82\u0e94\u0e97\u0ebb\u0ec8\u0e87",
                    "\u0ec0\u0e82\u0e94\u0e9e\u0eb9",
                    "zone",
                    "category",
                    "land type",
                ],
            ),
            (
                "\u0e9b\u0eb0\u0ec0\u0e9e\u0e94",
                [
                    "\u0e9b\u0eb0\u0ec0\u0e9e\u0e94",
                    "\u0ec0\u0e82\u0e94",
                    "\u0ec1\u0e9a\u0ec8\u0e87",
                    "\u0e88\u0eb1\u0e94\u0ec1\u0e9a\u0ec8\u0e87",
                    "\u0e9b\u0eb0\u0ec0\u0e9e\u0e94\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99",
                    "category",
                    "land type",
                    "zone",
                ],
            ),
            (
                "\u0e9b\u0ec8\u0ebd\u0e99",
                [
                    "\u0e9b\u0ec8\u0ebd\u0e99",
                    "\u0e9b\u0ebd\u0e99",
                    "\u0e9b\u0eb8\u0ec8\u0ebd\u0e99",
                    "\u0e97\u0eb1\u0e99\u0e9b\u0ebd\u0e99",
                    "\u0e97\u0eb1\u0e99\u0e9b\u0eb8\u0ec8\u0ebd\u0e99",
                    "\u0ead\u0eb0\u0e99\u0eb8\u0ea1\u0eb1\u0e94",
                    "\u0e9c\u0ebb\u0e99\u0e81\u0eb0\u0e97\u0ebb\u0e9a",
                    "\u0eaa\u0eb4\u0e87\u0ec1\u0ea7\u0e94",
                    "\u0e88\u0eb2\u0ecd\u0ec0\u0e9b\u0eb1\u0e99",
                    "change land type",
                    "approval",
                    "environmental impact",
                ],
            ),
            (
                "\u0ead\u0eb0\u0e99\u0eb8\u0ea1\u0eb1\u0e94",
                [
                    "\u0ead\u0eb0\u0e99\u0eb8\u0ea1\u0eb1\u0e94",
                    "\u0e9b\u0ec8\u0ebd\u0e99",
                    "\u0e9b\u0ebd\u0e99",
                    "\u0e9b\u0eb8\u0ec8\u0ebd\u0e99",
                    "\u0e9c\u0ebb\u0e99\u0e81\u0eb0\u0e97\u0ebb\u0e9a",
                    "\u0eaa\u0eb4\u0e87\u0ec1\u0ea7\u0e94",
                    "approval",
                ],
            ),
            (
                "\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2",
                [
                    "\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2",
                    "\u0e84\u0ec8\u0eb2\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2",
                    "\u0e9c\u0eb9\u0ec9\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2",
                    "lease",
                    "rent",
                    "tenant",
                    "\u0e40\u0e0a\u0e48\u0e32",
                    "\u0e04\u0e48\u0e32\u0e40\u0e0a\u0e48\u0e32",
                ],
            ),
            (
                "\u0e40\u0e0a\u0e48\u0e32",
                [
                    "\u0e40\u0e0a\u0e48\u0e32",
                    "\u0e04\u0e48\u0e32\u0e40\u0e0a\u0e48\u0e32",
                    "\u0e1c\u0e39\u0e49\u0e40\u0e0a\u0e48\u0e32",
                    "\u0e1c\u0e39\u0e49\u0e43\u0e2b\u0e49\u0e40\u0e0a\u0e48\u0e32",
                    "lease",
                    "rent",
                    "tenant",
                ],
            ),
            ("rent", ["rent", "lease", "tenant", "\u0e04\u0e48\u0e32\u0e40\u0e0a\u0e48\u0e32", "\u0e40\u0e0a\u0e48\u0e32", "\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2"]),
            ("lease", ["lease", "rent", "tenant", "\u0e04\u0e48\u0e32\u0e40\u0e0a\u0e48\u0e32", "\u0e40\u0e0a\u0e48\u0e32", "\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2"]),
            ("company", ["company", "enterprise", "shareholder", "director", "investment"]),
            ("labor", ["labor", "labour", "employment", "termination", "wage", "severance"]),
            ("tax", ["tax", "vat", "customs", "income", "declaration"]),
        ]
        for marker, synonyms in synonym_checks:
            if marker in lowered:
                terms.extend(synonyms)

        return unique_terms(terms)

    def _rank_keyword_terms(self, terms: list[str]) -> list[str]:
        article_targets = set(self._article_targets_from_terms(terms))
        generic_terms = {
            "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d",
            "\u0ea1\u0eb2\u0e94\u0e95\u0eb2",
            "\u0e94\u0eb3\u0ea5\u0eb1\u0e94",
            "\u0e84\u0eb3\u0eaa\u0eb1\u0ec8\u0e87",
            "\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99",
            "land",
            "property",
            "ownership",
        }

        def priority(term: str) -> tuple[int, int]:
            value = term.casefold().strip()
            if any(self._matches_article_term(value, target) for target in article_targets):
                return (0, -len(value))
            if value in generic_terms:
                return (4, -len(value))
            if not value.isascii() and len(value) >= 4:
                return (1, -len(value))
            if not value.isascii():
                return (2, -len(value))
            return (3, -len(value))

        return sorted(terms, key=priority)

    def _matches_article_term(self, term: str, target: str) -> bool:
        pattern = rf"(?:{LAO_ARTICLE}|{THAI_ARTICLE}|article|art\.?|section|sec\.?)\s*0*{re.escape(target)}(?:\D|$)"
        return bool(re.search(pattern, term, flags=re.IGNORECASE))

    def _safe_ilike_term(self, term: str) -> str | None:
        value = re.sub(r"[\x00\r\n,(){}\[\]_%]", " ", str(term)).strip()
        value = re.sub(r"\s+", " ", value)
        if len(value) < 2:
            return None
        return value[:80]

    def _keyword_relevance_score(self, row: dict[str, Any], terms: list[str]) -> float:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        haystack = " ".join(
            str(value or "")
            for value in (
                row.get("title"),
                row.get("content"),
                row.get("section_ref"),
                metadata.get("law_no"),
                metadata.get("article"),
            )
        ).casefold()
        normalised_haystack = normalise_search_text(haystack)
        heading = " ".join(
            str(value or "")
            for value in (
                row.get("title"),
                row.get("section_ref"),
                str(row.get("content") or "")[:280],
                metadata.get("article"),
            )
        ).casefold()
        normalised_heading = normalise_search_text(heading)
        score = 0.0
        for article in self._article_targets_from_terms(terms):
            if self._row_matches_article(row, article):
                score += 4.0
        for term in terms:
            value = term.casefold().strip()
            if not value:
                continue

            if term_matches_text(value, haystack, normalised_text=normalised_haystack):
                weight = self._keyword_term_weight(value)
                score += weight
                if term_matches_text(value, heading, normalised_text=normalised_heading):
                    score += min(0.75, weight * 0.55)

        if self._is_statute_like(row):
            score += 0.35
        if self._is_official_source(row):
            score += 0.45
        score -= table_of_contents_penalty(haystack)
        return score

    def _keyword_term_weight(self, term: str) -> float:
        value = normalise_search_text(term)
        generic_terms = {
            "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d",
            "\u0ea1\u0eb2\u0e94\u0e95\u0eb2",
            "\u0e94\u0eb3\u0ea5\u0eb1\u0e94",
            "\u0e84\u0eb3\u0eaa\u0eb1\u0ec8\u0e87",
            "\u0e97\u0eb5\u0e94\u0eb4\u0e99",
            "law",
            "land",
            "property",
            "ownership",
        }
        if value in generic_terms:
            return 0.35
        if value.isascii():
            return 0.7 if len(value) >= 5 else 0.35
        if len(value) >= 12:
            return 1.6
        if len(value) >= 7:
            return 1.15
        if len(value) >= 4:
            return 0.8
        return 0.35

    def _article_targets_from_terms(self, terms: list[str]) -> list[str]:
        targets: list[str] = []
        seen: set[str] = set()
        pattern = rf"(?:{LAO_ARTICLE}|{THAI_ARTICLE}|article|art\.?|section|sec\.?)\s*0*([0-9]{{1,4}})"
        for term in terms:
            for match in re.finditer(pattern, term, flags=re.IGNORECASE):
                target = match.group(1).lstrip("0") or "0"
                if target not in seen:
                    seen.add(target)
                    targets.append(target)
        return targets[:5]

    def _article_targets_from_text(self, text: str) -> list[str]:
        pattern = rf"(?:{LAO_ARTICLE}|{THAI_ARTICLE}|article|art\.?|section|sec\.?)\s*0*([0-9]{{1,4}})"
        targets: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            target = match.group(1).lstrip("0") or "0"
            if target not in seen:
                seen.add(target)
                targets.append(target)
        return targets[:5]

    def _row_matches_article(self, row: dict[str, Any], target: str) -> bool:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        text = " ".join(
            str(value or "")
            for value in (
                row.get("section"),
                row.get("section_ref"),
                row.get("content"),
                metadata.get("section"),
                metadata.get("article"),
            )
        )
        patterns = (
            rf"(?:{LAO_ARTICLE}|{THAI_ARTICLE}|article|art\.?|section|sec\.?)\s*0*{re.escape(target)}(?:\D|$)",
            rf"^0*{re.escape(target)}(?:\.|\s)",
        )
        return any(re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) for pattern in patterns)

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
            "query_embedding": embedding,
        }
        if jurisdiction:
            params["p_jurisdiction"] = jurisdiction

        result = await self._supabase.rpc("hybrid_legal_search", params).execute()
        data = [
            self._normalise_row({**row, "retrieval_source": "legacy_hybrid_rpc"})
            for row in (result.data or [])
        ]

        log.info("retriever.hybrid_search.ok", results=len(data), jurisdiction=jurisdiction)
        return data

    def _normalise_row(self, row: dict[str, Any]) -> dict[str, Any]:
        source_table = str(row.get("source_table") or "").lower()
        doc_type = str(row.get("doc_type") or row.get("document_type") or "").lower()
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}

        normalised_type = str(row.get("type") or doc_type or source_table or "doc").lower()
        if source_table == "cases" and self._has_statute_text_signal(row):
            normalised_type = "law"
        elif source_table == "cases":
            normalised_type = "case"
        elif source_table == "laws":
            normalised_type = "law"
        elif source_table == "legal_forms":
            normalised_type = "form"

        source_id = row.get("source_id") or metadata.get("source_id") or row.get("document_id") or row.get("id")
        chunk_id = row.get("chunk_id") or metadata.get("chunk_id") or row.get("id")
        source_url = (
            row.get("source_url")
            or row.get("official_source_url")
            or metadata.get("source_url")
            or metadata.get("official_source_url")
        )

        return {
            **row,
            "id": source_id,
            "type": normalised_type,
            "section": row.get("section") or row.get("section_number") or row.get("section_ref") or metadata.get("section"),
            "chunk_id": chunk_id,
            "source_id": source_id,
            "source_url": source_url,
            "official_source_url": row.get("official_source_url") or metadata.get("official_source_url"),
            "source_authority": row.get("source_authority") or metadata.get("source_authority"),
        }

    def _is_statute_like(self, row: dict[str, Any]) -> bool:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        values = " ".join(
            str(value or "")
            for value in (
                row.get("type"),
                row.get("doc_type"),
                row.get("document_type"),
                row.get("source_table"),
                metadata.get("document_type"),
            )
        ).casefold()
        return any(word in values for word in ("law", "laws", "statute", "regulation", "decree")) or self._has_statute_text_signal(row)

    def _is_official_source(self, row: dict[str, Any]) -> bool:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        values = " ".join(
            str(value or "")
            for value in (
                row.get("source_url"),
                row.get("official_source_url"),
                row.get("source_authority"),
                metadata.get("source_url"),
                metadata.get("official_source_url"),
                metadata.get("source_authority"),
            )
        ).casefold()
        return "laoofficialgazette.gov.la" in values or "official" in values

    def _has_statute_text_signal(self, row: dict[str, Any]) -> bool:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        text = " ".join(
            str(value or "")
            for value in (
                row.get("title"),
                row.get("section"),
                row.get("section_ref"),
                row.get("content"),
                metadata.get("law_no"),
                metadata.get("article"),
            )
        ).casefold()
        markers = (
            "law",
            "article",
            "decree",
            "regulation",
            "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d",
            "\u0ea1\u0eb2\u0e94\u0e95\u0eb2",
            "\u0e94\u0eb3\u0ea5\u0eb1\u0e94",
            "\u0e01\u0e0e\u0e2b\u0e21\u0e32\u0e22",
            "\u0e21\u0e32\u0e15\u0e23\u0e32",
        )
        return any(marker in text for marker in markers)
