"""
Hybrid retrieval for Agentic RAG.

Order of operations:
1. Chunk-level hybrid search RPC (semantic + keyword/RRF).
2. Direct keyword fallback for periods where embeddings are unavailable.
3. Legacy document-level hybrid search while older deployments migrate.
"""
from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from core.config import get_settings
from core.jurisdiction import canonical_jurisdiction, contains_lao_script, contains_thai_script
from core.logging import get_logger
from rag.legal_text_matching import (
    extract_lao_legal_terms,
    normalise_search_text,
    prepare_lao_fts_query,
    table_of_contents_penalty,
    term_matches_text,
    unique_terms,
)
from rag.retrieval_cache import deserialise_chunks, retrieval_cache_key, serialise_chunks

if TYPE_CHECKING:
    from supabase import AsyncClient  # pragma: no cover
    import redis.asyncio as aioredis

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
    def __init__(
        self,
        supabase: "AsyncClient | None" = None,
        redis: "aioredis.Redis | None" = None,
    ) -> None:
        self._supabase = supabase
        self._redis = redis
        self._settings = get_settings()
        self._chunk_search_supports_tenant_param: bool | None = None
        self._dedicated_rpcs_available: bool | None = None

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

        canonical = canonical_jurisdiction(jurisdiction)
        effective_tenant_id = tenant_id or self._settings.default_tenant_id
        cache_key = retrieval_cache_key(
            query=query,
            jurisdiction=canonical,
            tenant_id=effective_tenant_id,
            top_k=top_k,
            embedded=embedding is not None,
        )
        if self._redis:
            try:
                cached = await self._redis.get(cache_key)
                if cached:
                    rows = deserialise_chunks(cached.decode() if isinstance(cached, bytes) else cached)
                    if rows:
                        log.debug("retriever.cache_hit", results=len(rows), jurisdiction=canonical)
                        return rows[:top_k]
            except Exception as exc:
                log.debug("retriever.cache_get.failed", error=str(exc))

        try:
            rows = await self._hybrid_search(
                query=query,
                embedding=embedding,
                jurisdiction=canonical,
                tenant_id=tenant_id,
                top_k=top_k,
            )
        except Exception as exc:
            log.warning("retriever.search.failed", error=str(exc))
            return []

        if self._redis and rows:
            try:
                await self._redis.setex(
                    cache_key,
                    self._settings.cache_ttl_retrieval_seconds,
                    serialise_chunks(rows),
                )
            except Exception as exc:
                log.debug("retriever.cache_set.failed", error=str(exc))
        return rows

    async def _hybrid_search(
        self,
        *,
        query: str,
        embedding: list[float] | None,
        jurisdiction: str | None,
        tenant_id: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        # Run the title fast-path, direct-article, and semantic+lexical chunk legs
        # CONCURRENTLY and fuse them. We deliberately do NOT short-circuit on title
        # matches: a title hint can match a tangential statute (e.g. an authority-
        # hint expansion), and short-circuiting there would starve the candidate set
        # so the genuinely relevant law (found via semantic/lexical search) never
        # surfaces. Semantic search is index-backed and fast, so always run it.
        title_rows, article_rows, chunk_rows = await asyncio.gather(
            self._title_statute_search(
                query=query,
                jurisdiction=jurisdiction,
                tenant_id=tenant_id,
                top_k=top_k,
            ),
            self._direct_article_search(
                query=query,
                jurisdiction=jurisdiction,
                tenant_id=tenant_id,
                top_k=top_k,
            ),
            self._chunk_search(
                query=query,
                embedding=embedding,
                jurisdiction=jurisdiction,
                tenant_id=tenant_id,
                top_k=top_k,
            ),
        )

        if chunk_rows or article_rows:
            combined = self._merge_rows(article_rows, chunk_rows, title_rows, top_k)
            if chunk_rows and self._should_supplement_keyword(query):
                keyword_rows = await self._direct_keyword_search(
                    query=query,
                    jurisdiction=jurisdiction,
                    tenant_id=tenant_id,
                    top_k=top_k,
                )
                if keyword_rows:
                    log.info(
                        "retriever.chunk_search.keyword_supplement",
                        chunk_results=len(combined),
                        keyword_results=len(keyword_rows),
                        jurisdiction=jurisdiction,
                    )
                    return self._merge_rows(keyword_rows, combined, top_k)
            if combined:
                source = "chunk_rpc" if chunk_rows else "article_fast_path"
                log.info("retriever.search.ok", results=len(combined), jurisdiction=jurisdiction, source=source)
                return combined

        keyword_rows = await self._direct_keyword_search(
            query=query,
            jurisdiction=jurisdiction,
            tenant_id=tenant_id,
            top_k=top_k,
        )
        if keyword_rows:
            log.info("retriever.direct_keyword.ok", results=len(keyword_rows), jurisdiction=jurisdiction)
            return self._merge_rows(title_rows, keyword_rows, top_k)

        if title_rows:
            log.info("retriever.search.ok", results=len(title_rows), jurisdiction=jurisdiction, source="title_only")
            return title_rows[:top_k]

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

    def _merge_rows(self, *lists_and_top_k: Any) -> list[dict[str, Any]]:
        top_k = 10
        row_lists: list[list[dict[str, Any]]] = []
        for item in lists_and_top_k:
            if isinstance(item, int):
                top_k = item
            elif isinstance(item, list):
                row_lists.append(item)
        merged: dict[str, dict[str, Any]] = {}
        for rows in row_lists:
            for row in rows:
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

    def _title_hints_from_query(self, query: str) -> list[str]:
        text = str(query or "").strip()
        if not text:
            return []

        hints: list[str] = []
        subject_match = re.search(
            r"ກົດໝາຍວ່າດ້ວຍ\s+(.+?)(?:\s+ກຳນົດ|\s+ແມ່ນ|\?|$)",
            text,
            flags=re.IGNORECASE,
        )
        if subject_match:
            hints.append(subject_match.group(1).strip()[:48])

        skip = {"ກົດໝາຍ", "ວ່າດ້ວຍ", "ກຳນົດ", "ແນວໃດ", "ສະບັບ", "ປັບປຸງ"}
        for token in re.findall(r"[\u0e80-\u0eff]{3,}", text):
            if token not in skip:
                hints.append(token)

        hints.extend(extract_lao_legal_terms(text))
        return unique_terms(hints)[:4]

    async def _title_statute_search(
        self,
        *,
        query: str,
        jurisdiction: str | None,
        tenant_id: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not self._supabase:
            return []

        hints = self._title_hints_from_query(query)
        if not hints:
            return []

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hint in hints:
            safe_hint = re.sub(r"[%_,()]", " ", hint).strip()
            if len(safe_hint) < 3:
                continue
            for scope in self._keyword_tenant_scopes(tenant_id):
                try:
                    request = (
                        self._supabase.table("document_chunks")
                        .select(
                            "source_id, id, tenant_id, source_table, title, content, document_type, "
                            "jurisdiction, status, review_status, metadata, section_ref"
                        )
                        .eq("status", "active")
                        .eq("review_status", "approved")
                        .ilike("title", f"%{safe_hint}%")
                        .limit(max(top_k, top_k * 2))
                    )
                    if jurisdiction:
                        request = request.eq("jurisdiction", jurisdiction)
                    if scope:
                        request = request.eq("tenant_id", scope)
                    else:
                        request = request.is_("tenant_id", "null")
                    result = await request.execute()
                except Exception as exc:
                    log.debug("retriever.title_search.failed", hint=hint, error=str(exc))
                    continue

                for row in result.data or []:
                    normalised = self._normalise_row({**row, "retrieval_source": "title_fast_path", "score": 0.85})
                    key = str(normalised.get("chunk_id") or normalised.get("id"))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(normalised)
                if len(rows) >= top_k:
                    break
            if len(rows) >= top_k:
                break

        return sorted(rows, key=self._row_score, reverse=True)[:top_k]

    async def _chunk_search(
        self,
        *,
        query: str,
        embedding: list[float] | None,
        jurisdiction: str | None,
        tenant_id: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        # Semantic-first: dedicated single-leg RPCs are index-backed (HNSW + GIN)
        # and fast. They make meaning-based retrieval the primary signal instead of
        # degrading to keyword matching. Fall back to the combined hybrid RPC, then
        # to keyword search, if the dedicated functions are not deployed yet.
        if self._dedicated_rpcs_available is not False:
            dedicated = await self._semantic_first_search(
                query=query,
                embedding=embedding,
                jurisdiction=jurisdiction,
                tenant_id=tenant_id,
                top_k=top_k,
            )
            if dedicated is not None:
                return dedicated

        return await self._hybrid_rpc_search(
            query=query,
            embedding=embedding,
            jurisdiction=jurisdiction,
            tenant_id=tenant_id,
            top_k=top_k,
        )

    async def _semantic_first_search(
        self,
        *,
        query: str,
        embedding: list[float] | None,
        jurisdiction: str | None,
        tenant_id: str | None,
        top_k: int,
    ) -> list[dict[str, Any]] | None:
        """Run dedicated semantic + lexical RPCs in parallel and fuse them.

        Returns a (possibly empty) list when the dedicated RPCs exist, or None
        when they are missing so the caller can fall back to the hybrid RPC.
        """
        settings = get_settings()
        effective_tenant_id = tenant_id or settings.default_tenant_id
        fts_query = prepare_lao_fts_query(query, jurisdiction=jurisdiction)

        base: dict[str, Any] = {
            "p_status": "active",
            "p_review_status": "approved",
            "p_tenant_id": effective_tenant_id,
            "p_law_category": None,
            "match_count": top_k,
        }
        if jurisdiction:
            base["p_jurisdiction"] = jurisdiction

        tasks: list[Any] = []
        legs: list[str] = []
        if embedding:
            sem_params = {**base, "query_embedding": embedding}
            tasks.append(self._supabase.rpc("semantic_document_chunk_search", sem_params).execute())
            legs.append("semantic")
        lex_params = {**base, "query_text": fts_query}
        tasks.append(self._supabase.rpc("lexical_document_chunk_search", lex_params).execute())
        legs.append("lexical")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        row_lists: list[list[dict[str, Any]]] = []
        missing = False
        for leg, result in zip(legs, results):
            if isinstance(result, Exception):
                message = str(result)
                if self._rpc_missing(message):
                    missing = True
                    log.info("retriever.dedicated_rpc.absent", leg=leg)
                else:
                    log.warning("retriever.dedicated_rpc.failed", leg=leg, error=message)
                continue
            row_lists.append([
                self._normalise_row({**row, "retrieval_source": f"chunk_{leg}"})
                for row in (result.data or [])
            ])

        if missing and not row_lists:
            self._dedicated_rpcs_available = False
            return None

        self._dedicated_rpcs_available = True
        if not row_lists:
            return []
        merged = self._merge_rows(*row_lists, top_k)
        log.info(
            "retriever.semantic_first.ok",
            results=len(merged),
            jurisdiction=jurisdiction,
            legs=",".join(legs),
        )
        return merged

    @staticmethod
    def _rpc_missing(message: str) -> bool:
        lowered = message.lower()
        return (
            "pgrst202" in lowered
            or "could not find the function" in lowered
            or "does not exist" in lowered
            or "schema cache" in lowered
        )

    async def _hybrid_rpc_search(
        self,
        *,
        query: str,
        embedding: list[float] | None,
        jurisdiction: str | None,
        tenant_id: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        settings = get_settings()
        fts_query = prepare_lao_fts_query(query, jurisdiction=jurisdiction)
        tenant_scopes = self._keyword_tenant_scopes(tenant_id)

        for scope in tenant_scopes:
            effective_tenant_id = scope or settings.default_tenant_id
            params: dict[str, Any] = {
                "query_text": fts_query,
                "match_count": top_k,
                "rrf_k": 60,
                "p_status": "active",
                "p_review_status": "approved",
                "p_tenant_id": effective_tenant_id,
                "p_law_category": None,
            }
            if embedding:
                params["query_embedding"] = embedding
            if jurisdiction:
                params["p_jurisdiction"] = jurisdiction

            try:
                result = await self._supabase.rpc("hybrid_document_chunk_search", params).execute()
                self._chunk_search_supports_tenant_param = True
                rows = [
                    self._normalise_row({**row, "retrieval_source": "chunk_rpc"})
                    for row in (result.data or [])
                ]
                if rows:
                    return rows
            except Exception as exc:
                if "p_law_category" in str(exc) or "p_tenant_id" in str(exc):
                    fallback_params = {
                        key: value for key, value in params.items()
                        if key not in {"p_law_category", "p_tenant_id"}
                    }
                    try:
                        result = await self._supabase.rpc("hybrid_document_chunk_search", fallback_params).execute()
                        self._chunk_search_supports_tenant_param = False
                        rows = [
                            self._normalise_row({**row, "retrieval_source": "chunk_rpc"})
                            for row in (result.data or [])
                        ]
                        if rows:
                            return rows
                    except Exception as fallback_exc:
                        log.warning("retriever.chunk_search.failed", error=str(fallback_exc), tenant_scope=scope)
                        continue

                log.warning("retriever.chunk_search.failed", error=str(exc), tenant_scope=scope)
                continue

        return []

    async def _direct_article_search(
        self,
        *,
        query: str,
        jurisdiction: str | None,
        tenant_id: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not self._supabase:
            return []

        targets = self._article_targets_from_text(query.casefold())
        if not targets:
            return []

        tasks: list[Any] = []
        for target in targets[:4]:
            for prefix in (f"{LAO_ARTICLE} {target}", f"{THAI_ARTICLE} {target}"):
                for scope in self._keyword_tenant_scopes(tenant_id):
                    tasks.append(
                        self._fetch_section_ref_matches(
                            section_prefix=prefix,
                            jurisdiction=jurisdiction,
                            tenant_id=scope,
                            top_k=top_k,
                        )
                    )

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for batch in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(batch, Exception):
                continue
            for row in batch:
                key = str(row.get("chunk_id") or row.get("id"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

        return sorted(rows, key=self._row_score, reverse=True)[:top_k]

    async def fetch_related_articles(
        self,
        *,
        article_numbers: list[str],
        source_ids: list[str] | None = None,
        jurisdiction: str | None = None,
        tenant_id: str | None = None,
        top_k: int = 12,
    ) -> list[dict[str, Any]]:
        """Fetch specific articles (referenced or adjacent) — like a lawyer pulling
        the surrounding provisions of a statute instead of reading one line in isolation.

        Returns normalised chunk rows tagged with retrieval_source="cross_reference".
        """
        if not self._supabase or not article_numbers:
            return []

        canonical = canonical_jurisdiction(jurisdiction)
        wanted = []
        seen_targets: set[str] = set()
        for raw in article_numbers:
            target = str(raw).strip().lstrip("0") or "0"
            if target and target not in seen_targets:
                seen_targets.add(target)
                wanted.append(target)
        source_filter = {str(s) for s in (source_ids or []) if s}

        tasks: list[Any] = []
        for target in wanted[:12]:
            for prefix in (f"{LAO_ARTICLE} {target}", f"{THAI_ARTICLE} {target}"):
                for scope in self._keyword_tenant_scopes(tenant_id):
                    tasks.append(
                        self._fetch_section_ref_matches(
                            section_prefix=prefix,
                            jurisdiction=canonical,
                            tenant_id=scope,
                            top_k=top_k,
                            retrieval_source="cross_reference",
                        )
                    )

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for batch in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(batch, Exception):
                continue
            for row in batch:
                if source_filter:
                    row_source = str(row.get("source_id") or "")
                    if row_source and row_source not in source_filter:
                        continue
                key = str(row.get("chunk_id") or row.get("id"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
        return rows[:top_k]

    async def _fetch_section_ref_matches(
        self,
        *,
        section_prefix: str,
        jurisdiction: str | None,
        tenant_id: str | None,
        top_k: int,
        retrieval_source: str = "article_fast_path",
    ) -> list[dict[str, Any]]:
        if not self._supabase:
            return []

        request = (
            self._supabase.table("document_chunks")
            .select(
                "source_id, id, tenant_id, source_table, title, content, document_type, "
                "jurisdiction, status, review_status, metadata, section_ref"
            )
            .eq("status", "active")
            .eq("review_status", "approved")
            .ilike("section_ref", f"{section_prefix}%")
            .limit(max(top_k, top_k * 2))
        )
        if jurisdiction:
            request = request.eq("jurisdiction", jurisdiction)
        if tenant_id:
            request = request.eq("tenant_id", tenant_id)
        else:
            request = request.is_("tenant_id", "null")

        try:
            result = await request.execute()
        except Exception as exc:
            log.debug("retriever.article_search.failed", section_prefix=section_prefix, error=str(exc))
            return []

        rows: list[dict[str, Any]] = []
        score = 4.8 if retrieval_source == "article_fast_path" else 3.2
        for row in result.data or []:
            rows.append(self._normalise_row({
                **row,
                "final_score": score,
                "retrieval_source": retrieval_source,
            }))
        return rows

    async def _direct_keyword_search(
        self,
        *,
        query: str,
        jurisdiction: str | None,
        tenant_id: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        terms = self._rank_keyword_terms(self._keyword_terms(query))
        if not terms:
            return []

        async def _search_term(term: str) -> list[dict[str, Any]]:
            safe_term = self._safe_ilike_term(term)
            if not safe_term:
                return []

            local_rows: list[dict[str, Any]] = []
            for scope in self._keyword_tenant_scopes(tenant_id):
                try:
                    request = self._document_chunks_keyword_request(
                        safe_term=safe_term,
                        jurisdiction=jurisdiction,
                        tenant_id=scope,
                        top_k=top_k,
                    )
                    result = await request.execute()
                    for row in result.data or []:
                        score = self._keyword_relevance_score(row, terms)
                        if score <= 0:
                            continue
                        local_rows.append(self._normalise_row({
                            **row,
                            "final_score": score,
                            "retrieval_source": "direct_keyword",
                        }))
                except Exception as exc:
                    log.debug(
                        "retriever.direct_keyword.term_failed",
                        term=safe_term,
                        tenant_scope=scope or "public",
                        error=str(exc),
                    )
            return local_rows

        batches = await asyncio.gather(
            *[_search_term(term) for term in terms[:18]],
            return_exceptions=True,
        )

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for batch in batches:
            if isinstance(batch, Exception):
                continue
            for normalised in batch:
                key = str(normalised.get("chunk_id") or normalised.get("id"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(normalised)
        return sorted(rows, key=self._row_score, reverse=True)[:top_k]

    def _keyword_tenant_scopes(self, tenant_id: str | None) -> tuple[str | None, ...]:
        default_tenant_id = get_settings().default_tenant_id
        effective = tenant_id or default_tenant_id
        if tenant_id:
            return (tenant_id, default_tenant_id, None)
        return (default_tenant_id, None)

    def _document_chunks_keyword_request(
        self,
        *,
        safe_term: str,
        jurisdiction: str | None,
        tenant_id: str | None,
        top_k: int,
    ):
        request = (
            self._supabase.table("document_chunks")
            .select(
                "source_id, id, tenant_id, source_table, title, content, document_type, "
                "jurisdiction, status, review_status, metadata, section_ref"
            )
            .eq("status", "active")
            .eq("review_status", "approved")
            .or_(f"title.ilike.%{safe_term}%,content.ilike.%{safe_term}%,section_ref.ilike.%{safe_term}%")
            .limit(max(top_k, top_k * 2))
        )
        if jurisdiction:
            request = request.eq("jurisdiction", jurisdiction)
        if tenant_id:
            return request.eq("tenant_id", tenant_id)
        return request.is_("tenant_id", "null")

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
            "\u0e94\u0eb4\u0e99",
            "\u0e99\u0ecd\u0eb2\u0ec3\u0e8a\u0ec9\u0e94\u0eb4\u0e99",
            "\u0e99\u0eb3\u0ec3\u0e8a\u0ec9\u0e94\u0eb4\u0e99",
            "\u0e99\u0eb2\u0ecd\u0ec3\u0e8a\u0ec9\u0e94\u0eb4\u0e99",
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
                "\u0e94\u0eb4\u0e99",
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
            "law_category": row.get("law_category") or metadata.get("law_category"),
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
