"""
agents/research_agent.py
=========================
Legal Research Agent — CORE agent, runs on every query.

Responsibilities:
1. Generate query embedding
2. Hybrid search (semantic + keyword RRF) via Supabase pgvector + FTS
3. Cross-reference & adjacent-article expansion (read surrounding provisions)
4. Case law graph expansion (precedent chains, non-Lao jurisdictions)
5. Heuristic relevance reranking (article/authority/structure boosts)
6. Assemble structured legal context for IRAC agent

Output schema:
    retrieved_documents: list of ranked legal chunks
    case_graph_context:  related precedents from graph traversal
    memory_highlights:   relevant past case facts
"""
from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from agents.base_agent import BaseAgent
from core.config import get_settings
from core.exceptions import ProviderNotConfiguredError
from core.jurisdiction import canonical_jurisdiction, needs_multilingual_embedding
from core.logging import get_logger
from rag.agentic_planner import AgenticRetrievalPlanner, RetrievalQuery
from rag.embedder import Embedder
from rag.graph_expander import GraphExpander
from rag.legal_query_analyzer import LegalQueryAnalysis, LegalQueryAnalyzer
from rag.reranker import Reranker
from rag.retriever import Retriever
from rag.smart_query_expander import SmartQueryExpander, merge_understanding_into_analysis

log = get_logger(__name__)

# Distinctive Lao title keyword(s) per practice area. Used to deterministically
# boost chunks whose statute title contains the keyword during reranking, even
# when the LLM phrases the candidate law differently or the multilingual
# embedding is weak on Lao. Keywords (not full law names) keep this robust to the
# exact wording / version suffix of each statute's title.
_PRACTICE_AREA_LAO_STATUTE = {
    "land": ["\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99"],                       # ທີ່ດິນ
    "labor": ["\u0ec1\u0eae\u0e87\u0e87\u0eb2\u0e99"],                      # ແຮງງານ
    "family": ["\u0e84\u0ead\u0e9a\u0e84\u0ebb\u0ea7"],                    # ຄອບຄົວ
    "criminal": ["\u0ead\u0eb2\u0e8d\u0eb2"],                              # ອາຍາ
    "tax": ["\u0ead\u0eb2\u0e81\u0ead\u0e99"],                             # ອາກອນ
    "company": ["\u0ea7\u0eb4\u0eaa\u0eb2\u0eab\u0eb0\u0e81\u0eb4\u0e94"],    # ວິສາຫະກິດ
    "investment": ["\u0e81\u0eb2\u0e99\u0ea5\u0ebb\u0e87\u0e97\u0eb6\u0e99"],  # ການລົງທຶນ
    "environment": ["\u0eaa\u0eb4\u0ec8\u0e87\u0ec1\u0ea7\u0e94\u0ec9\u0ead\u0ea1"],  # ສິ່ງແວດລ້ອມ
    "immigration": ["\u0e84\u0ebb\u0e99\u0e95\u0ec8\u0eb2\u0e87\u0e94\u0ec9\u0eb2\u0ea7"],  # ຄົນຕ່າງດ້າວ
    "education": ["\u0eaa\u0eb6\u0e81\u0eaa\u0eb2"],                         # ສຶກສາ
    "health": ["\u0e8d\u0eb2", "\u0e9b\u0eb4\u0ec8\u0e99\u0e9b\u0ebb\u0ea7", "\u0eaa\u0eb8\u0e82\u0eb0\u0e9e\u0eb2\u0e9a"],  # ຢາ, ປິ່ນປົວ, ສຸຂະພາບ
}

if TYPE_CHECKING:
    from supabase import AsyncClient  # pragma: no cover
    import redis.asyncio as aioredis


class LegalResearchAgent(BaseAgent):
    name = "research"

    def __init__(
        self,
        *,
        supabase: "AsyncClient | None" = None,
        redis: "aioredis.Redis | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._embedder = Embedder(redis=redis)
        self._retriever = Retriever(supabase=supabase, redis=redis)
        self._graph = GraphExpander(supabase=supabase)
        self._reranker = Reranker()
        self._planner = AgenticRetrievalPlanner()
        self._query_analyzer = LegalQueryAnalyzer()
        self._query_expander = SmartQueryExpander(llm=self._llm)
        self._embedding_unavailable = False

    async def _execute(
        self,
        *,
        question: str,
        memory: dict,
        jurisdiction: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        settings = get_settings()
        canonical_jurisdiction_value = canonical_jurisdiction(jurisdiction)
        query_analysis = self._query_analyzer.analyze(
            question,
            jurisdiction=canonical_jurisdiction_value,
            memory=memory,
        )
        effective_jurisdiction = query_analysis.jurisdiction or canonical_jurisdiction_value

        # Smart layer: let a cheap LLM understand colloquial questions and map them
        # to Lao legal concepts / candidate statutes / articles + a HyDE passage that
        # makes the semantic embedding far stronger. Pure rule-based keyword analysis
        # alone returns "general" for natural-language questions and retrieves poorly.
        analysis_dict = query_analysis.to_dict()
        hyde_passage = ""
        if settings.rag_llm_query_understanding:
            try:
                understanding = await self._query_expander.expand(question)
                analysis_dict = merge_understanding_into_analysis(analysis_dict, understanding)
                hyde_passage = understanding.hyde_passage
            except Exception as exc:  # noqa: BLE001
                log.info("research.query_understanding.skipped", error=str(exc))

        chunks, retrieval_trace, embedding_tokens, retrieval_coverage = await self._agentic_retrieve(
            question=question,
            jurisdiction=effective_jurisdiction,
            query_analysis=query_analysis,
            analysis_dict=analysis_dict,
            hyde_passage=hyde_passage,
            tenant_id=tenant_id,
            top_k=max(settings.rag_top_k, settings.rag_top_k * 2),
        )

        # Cross-reference & adjacent-article expansion: a senior lawyer never reads a
        # single article in isolation — they pull the provisions it references and the
        # neighbouring articles (definitions, exceptions, penalties).
        related_chunks = await self._expand_cross_references(
            chunks=chunks,
            question=question,
            query_analysis=query_analysis,
            jurisdiction=effective_jurisdiction,
            tenant_id=tenant_id,
        )
        if related_chunks:
            before = len(chunks)
            chunks = self._dedupe_chunks(chunks + related_chunks)
            retrieval_trace.append({
                "purpose": "cross_reference_expansion",
                "mode": "related_articles",
                "added": len(chunks) - before,
            })

        # Step 3: Graph expansion from top case hits
        top_case_ids = [
            str(c.get("source_id") or c.get("id")) for c in chunks[:5]
            if c.get("type") == "case" and (c.get("source_id") or c.get("id"))
        ]
        graph_results = []
        if top_case_ids and effective_jurisdiction != "laos":
            graph_results = await self._graph.expand(
                case_ids=top_case_ids,
                depth=settings.graph_depth,
            )

        # Step 4: Rerank combined results. Feed the question-understanding signals
        # (candidate statutes + legal concepts) so the right law is ranked first
        # even when the multilingual embedding is weak on Lao.
        all_chunks = chunks + graph_results
        # Precision boost: only the canonical statute for the *determined* practice
        # area. The LLM's raw candidate-law list is great for recall (it drives the
        # title fast-path) but too noisy for ranking — e.g. it suggests the Land Law
        # for a divorce "property division" question — so we keep it out of the boost.
        focus_titles: list[str] = list(
            _PRACTICE_AREA_LAO_STATUTE.get(str(analysis_dict.get("practice_area") or ""), [])
        )
        focus_terms = list(analysis_dict.get("llm_legal_concepts") or [])
        reranked = await self._reranker.rerank(
            query=question,
            chunks=all_chunks,
            top_k=settings.rag_top_k,
            focus_titles=focus_titles,
            focus_terms=focus_terms,
        )
        final_coverage = self._planner.assess_coverage(reranked, effective_jurisdiction)

        # Step 5: Build memory highlight summary
        memory_highlights = self._extract_memory_highlights(memory)

        log.info(
            "research.done",
            chunks_retrieved=len(chunks),
            graph_nodes=len(graph_results),
            final_chunks=len(reranked),
        )

        return {
            "retrieved_documents": reranked,
            "case_graph_context": graph_results,
            "memory_highlights": memory_highlights,
            "query_analysis": query_analysis.to_dict(),
            "retrieval": {
                "source": self._retrieval_source(reranked),
                "count": len(reranked),
                "jurisdiction": effective_jurisdiction,
                "trace": retrieval_trace,
                "coverage": {
                    **final_coverage.metrics,
                    "enough_results": final_coverage.enough_results,
                    "has_statute": final_coverage.has_statute,
                    "has_official_source": final_coverage.has_official_source,
                    "has_clean_text": final_coverage.has_clean_text,
                    "reason": final_coverage.reason or retrieval_coverage.reason,
                },
            },
            "_confidence": min(1.0, len(reranked) / max(1, settings.rag_top_k)),
            "_tokens": embedding_tokens,
        }

    async def _agentic_retrieve(
        self,
        *,
        question: str,
        jurisdiction: str | None,
        query_analysis: LegalQueryAnalysis,
        analysis_dict: dict[str, Any] | None = None,
        hyde_passage: str = "",
        tenant_id: str | None,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, Any]:
        analysis = analysis_dict if analysis_dict is not None else query_analysis.to_dict()
        plan = self._planner.plan(question, jurisdiction, analysis=analysis)

        # HyDE: a hypothetical statutory passage embeds much closer to the real law
        # text than a short colloquial question, dramatically improving semantic recall.
        concepts = analysis.get("llm_legal_concepts") or []
        hyde_text = " ".join(part for part in (question, hyde_passage, " ".join(concepts)) if part).strip()
        if hyde_passage or concepts:
            plan.insert(
                0,
                RetrievalQuery(
                    query=hyde_text,
                    purpose="hyde_semantic_primary",
                    jurisdiction=jurisdiction,
                    priority=0,
                    required=True,
                    metadata={"authority": "hyde", "mode": "semantic"},
                ),
            )

        chunks, trace, tokens = await self._run_retrieval_plan(plan, tenant_id=tenant_id, top_k=top_k)
        chunks = self._dedupe_chunks(chunks)
        coverage = self._planner.assess_coverage(chunks, jurisdiction)

        if coverage.should_second_pass:
            second_pass = self._planner.second_pass(question, jurisdiction, analysis=analysis)
            more_chunks, more_trace, more_tokens = await self._run_retrieval_plan(
                second_pass,
                tenant_id=tenant_id,
                top_k=top_k,
            )
            chunks.extend(more_chunks)
            chunks = self._dedupe_chunks(chunks)
            trace.extend(more_trace)
            tokens += more_tokens
            coverage = self._planner.assess_coverage(chunks, jurisdiction)

        trace.append({
            "purpose": "coverage_assessment",
            "jurisdiction": jurisdiction,
            "results": len(chunks),
            "mode": "agentic_quality_gate",
            "reason": coverage.reason,
            **coverage.metrics,
        })
        return chunks, trace, tokens, coverage

    _ARTICLE_REF_RE = re.compile(
        r"(?:\u0ea1\u0eb2\u0e94\u0e95\u0eb2|\u0e21\u0e32\u0e15\u0e23\u0e32|article|art\.?|section|sec\.?)\s*0*(\d{1,4})",
        flags=re.IGNORECASE,
    )

    async def _expand_cross_references(
        self,
        *,
        chunks: list[dict[str, Any]],
        question: str,
        query_analysis: LegalQueryAnalysis,
        jurisdiction: str | None,
        tenant_id: str | None,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []

        analysis = query_analysis.to_dict()
        existing_ids = {
            str(c.get("chunk_id") or c.get("id"))
            for c in chunks
            if c.get("chunk_id") or c.get("id")
        }

        # Articles the user is focused on (from question + authority hints) → also pull neighbours.
        focus_articles: set[str] = set(self._ARTICLE_REF_RE.findall(question or ""))
        for hint in (analysis.get("authority_hints") or []):
            if isinstance(hint, dict) and hint.get("article"):
                focus_articles.update(self._ARTICLE_REF_RE.findall(str(hint.get("article"))))

        # Articles cross-referenced inside the top retrieved provisions.
        referenced_articles: set[str] = set()
        top_source_ids: list[str] = []
        for chunk in chunks[:8]:
            sid = str(chunk.get("source_id") or "")
            if sid and sid not in top_source_ids:
                top_source_ids.append(sid)
            body = " ".join(
                str(chunk.get(k) or "")
                for k in ("content", "section", "section_ref")
            )
            referenced_articles.update(self._ARTICLE_REF_RE.findall(body))

        # Build the target set: focus articles + their neighbours + referenced articles.
        targets: set[str] = set()
        for art in focus_articles:
            targets.add(art)
            try:
                n = int(art)
                targets.add(str(n - 1)) if n > 1 else None
                targets.add(str(n + 1))
            except ValueError:
                continue
        targets.update(referenced_articles)

        # Drop articles already present to avoid wasted lookups.
        present_articles: set[str] = set()
        for chunk in chunks:
            for field in (chunk.get("section"), chunk.get("section_ref")):
                if field:
                    present_articles.update(self._ARTICLE_REF_RE.findall(str(field)))
        targets = {t for t in targets if t and t not in present_articles}
        if not targets:
            return []

        try:
            related = await self._retriever.fetch_related_articles(
                article_numbers=sorted(targets, key=lambda x: int(x) if x.isdigit() else 0)[:10],
                source_ids=top_source_ids or None,
                jurisdiction=jurisdiction,
                tenant_id=tenant_id,
                top_k=12,
            )
        except Exception as exc:
            log.debug("research.cross_reference.failed", error=str(exc))
            return []

        fresh = [
            row for row in related
            if str(row.get("chunk_id") or row.get("id")) not in existing_ids
        ]
        if fresh:
            log.info("research.cross_reference.expanded", added=len(fresh), targets=len(targets))
        return fresh

    async def _run_retrieval_plan(
        self,
        plan: list[RetrievalQuery],
        *,
        tenant_id: str | None,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        settings = get_settings()
        plan_slice = plan[:max(1, settings.rag_plan_max_queries)]
        can_embed = (
            settings._looks_configured_secret(settings.openai_api_key)
            and not self._embedding_unavailable
        )
        if not can_embed:
            reason = "embedding_provider_unavailable" if self._embedding_unavailable else "openai_api_key_not_configured"
            log.info("research.embedding.disabled_keyword_only", reason=reason)

        semaphore = asyncio.Semaphore(2)

        async def _limited_plan_query(item: RetrievalQuery) -> tuple[list[dict[str, Any]], dict[str, Any], int, bool]:
            async with semaphore:
                return await self._run_plan_query(
                    item,
                    tenant_id=tenant_id,
                    top_k=top_k,
                    can_embed=can_embed,
                )

        item_results = await asyncio.gather(
            *[_limited_plan_query(item) for item in plan_slice],
            return_exceptions=True,
        )

        all_chunks: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        total_tokens = 0
        for item, result in zip(plan_slice, item_results):
            if isinstance(result, Exception):
                log.warning("research.plan_query.failed", purpose=item.purpose, error=str(result))
                trace.append({
                    "purpose": item.purpose,
                    "jurisdiction": item.jurisdiction,
                    "results": 0,
                    "mode": "failed",
                    "error": str(result),
                })
                continue

            chunks, trace_entry, tokens, embedding_disabled = result
            all_chunks.extend(chunks)
            trace.append(trace_entry)
            total_tokens += tokens
            if embedding_disabled:
                can_embed = False
                self._embedding_unavailable = True

        deduped = self._dedupe_chunks(all_chunks)
        if self._has_sufficient_statutory_context(deduped, plan_slice[0].jurisdiction if plan_slice else None):
            coverage = self._planner.assess_coverage(deduped, plan_slice[0].jurisdiction if plan_slice else None)
            trace.append({
                "purpose": "early_stop_sufficient_statutory_context",
                "jurisdiction": plan_slice[0].jurisdiction if plan_slice else None,
                "results": len(deduped),
                "mode": "agentic_fast_path",
                "reason": coverage.reason or "sufficient_primary_context",
                **coverage.metrics,
            })

        return deduped, trace, total_tokens

    async def _run_plan_query(
        self,
        item: RetrievalQuery,
        *,
        tenant_id: str | None,
        top_k: int,
        can_embed: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], int, bool]:
        embedding_vector: list[float] | None = None
        embedding_model: str | None = None
        item_embedding_error: str | None = None
        embedding_disabled = False

        if can_embed:
            try:
                embedding_result = await self._embedder.embed(
                    item.query,
                    multilingual=needs_multilingual_embedding(item.query, item.jurisdiction),
                )
                embedding_vector = embedding_result.vector
                embedding_model = embedding_result.model
                tokens = embedding_result.tokens
            except ProviderNotConfiguredError as exc:
                item_embedding_error = str(exc)
                embedding_disabled = True
                tokens = 0
                log.warning("research.embedding.unavailable_keyword_only", error=str(exc))
            except Exception as exc:  # noqa: BLE001
                item_embedding_error = str(exc)
                embedding_disabled = True
                tokens = 0
                log.warning("research.embedding.failed_keyword_only", error=str(exc))
        else:
            tokens = 0

        chunks = await self._retriever.retrieve(
            query=item.query,
            embedding=embedding_vector,
            jurisdiction=item.jurisdiction,
            tenant_id=tenant_id,
            top_k=top_k,
        )
        trace_entry = {
            "purpose": item.purpose,
            "jurisdiction": item.jurisdiction,
            "results": len(chunks),
            "embedding_model": embedding_model,
            "mode": "hybrid" if embedding_vector else "keyword_only",
            "embedding_error": item_embedding_error,
        }
        return chunks, trace_entry, tokens, embedding_disabled

    def _extract_memory_highlights(self, memory: dict) -> dict:
        if memory.get("empty"):
            return {}
        return {
            "facts_summary": memory.get("facts_summary"),
            "key_citations": (memory.get("key_citations") or [])[:10],
            "past_strategies": memory.get("strategies") or [],
        }

    def _needs_multilingual_embedding(self, question: str, jurisdiction: str | None) -> bool:
        return needs_multilingual_embedding(question, jurisdiction)

    def _retrieval_source(self, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return "empty"
        return "database"

    def _has_sufficient_statutory_context(self, chunks: list[dict[str, Any]], jurisdiction: str | None) -> bool:
        if not chunks:
            return False
        coverage = self._planner.assess_coverage(self._dedupe_chunks(chunks), jurisdiction)
        return coverage.enough_results and coverage.has_statute and coverage.has_clean_text

    def _dedupe_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: dict[str, int] = {}
        unique: list[dict[str, Any]] = []
        for chunk in chunks:
            key = str(chunk.get("chunk_id") or chunk.get("id") or f"{chunk.get('title')}|{chunk.get('content', '')[:120]}")
            if key not in seen:
                seen[key] = len(unique)
                unique.append(chunk)
                continue

            current_index = seen[key]
            existing_score = self._score_for_dedupe(unique[current_index])
            new_score = self._score_for_dedupe(chunk)
            if new_score > existing_score:
                unique[current_index] = {**unique[current_index], **chunk}
        return unique

    def _score_for_dedupe(self, chunk: dict[str, Any]) -> float:
        for key in ("_rerank_score", "final_score", "score"):
            try:
                return float(chunk.get(key))
            except (TypeError, ValueError):
                continue
        return 0.0
