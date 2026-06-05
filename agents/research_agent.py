"""
agents/research_agent.py
=========================
Legal Research Agent — CORE agent, runs on every query.

Responsibilities:
1. Generate query embedding
2. Hybrid search (semantic + BM25) via Supabase pgvector + FTS
3. Case law graph expansion (precedent chains)
4. Cross-encoder reranking
5. Assemble structured legal context for IRAC agent

Output schema:
    retrieved_documents: list of ranked legal chunks
    case_graph_context:  related precedents from graph traversal
    memory_highlights:   relevant past case facts
"""
from __future__ import annotations

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

log = get_logger(__name__)

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
        self._retriever = Retriever(supabase=supabase)
        self._graph = GraphExpander(supabase=supabase)
        self._reranker = Reranker()
        self._planner = AgenticRetrievalPlanner()
        self._query_analyzer = LegalQueryAnalyzer()
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

        chunks, retrieval_trace, embedding_tokens, retrieval_coverage = await self._agentic_retrieve(
            question=question,
            jurisdiction=effective_jurisdiction,
            query_analysis=query_analysis,
            tenant_id=tenant_id,
            top_k=max(settings.rag_top_k, settings.rag_top_k * 2),
        )

        # Step 3: Graph expansion from top case hits
        top_case_ids = [
            str(c.get("source_id") or c.get("id")) for c in chunks[:5]
            if c.get("type") == "case" and (c.get("source_id") or c.get("id"))
        ]
        graph_results = []
        if top_case_ids:
            graph_results = await self._graph.expand(
                case_ids=top_case_ids,
                depth=settings.graph_depth,
            )

        # Step 4: Rerank combined results
        all_chunks = chunks + graph_results
        reranked = await self._reranker.rerank(
            query=question,
            chunks=all_chunks,
            top_k=settings.rag_top_k,
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
        tenant_id: str | None,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, Any]:
        analysis = query_analysis.to_dict()
        plan = self._planner.plan(question, jurisdiction, analysis=analysis)
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

    async def _run_retrieval_plan(
        self,
        plan: list[RetrievalQuery],
        *,
        tenant_id: str | None,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        all_chunks: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        total_tokens = 0
        settings = get_settings()
        can_embed = (
            settings._looks_configured_secret(settings.openai_api_key)
            and not self._embedding_unavailable
        )
        if not can_embed:
            reason = "embedding_provider_unavailable" if self._embedding_unavailable else "openai_api_key_not_configured"
            log.info("research.embedding.disabled_keyword_only", reason=reason)

        for item in plan[:max(1, settings.rag_plan_max_queries)]:
            embedding_vector: list[float] | None = None
            embedding_model: str | None = None
            item_embedding_error: str | None = None
            if can_embed:
                try:
                    embedding_result = await self._embedder.embed(
                        item.query,
                        multilingual=needs_multilingual_embedding(item.query, item.jurisdiction),
                    )
                    embedding_vector = embedding_result.vector
                    embedding_model = embedding_result.model
                    total_tokens += embedding_result.tokens
                except ProviderNotConfiguredError as exc:
                    item_embedding_error = str(exc)
                    can_embed = False
                    self._embedding_unavailable = True
                    log.warning("research.embedding.unavailable_keyword_only", error=str(exc))
                except Exception as exc:  # noqa: BLE001
                    item_embedding_error = str(exc)
                    can_embed = False
                    self._embedding_unavailable = True
                    log.warning("research.embedding.failed_keyword_only", error=str(exc))

            chunks = await self._retriever.retrieve(
                query=item.query,
                embedding=embedding_vector,
                jurisdiction=item.jurisdiction,
                tenant_id=tenant_id,
                top_k=top_k,
            )
            all_chunks.extend(chunks)
            trace.append({
                "purpose": item.purpose,
                "jurisdiction": item.jurisdiction,
                "results": len(chunks),
                "embedding_model": embedding_model,
                "mode": "hybrid" if embedding_vector else "keyword_only",
                "embedding_error": item_embedding_error,
            })

            if self._has_sufficient_statutory_context(all_chunks, item.jurisdiction):
                coverage = self._planner.assess_coverage(self._dedupe_chunks(all_chunks), item.jurisdiction)
                trace.append({
                    "purpose": "early_stop_sufficient_statutory_context",
                    "jurisdiction": item.jurisdiction,
                    "results": len(all_chunks),
                    "mode": "agentic_fast_path",
                    "reason": coverage.reason or "sufficient_primary_context",
                    **coverage.metrics,
                })
                break

        return all_chunks, trace, total_tokens

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
