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

    async def _execute(
        self,
        *,
        question: str,
        memory: dict,
        jurisdiction: str | None = None,
    ) -> dict[str, Any]:
        settings = get_settings()
        canonical_jurisdiction_value = canonical_jurisdiction(jurisdiction)

        chunks, retrieval_trace, embedding_tokens = await self._agentic_retrieve(
            question=question,
            jurisdiction=canonical_jurisdiction_value,
            top_k=settings.rag_top_k * 3,
        )

        # Step 3: Graph expansion from top case hits
        top_case_ids = [
            c["id"] for c in chunks[:5]
            if c.get("type") == "case" and c.get("id")
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
            "retrieval": {
                "source": self._retrieval_source(reranked),
                "count": len(reranked),
                "jurisdiction": canonical_jurisdiction_value,
                "trace": retrieval_trace,
            },
            "_confidence": min(1.0, len(reranked) / max(1, settings.rag_top_k)),
            "_tokens": embedding_tokens,
        }

    async def _agentic_retrieve(
        self,
        *,
        question: str,
        jurisdiction: str | None,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        plan = self._planner.plan(question, jurisdiction)
        chunks, trace, tokens = await self._run_retrieval_plan(plan, top_k=top_k)

        if self._planner.should_second_pass(chunks):
            second_pass = self._planner.second_pass(question, jurisdiction)
            more_chunks, more_trace, more_tokens = await self._run_retrieval_plan(second_pass, top_k=top_k)
            chunks.extend(more_chunks)
            trace.extend(more_trace)
            tokens += more_tokens

        return self._dedupe_chunks(chunks), trace, tokens

    async def _run_retrieval_plan(
        self,
        plan: list[RetrievalQuery],
        *,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        all_chunks: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        total_tokens = 0

        for item in plan[:5]:
            embedding_vector: list[float] | None = None
            embedding_model: str | None = None
            try:
                embedding_result = await self._embedder.embed(
                    item.query,
                    multilingual=needs_multilingual_embedding(item.query, item.jurisdiction),
                )
                embedding_vector = embedding_result.vector
                embedding_model = embedding_result.model
                total_tokens += embedding_result.tokens
            except ProviderNotConfiguredError as exc:
                log.warning("research.embedding.unavailable_keyword_only", error=str(exc))

            chunks = await self._retriever.retrieve(
                query=item.query,
                embedding=embedding_vector,
                jurisdiction=item.jurisdiction,
                top_k=top_k,
            )
            all_chunks.extend(chunks)
            trace.append({
                "purpose": item.purpose,
                "jurisdiction": item.jurisdiction,
                "results": len(chunks),
                "embedding_model": embedding_model,
                "mode": "hybrid" if embedding_vector else "keyword_only",
            })

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

    def _dedupe_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for chunk in chunks:
            key = str(chunk.get("chunk_id") or chunk.get("id") or f"{chunk.get('title')}|{chunk.get('content', '')[:120]}")
            if key not in seen:
                seen.add(key)
                unique.append(chunk)
        return unique
