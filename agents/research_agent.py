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
from core.logging import get_logger
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

    async def _execute(
        self,
        *,
        question: str,
        memory: dict,
        jurisdiction: str | None = None,
    ) -> dict[str, Any]:
        settings = get_settings()

        # Step 1: Embed query
        embedding_result = await self._embedder.embed(
            question,
            multilingual=self._needs_multilingual_embedding(question, jurisdiction),
        )

        # Step 2: Hybrid search
        chunks = await self._retriever.retrieve(
            query=question,
            embedding=embedding_result.vector,
            jurisdiction=jurisdiction,
            top_k=settings.rag_top_k * 3,  # over-fetch before rerank
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
                "jurisdiction": jurisdiction,
                "embedding_model": embedding_result.model,
            },
            "_confidence": min(1.0, len(reranked) / max(1, settings.rag_top_k)),
            "_tokens": embedding_result.tokens,
        }

    def _extract_memory_highlights(self, memory: dict) -> dict:
        if memory.get("empty"):
            return {}
        return {
            "facts_summary": memory.get("facts_summary"),
            "key_citations": (memory.get("key_citations") or [])[:10],
            "past_strategies": memory.get("strategies") or [],
        }

    def _needs_multilingual_embedding(self, question: str, jurisdiction: str | None) -> bool:
        if jurisdiction and jurisdiction.upper() in {"TH", "LA"}:
            return True
        return any("\u0e00" <= ch <= "\u0e7f" or "\u0e80" <= ch <= "\u0eff" for ch in question)

    def _retrieval_source(self, chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return "empty"
        if all(str(chunk.get("id", "")).startswith("stub-") for chunk in chunks):
            return "stub"
        return "database"
