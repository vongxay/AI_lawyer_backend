from __future__ import annotations

import pytest

from rag.agentic_planner import AgenticRetrievalPlanner
from rag.legal_query_analyzer import LegalQueryAnalyzer
from rag.reranker import Reranker
from rag.retriever import Retriever


def test_lao_land_query_gets_authority_expansions():
    planner = AgenticRetrievalPlanner()

    plan = planner.plan("land ownership rights in Laos", jurisdiction="laos")

    purposes = {item.purpose for item in plan}
    assert "original_user_question" in purposes
    assert "lao_land_statutory_terms" in purposes
    assert "lao_land_english_authority_terms" in purposes
    assert any(item.required for item in plan)


def test_legal_query_analyzer_builds_research_brief():
    analyzer = LegalQueryAnalyzer()

    analysis = analyzer.analyze("Can I claim land ownership rights in Laos?", jurisdiction=None)

    assert analysis.jurisdiction == "laos"
    assert analysis.practice_area == "land"
    assert analysis.issue_type == "rights"
    assert analysis.authority_hints
    assert analysis.authority_hints[0].law_name == "Law on Land"
    assert "land ownership/use-right issue" in analysis.legal_issues


def test_planner_uses_query_analysis_authority_hints():
    analyzer = LegalQueryAnalyzer()
    planner = AgenticRetrievalPlanner()
    analysis = analyzer.analyze("Can I claim land ownership rights in Laos?")

    plan = planner.plan(
        analysis.original_question,
        jurisdiction=analysis.jurisdiction,
        analysis=analysis.to_dict(),
    )

    authority_queries = [item for item in plan if item.purpose.startswith("authority_hint")]
    assert authority_queries
    assert "Law on Land" in authority_queries[0].query


def test_coverage_flags_missing_official_lao_source():
    planner = AgenticRetrievalPlanner()

    coverage = planner.assess_coverage(
        [
            {
                "type": "law",
                "title": "Lao land law",
                "content": "Article 1 land ownership land use rights " * 8,
            }
            for _ in range(5)
        ],
        jurisdiction="laos",
    )

    assert coverage.has_statute is True
    assert coverage.has_official_source is False
    assert coverage.reason == "no_official_lao_source"
    assert coverage.should_second_pass is True


def test_retriever_treats_misclassified_law_chunk_as_law():
    retriever = Retriever()

    row = retriever._normalise_row(
        {
            "id": "chunk-1",
            "source_id": "doc-1",
            "source_table": "cases",
            "document_type": "case",
            "title": "Law on Land",
            "content": "Article 1 land use right",
            "metadata": {},
        }
    )

    assert row["type"] == "law"
    assert row["chunk_id"] == "chunk-1"
    assert row["source_id"] == "doc-1"


@pytest.mark.asyncio
async def test_reranker_prefers_official_statute_over_generic_chunk():
    reranker = Reranker()

    result = await reranker.rerank(
        query="land ownership article",
        chunks=[
            {
                "type": "doc",
                "title": "Generic note",
                "content": "land ownership article",
                "final_score": 0.5,
            },
            {
                "type": "law",
                "source_table": "laws",
                "jurisdiction": "laos",
                "source_url": "https://laoofficialgazette.gov.la/example",
                "title": "Law on Land",
                "section": "Article 1",
                "content": "land ownership article",
                "final_score": 0.5,
            },
        ],
        top_k=2,
    )

    assert result[0]["title"] == "Law on Land"
    assert "_rerank_score" in result[0]
