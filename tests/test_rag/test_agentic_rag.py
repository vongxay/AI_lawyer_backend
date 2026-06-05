from __future__ import annotations

import pytest

from rag.agentic_planner import AgenticRetrievalPlanner
from rag.legal_query_analyzer import LegalQueryAnalyzer
from rag.reranker import Reranker
from rag.retriever import Retriever

LAND_RIGHT_PROTECTION_Q = (
    "\u0e9c\u0eb9\u0ec9\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a\u0eaa\u0eb4\u0e94"
    "\u0e99\u0eb3\u0ec3\u0e8a\u0ec9\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99 "
    "\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a\u0e81\u0eb2\u0e99\u0e9b\u0ebb\u0e81"
    "\u0e9b\u0ec9\u0ead\u0e87\u0eaa\u0eb4\u0e94\u0ec3\u0e94\u0ec1\u0e94\u0ec8"
)


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


def test_lao_land_use_right_protection_uses_general_land_authority_hints():
    analyzer = LegalQueryAnalyzer()
    planner = AgenticRetrievalPlanner()

    analysis = analyzer.analyze(LAND_RIGHT_PROTECTION_Q)
    plan = planner.plan(
        analysis.original_question,
        jurisdiction=analysis.jurisdiction,
        analysis=analysis.to_dict(),
    )

    assert analysis.practice_area == "land"
    assert analysis.issue_type == "rights"
    assert analysis.authority_hints[0].law_name == "Law on Land"
    assert analysis.authority_hints[0].article is None
    assert "land-use-right protection issue" in analysis.legal_issues
    assert plan[0].purpose == "authority_hint_1"
    assert plan[0].metadata["article"] is None


def test_lao_land_topic_hints_do_not_hardcode_specific_articles():
    analyzer = LegalQueryAnalyzer()
    questions = [
        "\u0e81\u0eb2\u0e99\u0e99\u0eb3\u0ec3\u0e8a\u0ec9\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99\u0e9a\u0ecd\u0ea5\u0eb4\u0ec0\u0ea7\u0e99\u0e99\u0ec9\u0eb3 \u0e95\u0ec9\u0ead\u0e87\u0eab\u0ec9\u0eb2\u0ea1\u0eaa\u0eb4\u0ec8\u0e87\u0ec3\u0e94\u0ec1\u0e94\u0ec8",
        "\u0e81\u0eb2\u0e99\u0e9b\u0ec8\u0ebd\u0e99\u0e9b\u0eb0\u0ec0\u0e9e\u0e94\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99 \u0e95\u0ec9\u0ead\u0e87\u0ead\u0eb0\u0e99\u0eb8\u0ea1\u0eb1\u0e94\u0ec1\u0e99\u0ea7\u0ec3\u0e94",
        "\u0e9e\u0ebb\u0e99\u0ea5\u0eb0\u0ec0\u0ea1\u0eb7\u0ead\u0e87\u0ea5\u0eb2\u0ea7\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99\u0e82\u0ead\u0e87\u0ea5\u0eb1\u0e94\u0ec4\u0e94\u0ec9\u0e88\u0eb1\u0e81\u0e9b\u0eb5",
        "\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99\u0ec3\u0e99\u0ea5\u0eb2\u0ea7\u0ec1\u0e9a\u0ec8\u0e87\u0ec0\u0e9b\u0eb1\u0e99\u0ec0\u0e82\u0e94\u0ec1\u0ea5\u0eb0\u0e9b\u0eb0\u0ec0\u0e9e\u0e94\u0ec3\u0e94\u0ec1\u0e94\u0ec8",
    ]

    for question in questions:
        analysis = analyzer.analyze(question)
        assert analysis.authority_hints[0].law_name == "Law on Land"
        assert analysis.authority_hints[0].article is None


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


def test_retriever_terms_do_not_hardcode_land_use_right_article():
    retriever = Retriever()

    terms = retriever._keyword_terms(LAND_RIGHT_PROTECTION_Q)

    assert "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 5" not in terms
    assert "Article 5" not in terms
    assert "\u0eaa\u0eb4\u0e94\u0e99\u0eb3\u0ec3\u0e8a\u0ec9" in terms


def test_retriever_extracts_article_targets_from_full_query_text():
    retriever = Retriever()

    terms = retriever._keyword_terms(
        "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99 \u0ea1\u0eb2\u0e94\u0e95\u0eb2 25 \u0e9a\u0ecd\u0ea5\u0eb4\u0ec0\u0ea7\u0e99\u0e99\u0ec9\u0eb3"
    )

    assert "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 25" in terms
    assert "Article 25" in terms


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


@pytest.mark.asyncio
async def test_reranker_boosts_any_explicit_article_target():
    reranker = Reranker()

    result = await reranker.rerank(
        query="What does Article 77 require?",
        chunks=[
            {
                "type": "law",
                "title": "Example Law",
                "section": "Article 12",
                "content": "Article 12. Other rule.",
                "final_score": 1.2,
            },
            {
                "type": "law",
                "title": "Example Law",
                "section": "Article 77",
                "content": "Article 77. Relevant rule.",
                "final_score": 0.2,
            },
        ],
        top_k=2,
    )

    assert result[0]["section"] == "Article 77"
