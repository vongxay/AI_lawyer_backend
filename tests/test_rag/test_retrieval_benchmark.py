from __future__ import annotations

from evaluation.retrieval_benchmark import RetrievalBenchmarkCase, evaluate_retrieval_case


def test_retrieval_benchmark_detects_expected_article_rank():
    case = RetrievalBenchmarkCase(
        id="article_5",
        question="land use right protection",
        expected_articles=["5"],
        must_contain_any=["\u0eaa\u0eb4\u0e94\u0ec2\u0ead\u0e99"],
    )

    result = evaluate_retrieval_case(
        case,
        [
            {
                "title": "Law on Land",
                "section": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 12.",
                "content": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 12. unrelated",
            },
            {
                "title": "Law on Land",
                "section": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 5.",
                "content": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 5. \u0eaa\u0eb4\u0e94\u0ec2\u0ead\u0e99",
            },
        ],
    )

    assert result.article_rank == 2
    assert result.hit_at_1 is False
    assert result.hit_at_3 is True
    assert result.reciprocal_rank == 0.5
    assert result.term_coverage == 1.0
    assert result.passed is True


def test_retrieval_benchmark_fails_when_expected_article_missing():
    case = RetrievalBenchmarkCase(
        id="article_25",
        question="water-area land use",
        expected_articles=["25"],
    )

    result = evaluate_retrieval_case(
        case,
        [
            {
                "title": "Law on Land",
                "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 14.",
                "content": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 14. land type change",
            },
        ],
    )

    assert result.article_rank is None
    assert result.hit_at_5 is False
    assert result.passed is False
