from __future__ import annotations

from api.knowledge import _assess_chunk_audit, _map_chunk_audit_item


def test_chunk_audit_flags_article_mapping_issues() -> None:
    rows = [
        {
            "id": "chunk-1",
            "chunk_index": 0,
            "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1",
            "article": "1",
            "status": "pending",
            "review_status": "pending_review",
            "metadata": {"article": "1", "chunk_text_quality": {"score": 0.92}},
        },
        {
            "id": "chunk-2",
            "chunk_index": 1,
            "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 4",
            "article": "5",
            "status": "pending",
            "review_status": "pending_review",
            "metadata": {"article": "5"},
        },
        {
            "id": "chunk-3",
            "chunk_index": 2,
            "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 3",
            "article": "3",
            "status": "pending",
            "review_status": "pending_review",
            "metadata": {"article": "3"},
        },
        {
            "id": "chunk-4",
            "chunk_index": 3,
            "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 3",
            "article": "3",
            "status": "pending",
            "review_status": "pending_review",
            "metadata": {"article": "3"},
        },
    ]

    items = [_map_chunk_audit_item(row, {}) for row in rows]
    qa = _assess_chunk_audit(items, source_table="laws", audit_limit=5000, document_structure=None)

    assert qa["correctness"]["status"] == "needs_review"
    assert qa["articleMismatchCount"] == 1
    assert qa["outOfOrderCount"] == 1
    assert qa["missingArticles"] == ["2"]
    assert qa["duplicateSections"] == ["\u0ea1\u0eb2\u0e94\u0e95\u0eb2 3"]


def test_chunk_audit_accepts_aligned_lao_article_sequence() -> None:
    rows = [
        {
            "id": "chunk-1",
            "chunk_index": 0,
            "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1",
            "article": "1",
            "status": "active",
            "review_status": "approved",
            "metadata": {"article": "1"},
        },
        {
            "id": "chunk-2",
            "chunk_index": 1,
            "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 2",
            "article": "2",
            "status": "active",
            "review_status": "approved",
            "metadata": {"article": "2"},
        },
    ]

    items = [_map_chunk_audit_item(row, {}) for row in rows]
    qa = _assess_chunk_audit(items, source_table="laws", audit_limit=5000, document_structure=None)

    assert qa["correctness"]["status"] == "ok"
    assert qa["articleCount"] == 2
    assert qa["articleMismatchCount"] == 0
    assert qa["missingArticleCount"] == 0
