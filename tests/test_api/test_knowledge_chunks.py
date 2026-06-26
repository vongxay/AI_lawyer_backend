from __future__ import annotations

from api.knowledge import _assess_chunk_audit, _map_article_section_groups, _map_chunk_audit_item
from services.ingestion_service import chunk_legal_text


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


def test_chunk_audit_allows_front_matter_article_sequence_restart() -> None:
    rows = [
        {
            "id": "front-1",
            "chunk_index": 0,
            "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1",
            "article": "1",
            "status": "pending",
            "review_status": "pending_review",
            "metadata": {"article": "1"},
        },
        {
            "id": "front-2",
            "chunk_index": 1,
            "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 2",
            "article": "2",
            "status": "pending",
            "review_status": "pending_review",
            "metadata": {"article": "2"},
        },
        {
            "id": "code-1",
            "chunk_index": 2,
            "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1",
            "article": "1",
            "status": "pending",
            "review_status": "pending_review",
            "metadata": {"article": "1"},
        },
        {
            "id": "code-2",
            "chunk_index": 3,
            "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 2",
            "article": "2",
            "status": "pending",
            "review_status": "pending_review",
            "metadata": {"article": "2"},
        },
    ]

    items = [_map_chunk_audit_item(row, {}) for row in rows]
    qa = _assess_chunk_audit(items, source_table="laws", audit_limit=5000, document_structure=None)

    assert qa["correctness"]["status"] == "ok"
    assert qa["duplicateSectionCount"] == 0
    assert qa["outOfOrderCount"] == 0


def test_article_section_groups_reconstruct_long_continued_article() -> None:
    heading = "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 9"
    paragraphs = [
        f"\u0e82\u0ecd\u0ec9 marker-{index:03d} "
        + ("\u0e9a\u0eb8\u0e81\u0e84\u0ebb\u0e99 \u0ec1\u0ea5\u0eb0 \u0e81\u0eb2\u0e99\u0e88\u0eb1\u0e94\u0e95\u0eb1\u0ec9\u0e87 " * 8)
        for index in range(10)
    ]
    chunks = chunk_legal_text("\n\n".join([heading, *paragraphs]), max_chars=520, overlap=100)
    rows = [
        {
            "id": f"chunk-{chunk.index}",
            "chunk_index": chunk.index,
            "section_ref": chunk.section_ref,
            "article": "9",
            "content": chunk.content,
            "metadata": {
                "article": "9",
                "section_ref_base": heading,
                "section_part_index": index + 1,
                "section_total_parts": len(chunks),
            },
        }
        for index, chunk in enumerate(chunks)
    ]

    groups = _map_article_section_groups(rows, include_content=True, max_content_chars=10000)

    assert len(groups) == 1
    assert groups[0]["chunkCount"] == len(chunks)
    assert groups[0]["isComplete"] is True
    assert groups[0]["content"].count(heading) == 1
    for index in range(10):
        assert f"marker-{index:03d}" in groups[0]["content"]


def test_article_section_groups_keep_restarted_article_numbers_separate() -> None:
    rows = [
        {
            "id": "front-1",
            "chunk_index": 0,
            "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1",
            "article": "1",
            "content": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1\n\nfront matter",
            "metadata": {"section_ref_base": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1", "section_part_index": 1},
        },
        {
            "id": "front-2",
            "chunk_index": 1,
            "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 2",
            "article": "2",
            "content": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 2\n\nfront matter",
            "metadata": {"section_ref_base": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 2", "section_part_index": 1},
        },
        {
            "id": "code-1",
            "chunk_index": 2,
            "section_ref": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1",
            "article": "1",
            "content": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1\n\ncode article",
            "metadata": {"section_ref_base": "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1", "section_part_index": 1},
        },
    ]

    groups = _map_article_section_groups(rows, include_content=True, max_content_chars=10000)

    article_1_groups = [group for group in groups if group["sectionRefBase"] == "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1"]
    assert len(article_1_groups) == 2
    assert article_1_groups[0]["content"] != article_1_groups[1]["content"]
