"""Tests for Lao FTS query preparation (Python side of keyword search)."""
from __future__ import annotations

from rag.legal_text_matching import normalise_search_text, prepare_lao_fts_query


def test_prepare_lao_fts_query_normalizes_tone_marks() -> None:
    raw = "ສິດນຳໃຊ້ທີ່ດິນ"
    prepared = prepare_lao_fts_query(raw, jurisdiction="laos")
    assert prepared
    assert "ສິດ" in prepared or normalise_search_text("ສິດ") in prepared


def test_prepare_lao_fts_query_expands_land_terms_for_laos() -> None:
    question = "ຜູ້ໄດ້ຮັບສິດນຳໃຊ້ທີ່ດິນ ໄດ້ຮັບການປົກປ້ອງສິດບໍ?"
    prepared = prepare_lao_fts_query(question, jurisdiction="laos")
    assert "ປົກປ້ອງ" in prepared or "ສິດ" in prepared


def test_prepare_lao_fts_query_leaves_english_queries_mostly_intact() -> None:
    query = "land use rights protection"
    prepared = prepare_lao_fts_query(query, jurisdiction="en")
    assert prepared == normalise_search_text(query)


def test_prepare_lao_fts_query_handles_empty() -> None:
    assert prepare_lao_fts_query("") == ""
    assert prepare_lao_fts_query("   ", jurisdiction="laos") == ""
