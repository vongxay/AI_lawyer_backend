import services.ingestion_service as ingestion_service
from services.ingestion_service import (
    _build_extraction_candidate,
    _choose_best_candidate,
    _chunk_article_metadata,
    _document_article_metadata,
    _is_official_lao_source,
    _looks_like_garbled_pdf_text,
    _resolve_tesseract_languages,
    _source_authority,
    assess_lao_legal_text_quality,
    assess_legal_structure,
    chunk_legal_text,
    extract_text_with_metadata,
    normalise_lao_legal_text,
)


def test_detects_legacy_lao_pdf_mojibake() -> None:
    garbled = (
        "\u00ae\u00ea\u2020\u00a9\u00f2\u2212; 6. "
        "\u00af\u00bd-\u00aa\u00f2-\u00ae\u00f1\u00a9-\u00b2\u00f1\u2212-\u00ea\u00bd-"
        "\u00a1\u00c8\u00bc\u00b8-\u00a1\u00f1\u00ae-\u00ea\u2020-\u00a9\u00f2\u2212- "
        "\u00c0\u00a7\u201e\u2212: \u00b2\u00be-\u00a6\u00f3-\u00ea\u2020-\u00a9\u00f2\u2212, "
        "\u00ba\u00be\u00a1\u00ba\u2212- \u00a5\u00be\u00a1-\u00a1\u00be\u2212\u00b4\u00ba\u00ae-\u00c2\u00ba\u2212 "
    ) * 8

    assert _looks_like_garbled_pdf_text(garbled) is True


def test_does_not_flag_real_lao_or_english_text() -> None:
    lao = (
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 61 \u0eaa\u0eb0\u0e9e\u0eb2\u0e9a\u0e84\u0ea7\u0eb2\u0ea1"
        "\u0ec0\u0e9b\u0eb1\u0e99\u0e88\u0eb4\u0e87\u0e95\u0ec9\u0ead\u0e87\u0e8d\u0ead\u0ea1"
        "\u0eae\u0eb1\u0e9a \u0e9a\u0eb8\u0e81\u0e84\u0ebb\u0e99 \u0eab\u0ebc\u0eb7 "
        "\u0e81\u0eb2\u0e99\u0e88\u0eb1\u0e94\u0e95\u0eb1\u0ec9\u0e87 "
    ) * 8
    english = (
        "Article 61. A person or organization may request access through neighboring land "
        "when there is no practical passage to a public road. "
    ) * 8

    assert _looks_like_garbled_pdf_text(lao) is False
    assert _looks_like_garbled_pdf_text(english) is False


def test_flags_thai_script_contamination_in_lao_text() -> None:
    lao = (
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 61 "
        "\u0eaa\u0eb4\u0e94 \u0ec1\u0ea5\u0eb0 \u0e9e\u0eb1\u0e99\u0e97\u0eb0 "
        "\u0e82\u0ead\u0e87\u0e9c\u0eb9\u0ec9\u0e99\u0eb3\u0ec3\u0e8a\u0ec9\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99 "
    ) * 8
    thai_ocr_noise = (
        "\u0e01\u0e32\u0e23\u0e01\u0e33\u0e2b\u0e19\u0e14"
        "\u0e2a\u0e34\u0e17\u0e18\u0e34\u0e41\u0e25\u0e30\u0e2b\u0e19\u0e49\u0e32\u0e17\u0e35\u0e48 "
    ) * 8

    clean_quality = assess_lao_legal_text_quality(lao)
    mixed_quality = assess_lao_legal_text_quality(f"{lao} {thai_ocr_noise}")

    assert mixed_quality.language == "lo"
    assert mixed_quality.thai_ratio > 0.02
    assert mixed_quality.score < clean_quality.score
    assert any("Thai-script" in warning for warning in mixed_quality.warnings)


def test_flags_pdf_text_layer_with_thai_contamination_as_garbled() -> None:
    mixed = (
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 61 "
        "\u0eaa\u0eb4\u0e94 \u0ec1\u0ea5\u0eb0 \u0e9e\u0eb1\u0e99\u0e97\u0eb0 "
        "\u0e82\u0ead\u0e87\u0e9c\u0eb9\u0ec9\u0e99\u0eb3\u0ec3\u0e8a\u0ec9\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99 "
        "\u0e01\u0e32\u0e23\u0e01\u0e33\u0e2b\u0e19\u0e14"
        "\u0e2a\u0e34\u0e17\u0e18\u0e34\u0e41\u0e25\u0e30\u0e2b\u0e19\u0e49\u0e32\u0e17\u0e35\u0e48 "
    ) * 10

    assert _looks_like_garbled_pdf_text(mixed) is True


def test_candidate_selection_penalizes_lao_text_with_thai_ocr_contamination() -> None:
    clean = (
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 61 "
        "\u0eaa\u0eb4\u0e94 \u0ec1\u0ea5\u0eb0 \u0e9e\u0eb1\u0e99\u0e97\u0eb0 "
        "\u0e82\u0ead\u0e87\u0e9c\u0eb9\u0ec9\u0e99\u0eb3\u0ec3\u0e8a\u0ec9\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99 "
    ) * 12
    contaminated = clean + (
        "\u0e01\u0e32\u0e23\u0e01\u0e33\u0e2b\u0e19\u0e14"
        "\u0e2a\u0e34\u0e17\u0e18\u0e34\u0e41\u0e25\u0e30\u0e2b\u0e19\u0e49\u0e32\u0e17\u0e35\u0e48 "
    ) * 8

    text_layer = _build_extraction_candidate("pymupdf_text_layer", clean, source="pdf")
    ocr = _build_extraction_candidate("tesseract_ocr", contaminated, source="pdf")

    assert text_layer is not None
    assert ocr is not None
    assert _choose_best_candidate([ocr, text_layer]) is text_layer


def test_detects_lao_ocr_noise_without_mojibake() -> None:
    noisy = (
        "Rowe a \u0e9e\u0eb2\u0e81\u0e97 a "
        "\u0e9a\u0ebb\u0e94\u0e9a\u0eb1\u0e99\u0ead\u0eb1\u0e94\u0e97\u0ebb\u0ea7\u0ec4\u0e9b "
        "........................................... 5 "
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1 . "
        "\u0e88\u0eb8\u0e94\u0e9b\u0eb0\u0eaa\u0ebb\u0e87 "
        "\u0e82\u0ead\u0e87\u0e81\u0ebb\u0e94\u0e9a\u0ea1\u0eb2\u0e8d"
        " tea HOV ao guoon was Ill "
    ) * 8

    quality = assess_lao_legal_text_quality(noisy)

    assert quality.language == "lo"
    assert quality.score < 0.78
    assert _looks_like_garbled_pdf_text(noisy) is True


def test_normalises_pdf_artifact_lines() -> None:
    raw = "\n".join([
        "Rowe ao guoon",
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1 ........................................ 5",
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1. \u0e88\u0eb8\u0e94\u0e9b\u0eb0\u0eaa\u0ebb\u0e87",
    ])

    cleaned = normalise_lao_legal_text(raw, source="pdf")

    assert "Rowe" not in cleaned
    assert "................................" not in cleaned
    assert "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1. \u0e88\u0eb8\u0e94\u0e9b\u0eb0\u0eaa\u0ebb\u0e87" in cleaned


def test_chunking_preserves_lao_legal_sections_and_paragraphs() -> None:
    article_1 = "\n".join([
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1",
        "\u0e88\u0eb8\u0e94\u0e9b\u0eb0\u0eaa\u0ebb\u0e87",
        "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d\u0e99\u0eb5\u0ec9\u0e81\u0eb3\u0e99\u0ebb\u0e94"
        "\u0eab\u0ebc\u0eb1\u0e81\u0e81\u0eb2\u0e99 \u0ec1\u0ea5\u0eb0 "
        "\u0ea7\u0eb4\u0e97\u0eb5\u0e81\u0eb2\u0e99\u0e9b\u0eb0\u0e95\u0eb4\u0e9a\u0eb1\u0e94.",
    ])
    article_2 = "\n".join([
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 2",
        "\u0e82\u0ead\u0e9a\u0ec0\u0e82\u0e94\u0e81\u0eb2\u0e99\u0e9a\u0eb1\u0e87\u0e84\u0eb1\u0e9a\u0ec3\u0e8a\u0ec9",
    ])

    chunks = chunk_legal_text(f"{article_1}\n\n{article_2}", max_chars=320, overlap=60)

    assert chunks[0].section_ref == "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1"
    assert chunks[0].content.startswith("\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1\n\n")
    assert "\n\n" in chunks[0].content
    assert any(chunk.section_ref == "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 2" for chunk in chunks)


def test_chunking_keeps_heading_with_long_lao_article_body() -> None:
    heading = "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 7"
    body = (
        "\u0e9a\u0eb8\u0e81\u0e84\u0ebb\u0e99 \u0ec1\u0ea5\u0eb0 "
        "\u0e81\u0eb2\u0e99\u0e88\u0eb1\u0e94\u0e95\u0eb1\u0ec9\u0e87 "
        "\u0e95\u0ec9\u0ead\u0e87\u0e9b\u0eb0\u0e95\u0eb4\u0e9a\u0eb1\u0e94"
    ) * 20

    chunks = chunk_legal_text(f"{heading}\n{body}", max_chars=240, overlap=40)

    assert chunks[0].content.startswith(f"{heading}\n\n")
    assert chunks[0].content != heading
    assert all(chunk.content != heading for chunk in chunks)


def test_chunking_repairs_split_lao_article_numbers_from_ocr() -> None:
    text = "\n".join([
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1 14 \u0e81\u0eb2\u0e99\u0e9b\u0ead\u0e87\u0eae\u0ec9\u0eb2\u0e8d",
        "\u0e9a\u0eb8\u0e81\u0e84\u0ebb\u0e99\u0ec3\u0e94 "
        "\u0e97\u0eb5\u0ec8\u0e81\u0eb0\u0e97\u0eb3\u0e9c\u0eb4\u0e94.",
    ])

    chunks = chunk_legal_text(text)

    assert chunks[0].section_ref == "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 114"
    assert chunks[0].content.startswith("\u0ea1\u0eb2\u0e94\u0e95\u0eb2 114")


def test_chunk_article_metadata_prefers_each_chunk_section_over_document_article() -> None:
    chunks = chunk_legal_text("\n\n".join([
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1\n\u0e82\u0ecd\u0ec9\u0e84\u0ea7\u0eb2\u0ea1",
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 2\n\u0e82\u0ecd\u0ec9\u0e84\u0ea7\u0eb2\u0ea1",
    ]))

    first_article, first_source = _chunk_article_metadata(chunks[0], document_article="67", total_chunks=len(chunks))
    second_article, second_source = _chunk_article_metadata(chunks[1], document_article="67", total_chunks=len(chunks))

    assert (first_article, first_source) == ("1", "section_ref")
    assert (second_article, second_source) == ("2", "section_ref")


def test_document_article_metadata_omits_inferred_article_for_multi_article_code() -> None:
    chunks = chunk_legal_text("\n\n".join([
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1\n\u0e82\u0ecd\u0ec9\u0e84\u0ea7\u0eb2\u0ea1",
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 2\n\u0e82\u0ecd\u0ec9\u0e84\u0ea7\u0eb2\u0ea1",
    ]))

    article = _document_article_metadata(explicit_article=None, inferred_article="67", chunks=chunks)

    assert article is None


def test_legal_structure_flags_missing_duplicate_and_out_of_order_articles() -> None:
    chunks = chunk_legal_text("\n\n".join([
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1\n\u0e82\u0ecd\u0ec9\u0e84\u0ea7\u0eb2\u0ea1",
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 3\n\u0e82\u0ecd\u0ec9\u0e84\u0ea7\u0eb2\u0ea1",
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 2\n\u0e82\u0ecd\u0ec9\u0e84\u0ea7\u0eb2\u0ea1",
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 2\n\u0e82\u0ecd\u0ec9\u0e84\u0ea7\u0eb2\u0ea1\u0e8a\u0ecd\u0ec9\u0eb2",
    ]))

    report = assess_legal_structure(chunks)

    assert report.article_count == 3
    assert report.max_article_number == 3
    assert report.out_of_order_sections == 1
    assert "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 2" in report.duplicate_sections
    assert any("out of order" in warning for warning in report.warnings)


def test_quality_prefers_clean_lao_legal_text() -> None:
    clean = (
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 61 "
        "\u0eaa\u0eb4\u0e94 \u0ec1\u0ea5\u0eb0 \u0e9e\u0eb1\u0e99\u0e97\u0eb0 "
        "\u0e82\u0ead\u0e87\u0e9c\u0eb9\u0ec9\u0e99\u0eb3\u0ec3\u0e8a\u0ec9\u0e97\u0eb5\u0ec8\u0e94\u0eb4\u0e99 "
    ) * 12
    noisy = clean + (" Rowe tea HOV ao guoon was Ill +++ --- ................ " * 10)

    assert assess_lao_legal_text_quality(clean).score > assess_lao_legal_text_quality(noisy).score


def test_extracts_utf16_lao_text_without_replacement_noise() -> None:
    text = (
        "\u0ea1\u0eb2\u0e94\u0e95\u0eb2 1 "
        "\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d\u0e99\u0eb5\u0ec9\u0e81\u0eb3\u0e99\u0ebb\u0e94"
    ) * 20

    extracted = extract_text_with_metadata(
        text.encode("utf-16"),
        "text/plain",
        "lao-law.txt",
    )

    assert "\ufffd" not in extracted.text
    assert extracted.quality.language == "lo"
    assert extracted.quality.score >= 0.78


def test_resolves_lao_ocr_languages_without_thai_for_lao_jurisdiction(monkeypatch) -> None:
    class FakeSettings:
        pdf_ocr_languages = "lao+tha+eng"

    class FakeTesseract:
        @staticmethod
        def get_languages(config: str = "") -> list[str]:  # noqa: ARG004
            return ["lao", "tha", "eng"]

    monkeypatch.setattr(ingestion_service, "get_settings", lambda: FakeSettings())

    errors: list[str] = []
    resolved = _resolve_tesseract_languages(FakeTesseract(), errors=errors, jurisdiction="laos")

    assert resolved == "lao+eng"
    assert any("skipped: tha" in error for error in errors)


def test_refuses_lao_ocr_when_lao_traineddata_is_missing(monkeypatch) -> None:
    class FakeSettings:
        pdf_ocr_languages = "lao+tha+eng"

    class FakeTesseract:
        @staticmethod
        def get_languages(config: str = "") -> list[str]:  # noqa: ARG004
            return ["tha", "eng"]

    monkeypatch.setattr(ingestion_service, "get_settings", lambda: FakeSettings())

    errors: list[str] = []
    resolved = _resolve_tesseract_languages(FakeTesseract(), errors=errors, jurisdiction="laos")

    assert resolved == ""
    assert any("Lao OCR language data is required" in error for error in errors)


def test_treats_lao_national_assembly_as_official_source() -> None:
    source_url = "https://na.gov.la/wp-content/uploads/2026/01/law.pdf"

    assert _is_official_lao_source(source_url) is True
    assert _source_authority(source_url) == "lao_national_assembly"
