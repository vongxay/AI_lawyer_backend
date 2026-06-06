from services.ingestion_service import (
    extract_text_with_metadata,
    _looks_like_garbled_pdf_text,
    assess_lao_legal_text_quality,
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
