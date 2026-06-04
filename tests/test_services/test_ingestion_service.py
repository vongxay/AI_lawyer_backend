from services.ingestion_service import _looks_like_garbled_pdf_text


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
