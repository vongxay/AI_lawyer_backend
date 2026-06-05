"""
Small text matching helpers for Lao legal retrieval.

The uploaded Lao PDFs often contain OCR variants such as missing tone marks or
visually similar consonants. These helpers keep retrieval deterministic while
making lexical matching less brittle.
"""
from __future__ import annotations

import re
import unicodedata

LAO_ARTICLE = "\u0ea1\u0eb2\u0e94\u0e95\u0eb2"
THAI_ARTICLE = "\u0e21\u0e32\u0e15\u0e23\u0e32"

_LAO_REMOVE_MARKS = {
    "\u0ec8",  # mai ek
    "\u0ec9",  # mai tho
    "\u0eca",  # mai ti
    "\u0ecb",  # mai catawa
    "\u0ecc",  # cancellation mark
    "\u0ecd",  # nikhahit; OCR often writes decomposed vowel forms
}

_OCR_REPLACEMENTS = (
    ("\u0eb3", "\u0eb2"),  # match OCR decomposed vowel forms after mark removal
    ("\u0e97\u0eb5\u0ea5\u0eb4\u0e99", "\u0e97\u0eb5\u0e94\u0eb4\u0e99"),
    ("\u0e97\u0eb5\u0e9c\u0eb4\u0e99", "\u0e97\u0eb5\u0e94\u0eb4\u0e99"),
    ("\u0e97\u0eb5\u0ea7\u0eb4\u0e99", "\u0e97\u0eb5\u0e94\u0eb4\u0e99"),
    ("\u0e97\u0eb5\u0e95\u0eb4\u0e99", "\u0e97\u0eb5\u0e94\u0eb4\u0e99"),
    ("\u0eaa\u0eb4\u0e94\u0e99\u0ecd\u0eb2\u0ec3\u0e8a", "\u0eaa\u0eb4\u0e94\u0e99\u0eb2\u0ec3\u0e8a"),
    ("\u0eaa\u0eb4\u0e94\u0e99\u0eb2\u0ecd\u0ec3\u0e8a", "\u0eaa\u0eb4\u0e94\u0e99\u0eb2\u0ec3\u0e8a"),
    ("\u0e9c\u0ebb\u0e99\u0e9b\u0eb0\u0ec2\u0e97\u0e8d\u0e94", "\u0e9c\u0ebb\u0e99\u0e9b\u0eb0\u0ec2\u0eab\u0e8d\u0e94"),
    ("\u0e9a\u0ebb\u0e81\u0e9b\u0eb1\u0e81", "\u0e9b\u0ebb\u0e81\u0e9b\u0eb1\u0e81"),
    ("\u0ec3\u0e82", "\u0ec3\u0e8a"),
    ("\u0ec3\u0e97", "\u0ec3\u0eab"),
)

LAO_LEGAL_PHRASES = (
    "\u0e9c\u0eb9\u0ec9\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a",
    "\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a",
    "\u0eaa\u0eb4\u0e94\u0e99\u0eb3\u0ec3\u0e8a\u0ec9",
    "\u0eaa\u0eb4\u0e94\u0e99\u0ecd\u0eb2\u0ec3\u0e8a\u0ec9",
    "\u0eaa\u0eb4\u0e94\u0e99\u0eb2\u0ecd\u0ec3\u0e8a\u0ec9",
    "\u0e9b\u0ebb\u0e81\u0e9b\u0ec9\u0ead\u0e87",
    "\u0e9b\u0ebb\u0e81\u0e9b\u0eb1\u0e81\u0eae\u0eb1\u0e81\u0eaa\u0eb2",
    "\u0e9c\u0ebb\u0e99\u0e9b\u0eb0\u0ec2\u0eab\u0e8d\u0e94",
    "\u0eaa\u0eb4\u0e94\u0ec3\u0e8a\u0ec9",
    "\u0eaa\u0eb4\u0e94\u0ec4\u0e94\u0ec9\u0eae\u0eb1\u0e9a",
    "\u0eaa\u0eb4\u0e94\u0ec2\u0ead\u0e99",
    "\u0eaa\u0eb4\u0e94\u0eaa\u0eb7\u0e9a\u0e97\u0ead\u0e94",
    "\u0e9e\u0ebb\u0e99\u0ea5\u0eb0\u0ec0\u0ea1\u0eb7\u0ead\u0e87",
    "\u0e9e\u0ebb\u0e99\u0ea5\u0eb0\u0ec0\u0ea1\u0eb7\u0ead\u0e87\u0ea5\u0eb2\u0ea7",
    "\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2",
    "\u0ec0\u0e8a\u0ebb\u0eb2",
    "\u0e82\u0ead\u0e87\u0ea5\u0eb1\u0e94",
    "\u0eaa\u0eb9\u0e87\u0eaa\u0eb8\u0e94",
    "\u0e88\u0eb1\u0e81\u0e9b\u0eb5",
    "\u0e9a\u0ecd\u0ec8\u0ec0\u0e81\u0eb5\u0e99",
    "\u0ec1\u0e9a\u0ec8\u0e87",
    "\u0e88\u0eb1\u0e94\u0ec1\u0e9a\u0ec8\u0e87",
    "\u0ec1\u0e9a\u0ec8\u0e87\u0ec0\u0e9b\u0eb1\u0e99",
    "\u0ec0\u0e82\u0e94",
    "\u0e9b\u0eb0\u0ec0\u0e9e\u0e94",
    "\u0ec0\u0e82\u0e94\u0ec1\u0ea5\u0eb0\u0e9b\u0eb0\u0ec0\u0e9e\u0e94",
    "\u0e81\u0eb2\u0e99\u0e99\u0eb3\u0ec3\u0e8a\u0ec9",
    "\u0e9a\u0ecd\u0ea5\u0eb4\u0ec0\u0ea7\u0e99\u0e99\u0ec9\u0eb3",
    "\u0e9a\u0ecd\u0ea5\u0eb4\u0ec0\u0ea7\u0e99\u0e99\u0ecd\u0ec9\u0eb2",
    "\u0eab\u0ec9\u0eb2\u0ea1",
    "\u0eaa\u0eb4\u0ec8\u0e87\u0ec3\u0e94",
    "\u0e81\u0eb2\u0e99\u0e9b\u0ec8\u0ebd\u0e99",
    "\u0e9b\u0ec8\u0ebd\u0e99\u0e9b\u0eb0\u0ec0\u0e9e\u0e94",
    "\u0ec0\u0e87\u0eb7\u0ec8\u0ead\u0e99\u0ec4\u0e82",
    "\u0ead\u0eb0\u0e99\u0eb8\u0ea1\u0eb1\u0e94",
)

LAO_LEGAL_EXPANSIONS = (
    (
        "\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2",
        ("\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2", "\u0ec0\u0e8a\u0ebb\u0eb2", "\u0e84\u0ec8\u0eb2\u0ec0\u0e8a\u0ebb\u0ec8\u0eb2", "lease", "rent", "tenant"),
    ),
    (
        "\u0e9b\u0ebb\u0e81\u0e9b\u0ec9\u0ead\u0e87",
        (
            "\u0e9b\u0ebb\u0e81\u0e9b\u0ec9\u0ead\u0e87",
            "\u0e9b\u0ebb\u0e81\u0e9b\u0eb1\u0e81\u0eae\u0eb1\u0e81\u0eaa\u0eb2",
            "\u0e9c\u0ebb\u0e99\u0e9b\u0eb0\u0ec2\u0eab\u0e8d\u0e94",
            "\u0eaa\u0eb4\u0e94\u0ec3\u0e8a\u0ec9",
            "\u0eaa\u0eb4\u0e94\u0ec2\u0ead\u0e99",
            "\u0eaa\u0eb4\u0e94\u0eaa\u0eb7\u0e9a\u0e97\u0ead\u0e94",
        ),
    ),
    (
        "\u0ec1\u0e9a\u0ec8\u0e87",
        ("\u0ec1\u0e9a\u0ec8\u0e87", "\u0e88\u0eb1\u0e94\u0ec1\u0e9a\u0ec8\u0e87", "\u0ec0\u0e82\u0e94", "\u0e9b\u0eb0\u0ec0\u0e9e\u0e94", "zone", "category"),
    ),
    (
        "\u0e9b\u0eb0\u0ec0\u0e9e\u0e94",
        ("\u0e9b\u0eb0\u0ec0\u0e9e\u0e94", "\u0ec0\u0e82\u0e94", "\u0e88\u0eb1\u0e94\u0ec1\u0e9a\u0ec8\u0e87", "category", "land type"),
    ),
    (
        "\u0e99\u0ec9\u0eb3",
        ("\u0e99\u0ec9\u0eb3", "\u0e99\u0ecd\u0ec9\u0eb2", "\u0e99\u0eb2\u0ecd", "\u0e99\u0eb2\u0ec9", "water", "water area"),
    ),
    (
        "\u0e9b\u0ec8\u0ebd\u0e99",
        ("\u0e9b\u0ec8\u0ebd\u0e99", "\u0e9b\u0ebd\u0e99", "\u0ead\u0eb0\u0e99\u0eb8\u0ea1\u0eb1\u0e94", "\u0ec0\u0e87\u0eb7\u0ec8\u0ead\u0e99\u0ec4\u0e82", "approval", "change"),
    ),
)


def normalise_search_text(text: str) -> str:
    value = unicodedata.normalize("NFC", str(text or "").casefold())
    value = "".join(ch for ch in value if ch not in _LAO_REMOVE_MARKS)
    for old, new in _OCR_REPLACEMENTS:
        value = value.replace(old, new)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_lao_legal_terms(query: str) -> list[str]:
    lowered = str(query or "").casefold()
    normalised = normalise_search_text(lowered)
    terms: list[str] = []

    for phrase in LAO_LEGAL_PHRASES:
        if phrase in lowered or normalise_search_text(phrase) in normalised:
            terms.append(phrase)

    for marker, expansions in LAO_LEGAL_EXPANSIONS:
        if marker in lowered or normalise_search_text(marker) in normalised:
            terms.extend(expansions)

    return unique_terms(terms)


def unique_terms(terms: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        value = str(term or "").strip()
        key = normalise_search_text(value)
        if value and key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def term_matches_text(term: str, text: str, *, normalised_text: str | None = None) -> bool:
    value = str(term or "").casefold().strip()
    if not value:
        return False
    haystack = str(text or "").casefold()
    if value.isascii():
        if re.search(rf"\b{re.escape(value)}\b", haystack):
            return True
    elif value in haystack:
        return True

    normalised_value = normalise_search_text(value)
    if len(normalised_value) < 2:
        return False
    folded = normalised_text if normalised_text is not None else normalise_search_text(haystack)
    if value.isascii():
        return bool(re.search(rf"\b{re.escape(normalised_value)}\b", folded))
    return normalised_value in folded


def table_of_contents_penalty(text: str) -> float:
    sample = str(text or "")[:2500]
    if not sample.strip():
        return 0.0
    dotted_leaders = len(re.findall(r"(?:\.|-|_){6,}", sample))
    article_mentions = len(
        re.findall(rf"(?:{LAO_ARTICLE}|{THAI_ARTICLE}|article|section)\s*\d{{1,4}}", sample, flags=re.IGNORECASE)
    )
    if dotted_leaders >= 5 and article_mentions >= 4:
        return 1.25
    if dotted_leaders >= 3:
        return 0.85
    if article_mentions >= 8:
        return 0.45
    return 0.0
