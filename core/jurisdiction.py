"""
Jurisdiction and language helpers for legal workflows.

The backend receives mixed Thai/Lao/English UI inputs, while the database schema
uses canonical jurisdiction values such as ``laos`` and ``thailand``.
"""
from __future__ import annotations


LAO_KEYWORDS = {
    "lao",
    "laos",
    "lao pdr",
    "ລາວ",
    "ສປປ",
    "ສປປ ລາວ",
    "กฎหมายลาว",
    "ประเทศลาว",
}

THAI_KEYWORDS = {
    "thai",
    "thailand",
    "ไทย",
    "กฎหมายไทย",
}


def contains_lao_script(text: str) -> bool:
    return any("\u0e80" <= ch <= "\u0eff" for ch in text)


def contains_thai_script(text: str) -> bool:
    return any("\u0e00" <= ch <= "\u0e7f" for ch in text)


def infer_jurisdiction(text: str, explicit: str | None = None) -> str | None:
    if explicit:
        return canonical_jurisdiction(explicit)

    lowered = text.casefold()
    if contains_lao_script(text) or any(keyword in lowered for keyword in LAO_KEYWORDS):
        return "laos"
    if any(keyword in lowered for keyword in THAI_KEYWORDS):
        return "thailand"
    return None


def canonical_jurisdiction(value: str | None) -> str | None:
    if not value:
        return None

    normalized = value.strip().casefold().replace("_", " ").replace("-", " ")
    mapping = {
        "la": "laos",
        "lao": "laos",
        "laos": "laos",
        "lao pdr": "laos",
        "ລາວ": "laos",
        "th": "thailand",
        "thai": "thailand",
        "thailand": "thailand",
        "ไทย": "thailand",
        "intl": "international",
        "international": "international",
        "asean": "asean",
    }
    return mapping.get(normalized, normalized)


def short_jurisdiction(value: str | None) -> str | None:
    canonical = canonical_jurisdiction(value)
    if canonical == "laos":
        return "LA"
    if canonical == "thailand":
        return "TH"
    if canonical == "international":
        return "INTL"
    return canonical.upper() if canonical else None


def needs_multilingual_embedding(text: str, jurisdiction: str | None = None) -> bool:
    canonical = canonical_jurisdiction(jurisdiction)
    if canonical in {"laos", "thailand"}:
        return True
    return contains_lao_script(text) or contains_thai_script(text)
