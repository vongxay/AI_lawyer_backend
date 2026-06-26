"""
Lightweight case fact accumulation for follow-up legal conversations.

Facts are extracted from user questions and research briefs — not from model
hallucination — so they can safely inform later turns without becoming authority.
"""
from __future__ import annotations

from typing import Any


def merge_case_facts(
    existing_summary: str | None,
    *,
    question: str,
    material_facts: list[str] | None = None,
    legal_issues: list[str] | None = None,
    issue_primary: str | None = None,
    max_chars: int = 2400,
) -> str:
    segments: list[str] = []

    for value in (existing_summary or "").split(" | "):
        cleaned = value.strip()
        if cleaned and cleaned not in segments:
            segments.append(cleaned)

    if question.strip():
        question_fact = f"User stated: {question.strip()[:420]}"
        if question_fact not in segments:
            segments.append(question_fact)

    for fact in material_facts or []:
        cleaned = str(fact).strip()
        if not cleaned:
            continue
        entry = f"Fact: {cleaned[:280]}"
        if entry not in segments:
            segments.append(entry)

    for issue in legal_issues or []:
        cleaned = str(issue).strip()
        if not cleaned:
            continue
        entry = f"Issue: {cleaned[:220]}"
        if entry not in segments:
            segments.append(entry)

    if issue_primary and str(issue_primary).strip():
        entry = f"Primary issue: {str(issue_primary).strip()[:220]}"
        if entry not in segments:
            segments.append(entry)

    summary = " | ".join(segments)
    if len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 3].rstrip() + "..."
