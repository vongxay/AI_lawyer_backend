"""
agents/verification_agent.py
=============================
Citation Verification Agent — CORE agent, runs on every query.

Verifies each citation produced by the IRAC Reasoning Agent against:
1. Supabase knowledge base (laws + cases tables) — authoritative
2. The retrieved legal context actually used for the answer — grounding check

This is a CLOSED-LOOP verifier: it never uses the model's outside knowledge to
"bless" a citation. A 20-year lawyer only cites what they can point to in the
source documents in front of them, so citations are graded only against the DB
and the retrieved excerpts.

Status matrix:
    VERIFIED    — found in DB and currently ACTIVE/INDEXED
    OUTDATED    — found in DB but AMENDED or REPEALED
    UNVERIFIED  — not in DB, but grounded in retrieved context (or nothing to check against)
    REJECTED    — not in DB and NOT present in retrieved context → likely hallucinated

Alert: if rejection rate > threshold → triggers admin notification
"""
from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from agents.base_agent import BaseAgent
from core.config import get_settings
from core.logging import get_logger
from rag.legal_text_matching import normalise_search_text

if TYPE_CHECKING:
    from supabase import AsyncClient  # pragma: no cover

log = get_logger(__name__)


class CitationVerificationAgent(BaseAgent):
    name = "verification"

    def __init__(self, supabase: "AsyncClient | None" = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._supabase = supabase

    async def _execute(
        self,
        *,
        citations: list[dict],
        retrieved_documents: list[dict] | None = None,
        model_override: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if not citations:
            return {"citations": [], "citations_verified": True, "rejection_rate": 0.0, "_confidence": 1.0}

        # Build a grounding index from the retrieved context that produced the answer.
        grounding_text, grounding_sections = self._build_grounding_index(retrieved_documents)
        has_context = bool(grounding_text or grounding_sections)

        # Step 1: DB lookup (parallel) — authoritative source of truth.
        db_results = await asyncio.gather(
            *[self._check_db(c) for c in citations], return_exceptions=True
        )

        # Step 2: Grade everything not confirmed in the DB against the retrieved context.
        verified: list[dict] = []
        for citation, db_result in zip(citations, db_results):
            if isinstance(db_result, Exception):
                log.warning("verification.db_check.failed", ref=citation.get("ref"), error=str(db_result))
                verified.append(self._grade_against_context(citation, grounding_text, grounding_sections, has_context))
            elif db_result is not None:
                verified.append(db_result)
            else:
                verified.append(self._grade_against_context(citation, grounding_text, grounding_sections, has_context))

        # Step 3: Compute rejection rate and alert if needed
        total = len(verified)
        rejected = sum(1 for c in verified if c.get("status") == "REJECTED")
        rejection_rate = rejected / total if total > 0 else 0.0

        settings = get_settings()
        if rejection_rate > settings.citation_rejection_alert_rate:
            log.warning(
                "citation.high_rejection_rate",
                rate=round(rejection_rate, 2),
                rejected=rejected,
                total=total,
            )

        all_verified = all(c.get("status") == "VERIFIED" for c in verified) if verified else True
        confidence = max(0.0, 1.0 - rejection_rate * 2)

        return {
            "citations": verified,
            "citations_verified": all_verified,
            "rejection_rate": rejection_rate,
            "_confidence": confidence,
        }

    async def _check_db(self, citation: dict) -> dict | None:
        """Check citation against Supabase knowledge base."""
        if not self._supabase:
            return None  # Will fall through to LLM check

        ref = citation.get("ref", "")
        if not ref:
            return {**citation, "status": "REJECTED", "reason": "Empty citation reference"}

        try:
            chunk = await self._find_document_chunk(citation)
            if chunk:
                status_value = str(chunk.get("status", "")).casefold()
                review_status = str(chunk.get("review_status", "")).casefold()
                status = "VERIFIED" if status_value == "active" and review_status == "approved" else "UNVERIFIED"
                metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
                source_url = chunk.get("source_url") or metadata.get("source_url")
                return {
                    **citation,
                    "status": status,
                    "db_match": chunk.get("title") or chunk.get("section_ref") or chunk.get("id"),
                    "source_links": [source_url] if source_url else citation.get("source_links", []),
                }

            law = await self._find_law(ref)
            if law:
                status_value = str(law.get("status", "")).upper()
                status = "VERIFIED" if status_value in {"ACTIVE", "INDEXED"} else "OUTDATED"
                return {
                    **citation,
                    "status": status,
                    "db_match": law.get("title"),
                    "year": law.get("year") or law.get("year_be"),
                    "source_links": [law["source_url"]] if law.get("source_url") else citation.get("source_links", []),
                }

            case = await self._find_case(ref)
            if case:
                return {
                    **citation,
                    "status": "VERIFIED",
                    "db_match": case.get("case_no") or case.get("title"),
                    "year": case.get("year") or case.get("year_be"),
                    "source_links": [case["source_url"]] if case.get("source_url") else citation.get("source_links", []),
                }

            return None  # Not found → fall through to LLM

        except Exception as exc:
            log.warning("verification.db_error", ref=ref, error=str(exc))
            return None

    async def _find_document_chunk(self, citation: dict) -> dict | None:
        chunk_id = str(citation.get("chunk_id") or "").strip()
        if not chunk_id or not self._supabase:
            return None

        selects = (
            "id, title, status, review_status, source_url, metadata, section_ref",
            "id, title, status, review_status, section_ref",
        )
        for select in selects:
            try:
                result = await (
                    self._supabase.table("document_chunks")
                    .select(select)
                    .eq("id", chunk_id)
                    .limit(1)
                    .execute()
                )
                if result.data:
                    return result.data[0]
            except Exception:
                continue
        return None

    async def _find_law(self, ref: str) -> dict | None:
        terms = self._search_terms(ref)
        selects = (
            "id, title, status, year_be, section_number, source_url",
            "id, title, status",
        )
        filters = [(column, term) for term in terms for column in ("title", "full_text")]
        section = self._section_number(ref)
        if section:
            filters.append(("section_number", section))
        return await self._first_match("laws", selects, filters)

    async def _find_case(self, ref: str) -> dict | None:
        terms = self._search_terms(ref)
        selects = (
            "id, case_no, court, year_be, source_url",
            "id, case_no, court",
        )
        filters = [(column, term) for term in terms for column in ("case_no", "summary", "ruling")]
        return await self._first_match("cases", selects, filters)

    async def _first_match(
        self,
        table: str,
        selects: tuple[str, ...],
        filters: list[tuple[str, str]],
    ) -> dict | None:
        for select in selects:
            for column, term in filters[:8]:
                try:
                    result = await (
                        self._supabase.table(table)
                        .select(select)
                        .ilike(column, f"%{term[:80]}%")
                        .limit(1)
                        .execute()
                    )
                    if result.data:
                        return result.data[0]
                except Exception:
                    continue
        return None

    def _search_terms(self, ref: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", ref).strip()
        terms = [cleaned]
        section = self._section_number(ref)
        if section:
            terms.append(section)
        return [term for term in terms if term]

    def _section_number(self, ref: str) -> str | None:
        match = re.search(
            r"(?:\u0ea1\u0eb2\u0e94\u0e95\u0eb2|\u0e21\u0e32\u0e15\u0e23\u0e32|article|art\.?|section|sec\.?)\s*([0-9A-Za-z/.-]+)",
            ref,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        match = re.search(r"(?:มาตรา|section|sec\.?)\s*([0-9A-Za-z/.-]+)", ref, flags=re.IGNORECASE)
        return match.group(1) if match else None

    def _build_grounding_index(
        self, retrieved_documents: list[dict] | None
    ) -> tuple[str, set[str]]:
        """Normalised text blob + set of article numbers present in retrieved context."""
        if not retrieved_documents:
            return "", set()

        text_parts: list[str] = []
        sections: set[str] = set()
        for chunk in retrieved_documents:
            if not isinstance(chunk, dict):
                continue
            for key in ("content", "title", "section", "section_ref"):
                value = chunk.get(key)
                if value:
                    text_parts.append(str(value))
            for field in (chunk.get("section"), chunk.get("section_ref")):
                if field:
                    sections.update(self._article_numbers(str(field)))

        blob = normalise_search_text(" \n ".join(text_parts)) if text_parts else ""
        sections.update(self._article_numbers(blob))
        return blob, sections

    def _grade_against_context(
        self,
        citation: dict,
        grounding_text: str,
        grounding_sections: set[str],
        has_context: bool,
    ) -> dict:
        """Grade a citation not found in the DB strictly against retrieved context."""
        ref = str(citation.get("ref") or "").strip()
        if not ref:
            return {**citation, "status": "REJECTED", "reason": "Empty citation reference"}

        # With no retrieved context to check against, we cannot reject responsibly.
        if not has_context:
            return {**citation, "status": "UNVERIFIED", "note": "Not found in DB; no retrieved context to verify against"}

        section = self._section_number(ref)
        normalized_ref = normalise_search_text(ref)

        section_grounded = bool(section and section in grounding_sections)
        # Fall back to a phrase match (law name / key terms) when no article number is cited.
        phrase_grounded = False
        if not section:
            tokens = [t for t in normalized_ref.split() if len(t) >= 3]
            if tokens:
                phrase_grounded = sum(1 for t in tokens if t in grounding_text) >= max(1, len(tokens) // 2)

        if section_grounded or phrase_grounded:
            return {
                **citation,
                "status": "UNVERIFIED",
                "note": "Grounded in retrieved legal source but not confirmed in knowledge base",
            }
        return {
            **citation,
            "status": "REJECTED",
            "reason": "Citation not found in the retrieved legal sources (possible hallucination)",
        }

    @staticmethod
    def _article_numbers(text: str) -> set[str]:
        return set(re.findall(r"\b(\d{1,4}(?:/\d{1,3})?)\b", text or ""))
