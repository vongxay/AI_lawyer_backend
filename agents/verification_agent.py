"""
agents/verification_agent.py
=============================
Citation Verification Agent — CORE agent, runs on every query.

Verifies each citation produced by the IRAC Reasoning Agent against:
1. Supabase knowledge base (laws + cases tables)
2. Optional LLM cross-check for plausibility

Status matrix:
    VERIFIED    — found in DB and currently ACTIVE
    OUTDATED    — found in DB but AMENDED or REPEALED
    UNVERIFIED  — not found in DB, LLM considers plausible
    REJECTED    — hallucinated or clearly wrong → removed from response

Alert: if rejection rate > threshold → triggers admin notification
"""
from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from agents.base_agent import BaseAgent
from core.config import get_settings
from core.logging import get_logger

if TYPE_CHECKING:
    from supabase import AsyncClient  # pragma: no cover

log = get_logger(__name__)

_VERIFY_SYSTEM_PROMPT = """
You are a legal citation verifier. Your ONLY job is to assess whether citations are plausible.
For each citation, respond with JSON array:
[
  {
    "ref": "original citation string",
    "plausible": true|false,
    "reason": "brief reason"
  }
]
Do not add any text before or after the JSON array.
"""


class CitationVerificationAgent(BaseAgent):
    name = "verification"

    def __init__(self, supabase: "AsyncClient | None" = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._supabase = supabase

    async def _execute(self, *, citations: list[dict], **kwargs) -> dict[str, Any]:
        if not citations:
            return {"citations": [], "citations_verified": True, "rejection_rate": 0.0, "_confidence": 1.0}

        # Step 1: DB lookup (parallel)
        db_results = await asyncio.gather(
            *[self._check_db(c) for c in citations], return_exceptions=True
        )

        # Step 2: Collect unverified ones for LLM cross-check
        verified: list[dict] = []
        unverified_refs: list[dict] = []

        for citation, db_result in zip(citations, db_results):
            if isinstance(db_result, Exception):
                log.warning("verification.db_check.failed", ref=citation.get("ref"), error=str(db_result))
                unverified_refs.append(citation)
            elif db_result is not None:
                verified.append(db_result)
            else:
                unverified_refs.append(citation)

        # Step 3: LLM plausibility check for unverified
        if unverified_refs:
            llm_results = await self._llm_plausibility_check(unverified_refs)
            verified.extend(llm_results)

        # Step 4: Compute rejection rate and alert if needed
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
        match = re.search(r"(?:มาตรา|section|sec\.?)\s*([0-9A-Za-z/.-]+)", ref, flags=re.IGNORECASE)
        return match.group(1) if match else None

    async def _llm_plausibility_check(self, citations: list[dict]) -> list[dict]:
        """Use fast LLM model to assess plausibility of citations not in DB."""
        settings = get_settings()
        if not settings.llm_verify_citations_with_llm:
            log.info("verification.llm_check.skipped", reason="disabled_by_config", citations=len(citations))
            return [
                {**c, "status": "UNVERIFIED", "note": "Not found in DB; LLM plausibility check disabled"}
                for c in citations
            ]

        refs_text = "\n".join(f"- {c.get('ref', '')}" for c in citations)
        user_msg = f"Assess these legal citations for plausibility:\n{refs_text}"

        try:
            result = await self._call_llm(
                model=settings.model_verification,
                system=_VERIFY_SYSTEM_PROMPT,
                user_message=user_msg,
                max_tokens=settings.llm_max_tokens_verification,
            )
            import json, re
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", result.text.strip())
            llm_assessments = json.loads(clean)

            output: list[dict] = []
            for citation in citations:
                ref = citation.get("ref", "")
                assessment = next(
                    (a for a in llm_assessments if a.get("ref", "").strip() in ref or ref in a.get("ref", "")),
                    None,
                )
                if assessment and assessment.get("plausible"):
                    output.append({**citation, "status": "UNVERIFIED", "note": "Not in DB, LLM plausible"})
                else:
                    reason = assessment.get("reason", "Unknown") if assessment else "LLM could not verify"
                    output.append({**citation, "status": "REJECTED", "reason": reason})
            return output

        except Exception as exc:
            log.warning("verification.llm_check.failed", error=str(exc))
            # Safe default — mark as UNVERIFIED (not REJECTED) to avoid false positives
            return [{**c, "status": "UNVERIFIED", "note": "Verification service unavailable"} for c in citations]
