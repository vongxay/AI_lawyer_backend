"""
Post-answer guardrails for legal AI responses.

These checks are deterministic and run after reasoning + citation verification.
They do not replace a lawyer, but they prevent the system from presenting weakly
grounded analysis as high-confidence legal advice.
"""
from __future__ import annotations

from typing import Any

from core.jurisdiction import canonical_jurisdiction


class LegalAnswerGuardrails:
    def assess(
        self,
        *,
        jurisdiction: str | None,
        irac_data: dict[str, Any],
        verification_data: dict[str, Any] | None,
        research_quality: dict[str, Any],
    ) -> dict[str, Any]:
        citations = (verification_data or {}).get("citations") or []
        verified = [c for c in citations if isinstance(c, dict) and c.get("status") == "VERIFIED"]
        rejected = [c for c in citations if isinstance(c, dict) and c.get("status") == "REJECTED"]
        official_lao = [
            c for c in citations
            if isinstance(c, dict)
            and any("laoofficialgazette.gov.la" in str(link).lower() for link in c.get("source_links", []))
        ]

        warnings: list[str] = []
        cap = float(research_quality.get("confidence_cap", 0.75))
        requires_human_review = False

        if not citations:
            warnings.append("no_citations_extracted")
            cap = min(cap, 0.45)
            requires_human_review = True
        elif not verified:
            warnings.append("no_verified_citations")
            cap = min(cap, 0.55)
            requires_human_review = True

        if rejected:
            warnings.append("rejected_citations_present")
            cap = min(cap, 0.5)
            requires_human_review = True

        if canonical_jurisdiction(jurisdiction) == "laos":
            if not official_lao:
                warnings.append("lao_official_source_not_verified")
                cap = min(cap, 0.72 if verified else 0.5)
            if self._uses_precedent_as_binding(irac_data):
                warnings.append("lao_precedent_binding_risk")
                cap = min(cap, 0.65)
                requires_human_review = True

        return {
            "confidence_cap": cap,
            "warnings": warnings,
            "requires_human_review": requires_human_review,
            "verified_citation_count": len(verified),
            "rejected_citation_count": len(rejected),
        }

    def _uses_precedent_as_binding(self, irac_data: dict[str, Any]) -> bool:
        irac = irac_data.get("irac") or {}
        text = " ".join([
            str(irac.get("application", {}).get("analysis", "")),
            str(irac.get("conclusion", {}).get("recommendation", "")),
        ]).casefold()
        risky_terms = ("binding precedent", "บรรทัดฐานผูกพัน", "precedent is binding")
        return any(term in text for term in risky_terms)
