"""
agents/document_agent.py
=========================
Document Analysis Agent — SPECIALIST, invoked only when files are uploaded.

Handles:
- PDF / Word contract analysis (clause extraction, risk flags)
- Structured output: clauses, risk_flags, anomalies, summary

Uses GPT-4o for best Thai/EN document understanding.
"""
from __future__ import annotations

import base64
import json
import re
from typing import Any

from agents.base_agent import BaseAgent
from core.config import get_settings
from core.logging import get_logger

log = get_logger(__name__)

_DOCUMENT_SYSTEM_PROMPT = """
You are a senior legal document analyst specialising in Thai and international contracts.

Analyse the provided document and return strict JSON:
{
  "document_type": "contract|letter|evidence|form|other",
  "language": "TH|EN|LA|MIXED",
  "summary": "2-3 sentence executive summary",
  "parties": ["Party A name", "Party B name"],
  "clauses": [
    {
      "clause_no": "1",
      "title": "clause title",
      "text": "clause text (truncated to 200 chars)",
      "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
      "risk_reason": "why this is risky (if applicable)",
      "recommendation": "what to change or watch out for"
    }
  ],
  "risk_flags": [
    {
      "type": "UNFAIR_TERM|ILLEGAL_CLAUSE|MISSING_PROTECTION|AMBIGUOUS|UNUSUAL",
      "description": "what the problem is",
      "clause_ref": "clause number reference",
      "severity": "LOW|MEDIUM|HIGH|CRITICAL"
    }
  ],
  "missing_standard_clauses": ["clause type that should be present but isn't"],
  "overall_risk": "LOW|MEDIUM|HIGH|CRITICAL",
  "overall_risk_summary": "brief overall assessment"
}

Do not add any text before or after the JSON.
"""


class DocumentAnalysisAgent(BaseAgent):
    name = "document"

    async def _execute(
        self,
        *,
        question: str,
        document_text: str | None = None,
        document_base64: str | None = None,
        case_context: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        settings = get_settings()

        if not document_text and not document_base64:
            return self._no_document_result()

        # Build user message with document content
        if document_text:
            user_msg = self._build_text_prompt(question, document_text, case_context)
        else:
            # Vision-based analysis for scanned/image docs
            user_msg = self._build_vision_prompt(question, document_base64, case_context)

        result = await self._call_llm(
            model=settings.model_document,
            system=_DOCUMENT_SYSTEM_PROMPT,
            user_message=user_msg,
            max_tokens=4096,
        )

        parsed = self._parse_document_response(result.text)
        parsed["_tokens"] = result.total_tokens
        parsed["_confidence"] = self._compute_confidence(parsed)

        log.info(
            "document.analyzed",
            doc_type=parsed.get("document_type"),
            overall_risk=parsed.get("overall_risk"),
            clauses=len(parsed.get("clauses", [])),
            flags=len(parsed.get("risk_flags", [])),
        )
        return parsed

    def _build_text_prompt(self, question: str, text: str, context: str | None) -> str:
        ctx = f"\nCase context: {context}" if context else ""
        return f"Legal question: {question}{ctx}\n\nDOCUMENT CONTENT:\n{text[:8000]}"

    def _build_vision_prompt(self, question: str, b64: str | None, context: str | None) -> str:
        ctx = f"\nCase context: {context}" if context else ""
        return f"Legal question: {question}{ctx}\n\n[Document provided as image/PDF — analyse the content visible.]"

    def _parse_document_response(self, text: str) -> dict[str, Any]:
        try:
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", text.strip())
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            log.warning("document.json_parse_failed")
            return {
                "document_type": "unknown",
                "summary": text[:500],
                "clauses": [],
                "risk_flags": [],
                "overall_risk": "MEDIUM",
                "overall_risk_summary": "Manual review recommended — analysis incomplete.",
            }

    def _compute_confidence(self, data: dict) -> float:
        if not data.get("clauses") and not data.get("summary"):
            return 0.3
        if data.get("overall_risk") == "unknown":
            return 0.5
        return 0.85

    def _no_document_result(self) -> dict[str, Any]:
        return {
            "document_type": "none",
            "summary": "No document provided for analysis.",
            "clauses": [],
            "risk_flags": [],
            "overall_risk": "LOW",
            "overall_risk_summary": "Upload a document to receive analysis.",
            "_confidence": 1.0,
            "_tokens": 0,
        }
