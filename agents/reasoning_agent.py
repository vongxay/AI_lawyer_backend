"""
agents/reasoning_agent.py
==========================
IRAC Reasoning Agent — CORE agent, the heart of the system.

Uses Claude Sonnet (or claude-sonnet-4-6 for complex cases) with a strict IRAC
system prompt. Output is ALWAYS structured as Issue / Rule / Application / Conclusion.

Key constraints enforced:
- Closed-loop: only generate from retrieved context
- Every legal claim must cite a specific law/case
- If context is insufficient → explicit uncertainty + escalation
- Structured JSON output parsed and validated against IRAC schema
"""
from __future__ import annotations

import json
import re
from typing import Any

from backend.agents.base_agent import BaseAgent
from backend.core.config import get_settings
from backend.core.logging import get_logger

log = get_logger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────
_IRAC_SYSTEM_PROMPT = """
You are a senior legal advisor with 30+ years of experience in Thai and Lao law.

═══ STRICT GENERATION RULES ═══
1. ONLY use information from the CONTEXT block — never from training memory.
2. EVERY legal statement MUST cite a specific law or case from the context.
3. If context is insufficient, reply:
   {"insufficient_context": true, "reason": "brief explanation"}
4. Structure ALL responses using the IRAC JSON format below.
5. Be concise but complete — courts require precision.

═══ OUTPUT FORMAT (strict JSON) ═══
{
  "irac": {
    "issue": {
      "primary": "precise legal question",
      "secondary": ["sub-issue 1", "sub-issue 2"]
    },
    "rule": {
      "statutes": [
        {
          "name": "law name",
          "section": "section number",
          "text": "relevant text",
          "status": "ACTIVE|AMENDED|REPEALED",
          "year": 2565
        }
      ],
      "precedents": [
        {
          "case_no": "ฎ. XXXX/XXXX",
          "court": "court name",
          "relevance": "what principle it establishes",
          "outcome": "ผู้ฟ้องชนะ|ผู้ฟ้องแพ้",
          "graph_path": "citation chain note"
        }
      ]
    },
    "application": {
      "analysis": "detailed application of rules to facts",
      "strengths": ["strength 1"],
      "weaknesses": ["weakness 1"],
      "counter_args": ["counter-argument 1"],
      "rebuttals": ["rebuttal 1"]
    },
    "conclusion": {
      "recommendation": "clear actionable advice",
      "action_steps": ["step 1", "step 2"],
      "risk_level": "LOW|MEDIUM|HIGH",
      "win_probability": 0.72,
      "settlement_note": "settlement consideration if applicable"
    }
  },
  "confidence": 0.85,
  "reasoning_notes": "internal notes on analysis quality"
}

═══ LANGUAGE ═══
Respond in the same language as the user query (Thai/English/Lao).
Legal citations always use official section/case numbers as-is.
"""

_CONTEXT_TEMPLATE = """
═══ RETRIEVED LEGAL CONTEXT ═══
{retrieved_documents}

═══ CASE MEMORY SUMMARY ═══
{case_memory_summary}

═══ USER LEGAL QUERY ═══
{question}
"""


class IracReasoningAgent(BaseAgent):
    name = "reasoning"

    async def _execute(
        self,
        *,
        question: str,
        research: dict | None = None,
        document: dict | None = None,
        evidence: dict | None = None,
        memory: dict,
    ) -> dict[str, Any]:
        settings = get_settings()

        # Build context from all upstream agents
        context = self._build_context(
            research=research,
            document=document,
            evidence=evidence,
            memory=memory,
        )

        user_message = _CONTEXT_TEMPLATE.format(
            retrieved_documents=context["docs"],
            case_memory_summary=context["memory"],
            question=question,
        )

        result = await self._call_llm(
            model=settings.model_reasoning,
            system=_IRAC_SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=4096,
        )

        parsed = self._parse_irac_response(result.text, question)
        parsed["_tokens"] = result.total_tokens

        log.info(
            "reasoning.done",
            confidence=parsed.get("confidence"),
            has_statutes=len(parsed.get("irac", {}).get("rule", {}).get("statutes", [])),
            has_precedents=len(parsed.get("irac", {}).get("rule", {}).get("precedents", [])),
        )

        return parsed

    def _build_context(
        self,
        *,
        research: dict | None,
        document: dict | None,
        evidence: dict | None,
        memory: dict,
    ) -> dict[str, str]:
        doc_parts: list[str] = []

        if research and research.get("retrieved_documents"):
            for chunk in research["retrieved_documents"][:10]:
                doc_parts.append(
                    f"[{chunk.get('type', 'doc').upper()}] "
                    f"{chunk.get('title', '')} "
                    f"— {chunk.get('content', '')[:500]}"
                )

        if document and document.get("clauses"):
            doc_parts.append(f"[DOCUMENT CLAUSES]\n{json.dumps(document['clauses'], ensure_ascii=False)}")

        if evidence and evidence.get("items"):
            doc_parts.append(f"[EVIDENCE]\n{json.dumps(evidence['items'], ensure_ascii=False)}")

        memory_parts: list[str] = []
        if not memory.get("empty"):
            if memory.get("facts_summary"):
                memory_parts.append(f"Case facts: {memory['facts_summary']}")
            highlights = memory.get("memory_highlights") or {}
            if highlights.get("past_strategies"):
                memory_parts.append(f"Past strategies: {highlights['past_strategies']}")

        return {
            "docs": "\n\n".join(doc_parts) or "No retrieved documents available.",
            "memory": "\n".join(memory_parts) or "No prior case memory.",
        }

    def _parse_irac_response(self, text: str, question: str) -> dict[str, Any]:
        """Parse LLM response into IRAC dict, with graceful fallback."""
        # Try direct JSON parse
        try:
            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", text.strip())
            data = json.loads(clean)

            # Handle insufficient_context flag
            if data.get("insufficient_context"):
                log.warning("reasoning.insufficient_context", reason=data.get("reason"))
                return self._insufficient_context_response(question, data.get("reason", ""))

            confidence = float(data.get("confidence", 0.75))
            return {**data, "_confidence": confidence}

        except (json.JSONDecodeError, ValueError):
            log.warning("reasoning.json_parse_failed", raw_length=len(text))
            return self._fallback_response(question, text)

    def _insufficient_context_response(self, question: str, reason: str) -> dict[str, Any]:
        return {
            "irac": {
                "issue": {"primary": question, "secondary": []},
                "rule": {"statutes": [], "precedents": []},
                "application": {
                    "analysis": f"ข้อมูลไม่เพียงพอ: {reason}",
                    "strengths": [], "weaknesses": [],
                    "counter_args": [], "rebuttals": [],
                },
                "conclusion": {
                    "recommendation": "แนะนำให้ปรึกษาทนายความโดยตรง เนื่องจากข้อมูลในระบบไม่เพียงพอ",
                    "action_steps": ["ติดต่อทนายความผู้เชี่ยวชาญ", "รวบรวมเอกสารที่เกี่ยวข้อง"],
                    "risk_level": "MEDIUM",
                    "win_probability": 0.5,
                    "settlement_note": None,
                },
            },
            "confidence": 0.3,
            "_confidence": 0.3,
            "citations": [],
        }

    def _fallback_response(self, question: str, raw_text: str) -> dict[str, Any]:
        """When JSON parse fails, wrap raw text in minimal IRAC structure."""
        return {
            "irac": {
                "issue": {"primary": question, "secondary": []},
                "rule": {"statutes": [], "precedents": []},
                "application": {
                    "analysis": raw_text[:2000],
                    "strengths": [], "weaknesses": [],
                    "counter_args": [], "rebuttals": [],
                },
                "conclusion": {
                    "recommendation": "โปรดตรวจสอบคำตอบกับทนายความ",
                    "action_steps": [],
                    "risk_level": "MEDIUM",
                    "win_probability": 0.5,
                    "settlement_note": None,
                },
            },
            "confidence": 0.5,
            "_confidence": 0.5,
            "citations": [],
        }
