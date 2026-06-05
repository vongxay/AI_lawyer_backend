"""
agents/risk_strategy_agent.py
==============================
Risk & Strategy Agent — SPECIALIST, invoked for case_strategy queries.

Provides:
- Win probability assessment (calibrated, not just IRAC's estimate)
- Strategic options with pros/cons and timeline
- Settlement vs. litigation recommendation
- Cost-benefit analysis framing
"""
from __future__ import annotations

import json
import re
from typing import Any

from agents.base_agent import BaseAgent
from core.config import get_settings
from core.jurisdiction import infer_response_language, response_language_instruction
from core.logging import get_logger

log = get_logger(__name__)

_RISK_SYSTEM_PROMPT = """
You are a senior litigation strategist with 25+ years experience in Thai and Lao courts.
You assess litigation risk and recommend optimal legal strategies.

Return strict JSON:
{
  "win_probability": 0.72,
  "win_probability_basis": "why this probability; cite specific legal strengths/weaknesses from IRAC and retrieved authority",
  "risk_level": "LOW|MEDIUM|HIGH",
  "cost_estimate": {
    "currency": "LAK|THB|USD|UNKNOWN",
    "min": null,
    "max": null,
    "notes": "only estimate when facts and jurisdiction support it; otherwise explain why unknown"
  },
  "timeline_estimate_days": {
    "option_name": 0
  },
  "strategic_options": [
    {
      "name": "strategy option grounded in the specific facts",
      "description": "brief description tied to the user's facts",
      "pros": ["fact-specific advantage"],
      "cons": ["fact-specific downside"],
      "recommended_when": "specific condition from the case facts",
      "success_likelihood": "HIGH|MEDIUM|LOW",
      "est_days": null
    }
  ],
  "recommended_option": "option name",
  "recommended_option_rationale": "why this is the best approach given the specific facts",
  "critical_deadlines": [
    {"event": "prescription deadline", "deadline_note": "check from date of incident"}
  ],
  "preserve_evidence_checklist": ["item to preserve 1", "item 2"],
  "immediate_actions": ["action to take right now 1", "action 2"]
}
Do not add text before or after the JSON.
"""


class RiskStrategyAgent(BaseAgent):
    name = "risk"

    async def _execute(
        self,
        *,
        question: str,
        irac: dict,
        research: dict | None = None,
        response_language: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        settings = get_settings()
        language = infer_response_language(question, response_language)

        # Extract relevant context from IRAC result
        irac_data = irac.get("irac", {})
        conclusion = irac_data.get("conclusion", {})
        application = irac_data.get("application", {})

        context_summary = self._build_context(question, irac_data, conclusion, application, research)

        result = await self._call_llm(
            model=settings.model_risk,
            system=f"{_RISK_SYSTEM_PROMPT}\n\nLANGUAGE OVERRIDE:\n{response_language_instruction(language)}",
            user_message=context_summary,
            max_tokens=settings.llm_max_tokens_risk,
        )

        parsed = self._parse_strategy_response(result.text, conclusion, language)
        parsed["_tokens"] = result.total_tokens
        parsed["_confidence"] = 0.80

        log.info(
            "risk.analyzed",
            win_probability=parsed.get("win_probability"),
            risk_level=parsed.get("risk_level"),
            options=len(parsed.get("strategic_options", [])),
        )
        return parsed

    def _build_context(
        self,
        question: str,
        irac_data: dict,
        conclusion: dict,
        application: dict,
        research: dict | None,
    ) -> str:
        parts = [f"Legal question: {question}\n"]

        if irac_data.get("issue", {}).get("primary"):
            parts.append(f"Legal issue: {irac_data['issue']['primary']}")

        if application.get("strengths"):
            parts.append(f"Case strengths: {', '.join(application['strengths'])}")
        if application.get("weaknesses"):
            parts.append(f"Case weaknesses: {', '.join(application['weaknesses'])}")
        if application.get("counter_args"):
            parts.append(f"Opposing arguments: {', '.join(application['counter_args'])}")

        if conclusion:
            parts.append(f"IRAC win probability estimate: {conclusion.get('win_probability', 'unknown')}")
            parts.append(f"IRAC risk level: {conclusion.get('risk_level', 'unknown')}")

        if research and research.get("retrieved_documents"):
            law_refs = [
                f"{d.get('title', '')} {d.get('section', '')}"
                for d in research["retrieved_documents"][:3]
            ]
            parts.append(f"Relevant laws: {'; '.join(law_refs)}")

        return "\n".join(parts)

    def _parse_strategy_response(
        self,
        text: str,
        irac_conclusion: dict,
        language: str = "en",
    ) -> dict[str, Any]:
        try:
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", text.strip())
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            log.warning("risk.json_parse_failed")
            return self._safe_strategy_fallback(irac_conclusion, language)

    def _safe_strategy_fallback(self, irac_conclusion: dict, language: str) -> dict[str, Any]:
        """Never invent litigation strategy when the risk model output is unusable."""
        action_steps = [
            str(step).strip()
            for step in (irac_conclusion.get("action_steps") or [])
            if str(step).strip()
        ]
        if language == "lo":
            basis = (
                "ລະບົບວິເຄາະຄວາມສ່ຽງບໍ່ໄດ້ສົ່ງຜົນເປັນໂຄງສ້າງທີ່ໃຊ້ງານໄດ້, "
                "ຈຶ່ງບໍ່ຄວນສ້າງຍຸດທະສາດຄະດີຂຶ້ນເອງ."
            )
            recommended = "ກວດຄືນຜົນວິເຄາະ IRAC ແລະ ໃຫ້ທະນາຍຄວາມກວດຂໍ້ເທັດຈິງກ່ອນຕັດສິນໃຈ."
        elif language == "th":
            basis = (
                "ระบบวิเคราะห์ความเสี่ยงไม่ได้ส่งผลลัพธ์เป็นโครงสร้างที่ใช้งานได้ "
                "จึงไม่ควรสร้างกลยุทธ์คดีขึ้นเอง"
            )
            recommended = "ตรวจทานผลวิเคราะห์ IRAC และให้ทนายความตรวจข้อเท็จจริงก่อนตัดสินใจ"
        else:
            basis = (
                "The risk model did not return usable structured output, so the system did not invent litigation strategy."
            )
            recommended = "Review the IRAC analysis and have a qualified lawyer check the facts before deciding."

        return {
            "win_probability": irac_conclusion.get("win_probability", 0.0),
            "win_probability_basis": basis,
            "risk_level": irac_conclusion.get("risk_level", "MEDIUM"),
            "cost_estimate": {"currency": "UNKNOWN", "min": None, "max": None, "notes": basis},
            "timeline_estimate_days": {},
            "strategic_options": [],
            "recommended_option": recommended,
            "recommended_option_rationale": basis,
            "critical_deadlines": [],
            "preserve_evidence_checklist": [],
            "immediate_actions": action_steps,
        }
