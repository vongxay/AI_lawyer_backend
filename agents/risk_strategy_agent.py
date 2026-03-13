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
from core.logging import get_logger

log = get_logger(__name__)

_RISK_SYSTEM_PROMPT = """
You are a senior litigation strategist with 25+ years experience in Thai and Lao courts.
You assess litigation risk and recommend optimal legal strategies.

Return strict JSON:
{
  "win_probability": 0.72,
  "win_probability_basis": "why this probability — cite specific legal strengths/weaknesses",
  "risk_level": "LOW|MEDIUM|HIGH",
  "cost_estimate_thb": {
    "min": 50000,
    "max": 200000,
    "notes": "what drives cost variance"
  },
  "timeline_estimate_days": {
    "negotiation": 30,
    "litigation_first_instance": 365,
    "appeal": 540
  },
  "strategic_options": [
    {
      "name": "Negotiate and Settle",
      "description": "brief description",
      "pros": ["faster", "lower cost"],
      "cons": ["may get less than full claim"],
      "recommended_when": "when cost > potential recovery",
      "success_likelihood": "HIGH|MEDIUM|LOW",
      "est_days": 45
    },
    {
      "name": "Full Litigation",
      "description": "proceed to court",
      "pros": ["binding judgment", "full remedy possible"],
      "cons": ["expensive", "time-consuming"],
      "recommended_when": "when principle matters or amount is large",
      "success_likelihood": "MEDIUM",
      "est_days": 365
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
        **kwargs,
    ) -> dict[str, Any]:
        settings = get_settings()

        # Extract relevant context from IRAC result
        irac_data = irac.get("irac", {})
        conclusion = irac_data.get("conclusion", {})
        application = irac_data.get("application", {})

        context_summary = self._build_context(question, irac_data, conclusion, application, research)

        result = await self._call_llm(
            model=settings.model_risk,
            system=_RISK_SYSTEM_PROMPT,
            user_message=context_summary,
            max_tokens=2048,
        )

        parsed = self._parse_strategy_response(result.text, conclusion)
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

    def _parse_strategy_response(self, text: str, irac_conclusion: dict) -> dict[str, Any]:
        try:
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", text.strip())
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            log.warning("risk.json_parse_failed")
            # Graceful fallback using IRAC conclusion data
            return {
                "win_probability": irac_conclusion.get("win_probability", 0.5),
                "risk_level": irac_conclusion.get("risk_level", "MEDIUM"),
                "strategic_options": [
                    {
                        "name": "เจรจา / Negotiate",
                        "pros": ["เร็ว", "ลดต้นทุน"],
                        "cons": ["อาจได้ผลลัพธ์น้อยกว่า"],
                        "est_days": 30,
                        "success_likelihood": "MEDIUM",
                    },
                    {
                        "name": "ดำเนินคดี / Litigate",
                        "pros": ["ได้คำพิพากษา", "บังคับคดีได้"],
                        "cons": ["ใช้เวลา", "ค่าใช้จ่ายสูง"],
                        "est_days": 365,
                        "success_likelihood": "MEDIUM",
                    },
                ],
                "recommended_option": "เจรจา / Negotiate",
                "immediate_actions": irac_conclusion.get("action_steps", []),
            }
