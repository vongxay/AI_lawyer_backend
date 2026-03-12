from __future__ import annotations

from backend.agents.base_agent import BaseAgent


class EvidenceAnalyzerAgent(BaseAgent):
    name = "evidence"

    async def run(self, *, question: str) -> dict:
        return {
            "items": [],
            "overall_strength": "UNKNOWN",
            "gaps": ["stub: ต้องแนบไฟล์หลักฐานเพื่อวิเคราะห์"],
        }

