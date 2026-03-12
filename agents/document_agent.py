from __future__ import annotations

from backend.agents.base_agent import BaseAgent


class DocumentAnalysisAgent(BaseAgent):
    name = "document"

    async def run(self, *, question: str) -> dict:
        return {
            "clauses": [],
            "risk_flags": [],
            "summary": "stub document analysis",
        }

