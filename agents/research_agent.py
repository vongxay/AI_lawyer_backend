from __future__ import annotations

from backend.agents.base_agent import BaseAgent


class LegalResearchAgent(BaseAgent):
    name = "research"

    async def run(self, *, question: str, memory: dict) -> dict:
        # Stubbed retrieval output: callers expect "retrieved_documents" content + citations.
        return {
            "retrieved_documents": [
                {
                    "type": "law",
                    "title": "ประมวลกฎหมายแพ่งและพาณิชย์",
                    "section": "มาตรา 420",
                    "text": "ผู้ใดจงใจหรือประมาทเลินเล่อ ทำต่อผู้อื่นโดยผิดกฎหมายให้เขาเสียหาย...",
                    "jurisdiction": "TH",
                    "status": "ACTIVE",
                    "year": 2535,
                }
            ],
            "case_graph_context": [],
            "memory_highlights": memory,
        }

