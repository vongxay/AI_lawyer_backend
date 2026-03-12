from __future__ import annotations

from dataclasses import dataclass

from backend.orchestrator.query_classifier import QueryType


@dataclass(frozen=True)
class AgentPlan:
    use_research: bool
    use_reasoning: bool
    use_verification: bool
    use_document: bool
    use_evidence: bool
    use_risk: bool


AGENT_PLANS: dict[QueryType, AgentPlan] = {
    "legal_question": AgentPlan(True, True, True, False, False, False),
    "document_review": AgentPlan(True, True, True, True, False, False),
    "case_strategy": AgentPlan(True, True, True, False, False, True),
    "evidence_analysis": AgentPlan(True, True, True, False, True, False),
    "draft_document": AgentPlan(True, True, True, True, False, False),
}


class AgentSelector:
    async def select(self, query_type: QueryType) -> AgentPlan:
        return AGENT_PLANS[query_type]

