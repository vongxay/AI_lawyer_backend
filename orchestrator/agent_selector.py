"""
orchestrator/agent_selector.py
================================
Maps query type + request flags to an AgentPlan.

Design rule: never run more agents than needed.
Each flag in AgentPlan enables one agent — disabled agents are never instantiated.
"""
from __future__ import annotations

from dataclasses import dataclass

from orchestrator.query_classifier import QueryType


@dataclass(frozen=True)
class AgentPlan:
    use_research: bool = True
    use_reasoning: bool = True
    use_verification: bool = True
    use_document: bool = False
    use_evidence: bool = False
    use_risk: bool = False


# ── Static plans per query type ────────────────────────────────────────────────
# Matches Section 3.1 of blueprint exactly.
AGENT_PLANS: dict[QueryType, AgentPlan] = {
    "legal_question": AgentPlan(
        use_research=True,
        use_reasoning=True,
        use_verification=True,
    ),
    "document_review": AgentPlan(
        use_research=True,
        use_reasoning=True,
        use_verification=True,
        use_document=True,
    ),
    "case_strategy": AgentPlan(
        use_research=True,
        use_reasoning=True,
        use_verification=True,
        use_risk=True,
    ),
    "evidence_analysis": AgentPlan(
        use_research=True,
        use_reasoning=True,
        use_verification=True,
        use_evidence=True,
    ),
    "draft_document": AgentPlan(
        use_research=True,
        use_reasoning=True,
        use_verification=True,
        use_document=True,
    ),
}


class AgentSelector:
    def select(
        self,
        query_type: QueryType,
        *,
        force_document: bool = False,
        force_evidence: bool = False,
    ) -> AgentPlan:
        """
        Return the agent plan for this query type.
        force_* flags override plan when caller knows files are present.
        """
        base = AGENT_PLANS[query_type]
        if force_document or force_evidence:
            return AgentPlan(
                use_research=base.use_research,
                use_reasoning=base.use_reasoning,
                use_verification=base.use_verification,
                use_document=base.use_document or force_document,
                use_evidence=base.use_evidence or force_evidence,
                use_risk=base.use_risk,
            )
        return base
