from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from backend.agents.document_agent import DocumentAnalysisAgent
from backend.agents.evidence_agent import EvidenceAnalyzerAgent
from backend.agents.reasoning_agent import IracReasoningAgent
from backend.agents.research_agent import LegalResearchAgent
from backend.agents.risk_strategy_agent import RiskStrategyAgent
from backend.agents.verification_agent import CitationVerificationAgent
from backend.memory.case_memory import CaseMemoryService
from backend.orchestrator.agent_selector import AgentSelector
from backend.orchestrator.query_classifier import QueryClassifier
from backend.services.audit_service import AuditService


@dataclass(frozen=True)
class OrchestrationResult:
    response: dict
    confidence: float
    agents_used: list[str]
    processing_time_ms: int
    escalated_to_expert: bool


class WorkflowManager:
    def __init__(self) -> None:
        self.classifier = QueryClassifier()
        self.selector = AgentSelector()
        self.audit = AuditService()

        self.case_memory = CaseMemoryService()

        self.research_agent = LegalResearchAgent()
        self.reasoning_agent = IracReasoningAgent()
        self.verification_agent = CitationVerificationAgent()
        self.document_agent = DocumentAnalysisAgent()
        self.evidence_agent = EvidenceAnalyzerAgent()
        self.risk_agent = RiskStrategyAgent()

    async def orchestrate(
        self,
        *,
        question: str,
        case_id: str | None,
        has_documents: bool = False,
        has_evidence: bool = False,
        user_id: str | None = None,
    ) -> OrchestrationResult:
        started = time.perf_counter()

        memory_ctx = await self.case_memory.get(case_id)
        query_type = await self.classifier.classify(question)
        plan = await self.selector.select(query_type)

        agents_used: list[str] = []

        results: dict[str, object] = {}
        async with asyncio.TaskGroup() as tg:
            if plan.use_research:
                agents_used.append("research")
                results["research"] = tg.create_task(self.research_agent.run(question=question, memory=memory_ctx))
            if plan.use_document and has_documents:
                agents_used.append("document")
                results["document"] = tg.create_task(self.document_agent.run(question=question))
            if plan.use_evidence and has_evidence:
                agents_used.append("evidence")
                results["evidence"] = tg.create_task(self.evidence_agent.run(question=question))

        agents_used.append("reasoning")
        irac = await self.reasoning_agent.run(
            question=question,
            research=results.get("research"),
            document=results.get("document"),
            evidence=results.get("evidence"),
            memory=memory_ctx,
        )

        agents_used.append("verification")
        verification_task = asyncio.create_task(self.verification_agent.verify(irac.get("citations", [])))

        risk = None
        if plan.use_risk:
            agents_used.append("risk")
            risk = await self.risk_agent.run(question=question, irac=irac)

        verification = await verification_task

        confidence = float(irac.get("confidence", 0.75))
        if verification.get("citations_verified") is False:
            confidence = max(0.0, confidence - 0.15)

        escalated = confidence < 0.70

        await self.case_memory.update(case_id=case_id, question=question, irac=irac)
        await self.audit.log(user_id=user_id, agent="orchestrator", query=question, confidence=confidence, agents_used=agents_used)

        processing_ms = int((time.perf_counter() - started) * 1000)
        response = {
            **irac,
            "citations": verification.get("citations", []),
            "citations_verified": verification.get("citations_verified", True),
            "agents_used": agents_used,
            "processing_time_ms": processing_ms,
            "escalated_to_expert": escalated,
            "risk": risk,
            "disclaimer": "คำตอบนี้เป็นข้อมูลทั่วไปเท่านั้น ไม่ใช่คำปรึกษาทางกฎหมายอย่างเป็นทางการ",
        }
        return OrchestrationResult(
            response=response,
            confidence=confidence,
            agents_used=agents_used,
            processing_time_ms=processing_ms,
            escalated_to_expert=escalated,
        )

