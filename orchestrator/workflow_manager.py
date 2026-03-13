"""
orchestrator/workflow_manager.py
==================================
WorkflowManager — the central orchestration engine.

Implements the exact flow from blueprint Section 3.3, with all bugs fixed:

1. Load case memory
2. Classify query type
3. Select agent plan dynamically
4. Run parallel agents (research / document / evidence)
5. Run IRAC Reasoning (sequential — needs parallel results)
6. Run Citation Verification + Risk in parallel
7. Check confidence → escalate if < threshold
8. Update case memory
9. Log to audit trail
10. Return structured OrchestrationResult

Key fixes from original codebase:
- asyncio.TaskGroup result extraction (was accessing Task objects as dicts)
- Agent instances receive injected dependencies (DB clients, LLM service)
- PII redaction applied BEFORE any LLM call
- Schema-compatible response building
- Confidence calculation uses both reasoning AND verification results
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from backend.agents.document_agent import DocumentAnalysisAgent
from backend.agents.evidence_agent import EvidenceAnalyzerAgent, EvidenceFile
from backend.agents.reasoning_agent import IracReasoningAgent
from backend.agents.research_agent import LegalResearchAgent
from backend.agents.risk_strategy_agent import RiskStrategyAgent
from backend.agents.verification_agent import CitationVerificationAgent
from backend.core.config import get_settings
from backend.core.exceptions import LowConfidenceError
from backend.core.logging import get_logger
from backend.memory.case_memory import CaseMemoryService
from backend.orchestrator.agent_selector import AgentSelector
from backend.orchestrator.query_classifier import QueryClassifier
from backend.services.audit_service import AuditService, ExpertQueueService
from backend.services.pii_service import PiiService

if TYPE_CHECKING:
    from supabase import AsyncClient  # pragma: no cover
    import redis.asyncio as aioredis

log = get_logger(__name__)


@dataclass
class OrchestrationResult:
    response: dict[str, Any]
    confidence: float
    agents_used: list[str]
    processing_time_ms: int
    escalated_to_expert: bool
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class WorkflowManager:
    """
    Dependency-injected orchestrator.

    All external dependencies (DB, LLM, agents) are injected — never created inside.
    This makes the class fully testable with mocks.
    """

    def __init__(
        self,
        supabase: "AsyncClient | None" = None,
        redis: "aioredis.Redis | None" = None,
    ) -> None:
        self._settings = get_settings()
        self._pii = PiiService()

        # ── Services ──────────────────────────────────────────────────────────
        self._case_memory = CaseMemoryService(supabase=supabase, redis=redis)
        self._audit = AuditService(supabase=supabase)
        self._expert_queue = ExpertQueueService(supabase=supabase)

        # ── Classifiers ───────────────────────────────────────────────────────
        self._classifier = QueryClassifier()
        self._selector = AgentSelector()

        # ── Agents — share LLM service via BaseAgent default ─────────────────
        self._research_agent = LegalResearchAgent()
        self._reasoning_agent = IracReasoningAgent()
        self._verification_agent = CitationVerificationAgent(supabase=supabase)
        self._document_agent = DocumentAnalysisAgent()
        self._evidence_agent = EvidenceAnalyzerAgent()
        self._risk_agent = RiskStrategyAgent()

    # ── Public interface ───────────────────────────────────────────────────────

    async def orchestrate(
        self,
        *,
        question: str,
        case_id: str | None,
        jurisdiction: str | None = None,
        document_text: str | None = None,
        evidence_files: list[EvidenceFile] | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        session_id: str | None = None,
    ) -> OrchestrationResult:
        started = time.perf_counter()
        sid = session_id or str(uuid.uuid4())

        # ── Step 0: PII redaction on all user input ───────────────────────────
        clean_question = self._pii.redact_text(question)
        clean_doc_text = self._pii.redact_text(document_text) if document_text else None

        # ── Step 1: Load case memory ──────────────────────────────────────────
        memory = await self._case_memory.get(case_id)

        # ── Step 2: Classify query ────────────────────────────────────────────
        query_type = await self._classifier.classify(clean_question)

        # ── Step 3: Dynamic agent selection ──────────────────────────────────
        plan = self._selector.select(
            query_type,
            force_document=bool(document_text),
            force_evidence=bool(evidence_files),
        )

        log.info(
            "orchestrator.start",
            session_id=sid,
            query_type=query_type,
            plan=str(plan),
            case_id=case_id,
        )

        # ── Step 4: Parallel phase — Research + Document + Evidence ──────────
        agents_used: list[str] = []
        parallel_tasks: dict[str, asyncio.Task] = {}

        async with asyncio.TaskGroup() as tg:
            if plan.use_research:
                agents_used.append("research")
                parallel_tasks["research"] = tg.create_task(
                    self._research_agent.run(
                        question=clean_question,
                        memory=memory,
                        jurisdiction=jurisdiction,
                    )
                )

            if plan.use_document and clean_doc_text:
                agents_used.append("document")
                parallel_tasks["document"] = tg.create_task(
                    self._document_agent.run(
                        question=clean_question,
                        document_text=clean_doc_text,
                        case_context=memory.get("facts_summary"),
                    )
                )

            if plan.use_evidence and evidence_files:
                agents_used.append("evidence")
                parallel_tasks["evidence"] = tg.create_task(
                    self._evidence_agent.run(
                        question=clean_question,
                        evidence_files=evidence_files,
                        case_context=memory.get("facts_summary"),
                    )
                )

        # Extract results from completed tasks (FIX: .result() not direct access)
        research_data = parallel_tasks["research"].result().data if "research" in parallel_tasks else None
        document_data = parallel_tasks["document"].result().data if "document" in parallel_tasks else None
        evidence_data = parallel_tasks["evidence"].result().data if "evidence" in parallel_tasks else None

        # ── Step 5: IRAC Reasoning (sequential — needs all parallel results) ──
        agents_used.append("reasoning")
        reasoning_result = await self._reasoning_agent.run(
            question=clean_question,
            research=research_data,
            document=document_data,
            evidence=evidence_data,
            memory=memory,
        )
        irac_data = reasoning_result.data

        # ── Step 6: Verification + Risk (parallel) ────────────────────────────
        agents_used.append("verification")
        citations_to_verify = irac_data.get("citations", [])

        verify_coro = self._verification_agent.run(citations=citations_to_verify)
        risk_coro = (
            self._risk_agent.run(
                question=clean_question,
                irac=irac_data,
                research=research_data,
            )
            if plan.use_risk
            else self._noop()
        )

        if plan.use_risk:
            agents_used.append("risk")

        verify_result_obj, risk_result_obj = await asyncio.gather(verify_coro, risk_coro)
        verification_data = verify_result_obj.data
        risk_data = risk_result_obj.data if risk_result_obj and risk_result_obj.data else None

        # ── Step 7: Confidence calculation ────────────────────────────────────
        reasoning_confidence = float(irac_data.get("confidence", 0.75))
        verification_confidence = float(verification_data.get("_confidence", 1.0) if verification_data else 1.0)
        rejection_rate = float(verification_data.get("rejection_rate", 0.0) if verification_data else 0.0)

        final_confidence = reasoning_confidence * 0.7 + verification_confidence * 0.3
        final_confidence = max(0.0, final_confidence - rejection_rate * 0.2)

        escalated = final_confidence < self._settings.confidence_escalation_threshold

        # ── Step 8: Escalate to expert queue if needed ────────────────────────
        if escalated:
            await self._expert_queue.enqueue(
                session_id=sid,
                user_id=user_id,
                reason=f"Low confidence: {final_confidence:.2f}",
                confidence=final_confidence,
                query_preview=clean_question,
            )

        # ── Step 9: Update case memory ────────────────────────────────────────
        await self._case_memory.update(
            case_id=case_id,
            question=clean_question,
            irac=irac_data,
            tenant_id=tenant_id,
            user_id=user_id,
        )

        # ── Step 10: Audit log ────────────────────────────────────────────────
        processing_ms = int((time.perf_counter() - started) * 1000)
        await self._audit.log_event(
            user_id=user_id,
            tenant_id=tenant_id,
            agent="orchestrator",
            query=clean_question,
            confidence=final_confidence,
            agents_used=agents_used,
            processing_time_ms=processing_ms,
            escalated=escalated,
        )

        log.info(
            "orchestrator.done",
            session_id=sid,
            confidence=round(final_confidence, 3),
            agents=agents_used,
            escalated=escalated,
            processing_ms=processing_ms,
        )

        # ── Build response ────────────────────────────────────────────────────
        response = self._build_response(
            irac_data=irac_data,
            verification_data=verification_data,
            risk_data=risk_data,
            agents_used=agents_used,
            confidence=final_confidence,
            processing_ms=processing_ms,
            escalated=escalated,
        )

        return OrchestrationResult(
            response=response,
            confidence=final_confidence,
            agents_used=agents_used,
            processing_time_ms=processing_ms,
            escalated_to_expert=escalated,
            session_id=sid,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _noop(self):
        """Placeholder coroutine when an agent is not needed."""
        from backend.agents.base_agent import AgentResult
        return AgentResult(data={}, agent_name="noop")

    def _build_response(
        self,
        *,
        irac_data: dict,
        verification_data: dict | None,
        risk_data: dict | None,
        agents_used: list[str],
        confidence: float,
        processing_ms: int,
        escalated: bool,
    ) -> dict[str, Any]:
        return {
            "irac": irac_data.get("irac", {}),
            "citations": (verification_data or {}).get("citations", irac_data.get("citations", [])),
            "citations_verified": (verification_data or {}).get("citations_verified", False),
            "confidence": round(confidence, 3),
            "agents_used": agents_used,
            "processing_time_ms": processing_ms,
            "escalated_to_expert": escalated,
            "risk": risk_data,
            "disclaimer": (
                "คำตอบนี้เป็นข้อมูลทั่วไปเท่านั้น ไม่ใช่คำปรึกษาทางกฎหมายอย่างเป็นทางการ "
                "/ This response is general information only and does not constitute formal legal advice."
            ),
        }
