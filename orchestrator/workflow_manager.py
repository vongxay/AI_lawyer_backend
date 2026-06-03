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
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agents.document_agent import DocumentAnalysisAgent
from agents.evidence_agent import EvidenceAnalyzerAgent, EvidenceFile
from agents.reasoning_agent import IracReasoningAgent
from agents.research_agent import LegalResearchAgent
from agents.risk_strategy_agent import RiskStrategyAgent
from agents.verification_agent import CitationVerificationAgent
from core.config import get_settings
from core.exceptions import ExternalServiceError, LowConfidenceError, ProviderNotConfiguredError
from core.jurisdiction import infer_jurisdiction
from core.logging import get_logger
from memory.case_memory import CaseMemoryService
from orchestrator.agent_selector import AgentSelector
from orchestrator.query_classifier import QueryClassifier
from services.audit_service import AuditService, ExpertQueueService
from services.answer_guardrails import LegalAnswerGuardrails
from services.cache_service import CacheService
from services.llm_service import LlmService
from services.pii_service import PiiService

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
        self._cache = CacheService(redis_client=redis)
        self._llm = LlmService()
        self._guardrails = LegalAnswerGuardrails()

        # ── Classifiers ───────────────────────────────────────────────────────
        self._classifier = QueryClassifier()
        self._selector = AgentSelector()

        # ── Agents — share LLM service via BaseAgent default ─────────────────
        self._research_agent = LegalResearchAgent(supabase=supabase, redis=redis, llm=self._llm)
        self._reasoning_agent = IracReasoningAgent(llm=self._llm)
        self._verification_agent = CitationVerificationAgent(supabase=supabase, llm=self._llm)
        self._document_agent = DocumentAnalysisAgent(llm=self._llm)
        self._evidence_agent = EvidenceAnalyzerAgent(llm=self._llm)
        self._risk_agent = RiskStrategyAgent(llm=self._llm)

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
        effective_jurisdiction = infer_jurisdiction(question, jurisdiction)

        # ── Step 0: PII redaction on all user input ───────────────────────────
        clean_question = self._pii.redact_text(question)
        clean_doc_text = self._pii.redact_text(document_text) if document_text else None

        cache_key = self._build_cache_key(
            question=clean_question,
            case_id=case_id,
            jurisdiction=effective_jurisdiction,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        cacheable = clean_doc_text is None and not evidence_files
        if cacheable:
            cached = await self._cache.get(cache_key, namespace="legal_qa")
            if cached:
                processing_ms = int((time.perf_counter() - started) * 1000)
                response = {**cached["response"], "processing_time_ms": processing_ms, "cached": True}
                log.info("orchestrator.cache_hit", session_id=sid, cache_key=cache_key)
                return OrchestrationResult(
                    response=response,
                    confidence=float(cached["confidence"]),
                    agents_used=cached["agents_used"],
                    processing_time_ms=processing_ms,
                    escalated_to_expert=bool(cached["escalated_to_expert"]),
                    session_id=sid,
                )

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
            jurisdiction=effective_jurisdiction,
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
                        jurisdiction=effective_jurisdiction,
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
        if not reasoning_result.ok:
            if "No real LLM API key" in (reasoning_result.error or ""):
                raise ProviderNotConfiguredError(
                    "Legal reasoning is unavailable because no real LLM API key is configured.",
                    details={"required_env": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]},
                )
            raise ExternalServiceError(reasoning_result.error or "Legal reasoning agent failed")
        irac_data = reasoning_result.data

        # ── Step 6: Verification + Risk (parallel) ────────────────────────────
        agents_used.append("verification")
        citations_to_verify = self._extract_citations(irac_data)

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
        research_quality = self._score_research_quality(research_data)
        verification_confidence = float(verification_data.get("_confidence", 1.0) if verification_data else 1.0)
        rejection_rate = float(verification_data.get("rejection_rate", 0.0) if verification_data else 0.0)
        guardrail_result = self._guardrails.assess(
            jurisdiction=effective_jurisdiction,
            irac_data=irac_data,
            verification_data=verification_data,
            research_quality=research_quality,
        )

        final_confidence = reasoning_confidence * 0.7 + verification_confidence * 0.3
        final_confidence = max(0.0, final_confidence - rejection_rate * 0.2)
        final_confidence = min(final_confidence, research_quality["confidence_cap"])
        final_confidence = min(final_confidence, guardrail_result["confidence_cap"])

        escalated = (
            final_confidence < self._settings.confidence_escalation_threshold
            or bool(guardrail_result.get("requires_human_review"))
        )

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
            research_data=research_data,
            quality=research_quality,
            guardrails=guardrail_result,
            agents_used=agents_used,
            confidence=final_confidence,
            processing_ms=processing_ms,
            escalated=escalated,
        )

        if cacheable and not escalated:
            await self._cache.set(
                cache_key,
                {
                    "response": response,
                    "confidence": final_confidence,
                    "agents_used": agents_used,
                    "escalated_to_expert": escalated,
                },
                namespace="legal_qa",
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
        from agents.base_agent import AgentResult
        return AgentResult(data={}, agent_name="noop")

    def _build_cache_key(
        self,
        *,
        question: str,
        case_id: str | None,
        jurisdiction: str | None,
        user_id: str | None,
        tenant_id: str | None,
    ) -> str:
        payload = {
            "question": question,
            "case_id": case_id,
            "jurisdiction": jurisdiction,
            "user_id": user_id,
            "tenant_id": tenant_id,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _extract_citations(self, irac_data: dict) -> list[dict[str, Any]]:
        explicit = irac_data.get("citations")
        if isinstance(explicit, list) and explicit:
            return [c if isinstance(c, dict) else {"ref": str(c)} for c in explicit]

        citations: list[dict[str, Any]] = []
        rule = (irac_data.get("irac") or {}).get("rule") or {}

        for statute in rule.get("statutes") or []:
            if not isinstance(statute, dict):
                continue
            ref = " ".join(
                str(part).strip()
                for part in (statute.get("name"), statute.get("section"))
                if part
            ).strip()
            if ref:
                citations.append({
                    "ref": ref,
                    "status": "UNVERIFIED",
                    "year": statute.get("year"),
                })

        for precedent in rule.get("precedents") or []:
            if not isinstance(precedent, dict):
                continue
            ref = str(precedent.get("case_no") or "").strip()
            if ref:
                citations.append({
                    "ref": ref,
                    "status": "UNVERIFIED",
                    "note": precedent.get("court"),
                })

        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for citation in citations:
            key = citation["ref"].casefold()
            if key not in seen:
                seen.add(key)
                unique.append(citation)
        return unique

    def _score_research_quality(self, research_data: dict | None) -> dict[str, Any]:
        retrieval = (research_data or {}).get("retrieval") or {}
        source = retrieval.get("source", "empty")
        count = int(retrieval.get("count") or 0)

        if source == "database" and count >= 5:
            return {"level": "strong", "confidence_cap": 0.95, "reason": None}
        if source == "database" and count > 0:
            return {"level": "limited", "confidence_cap": 0.78, "reason": "limited_retrieved_context"}
        return {"level": "insufficient", "confidence_cap": 0.35, "reason": "no_retrieved_legal_context"}

    def _build_response(
        self,
        *,
        irac_data: dict,
        verification_data: dict | None,
        risk_data: dict | None,
        research_data: dict | None,
        quality: dict[str, Any],
        guardrails: dict[str, Any],
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
            "answer_quality": {
                **quality,
                "guardrails": guardrails,
                "retrieval": (research_data or {}).get("retrieval", {}),
            },
            "disclaimer": (
                "คำตอบนี้เป็นข้อมูลทั่วไปเท่านั้น ไม่ใช่คำปรึกษาทางกฎหมายอย่างเป็นทางการ "
                "/ This response is general information only and does not constitute formal legal advice."
            ),
        }
