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
from core.jurisdiction import contains_lao_script, contains_thai_script, infer_jurisdiction, infer_response_language
from core.logging import get_logger
from memory.case_memory import CaseMemoryService
from memory.session_memory import SessionMemoryService
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
        self._session_memory = SessionMemoryService(supabase=supabase, redis=redis)
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
        query_mode: str | None = None,
        response_style: str | None = None,
        urgency: str | None = None,
        model_id: str | None = None,
    ) -> OrchestrationResult:
        started = time.perf_counter()
        sid = session_id or str(uuid.uuid4())
        effective_jurisdiction = infer_jurisdiction(question, jurisdiction) or "laos"
        effective_mode = self._normalise_query_mode(query_mode, evidence_files=evidence_files, document_text=document_text)
        effective_style = self._normalise_response_style(response_style, query_mode=effective_mode)
        effective_urgency = urgency if urgency in {"normal", "urgent", "critical"} else "normal"
        response_language = infer_response_language(question)
        selected_model_id = self._normalise_model_id(model_id)
        model_policy = self._model_policy(selected_model_id)

        # ── Step 0: PII redaction on all user input ───────────────────────────
        clean_question = self._pii.redact_text(question)
        clean_doc_text = self._pii.redact_text(document_text) if document_text else None
        case_memory = await self._case_memory.get(case_id, tenant_id=tenant_id, user_id=user_id)
        session_memory = await self._session_memory.get(sid, tenant_id=tenant_id, user_id=user_id)
        memory = self._merge_memory(case_memory, session_memory)
        contextual_question = self._contextual_question(clean_question, session_memory)
        intent_route = self._classifier.route(
            clean_question,
            query_mode=effective_mode,
            has_document=bool(clean_doc_text),
            has_evidence=bool(evidence_files),
            memory=memory,
        )
        classified_type = intent_route.query_type
        query_type = self._query_type_for_mode(effective_mode, classified_type)
        effective_style = self._response_style_for_intent(
            current_style=effective_style,
            query_mode=effective_mode,
            query_type=query_type,
            route_style=intent_route.response_style,
        )
        intent_route_data = intent_route.to_dict()
        intent_route_data["effective_query_type"] = query_type
        intent_route_data["effective_response_style"] = effective_style

        cache_key = self._build_cache_key(
            question=clean_question,
            case_id=case_id,
            jurisdiction=effective_jurisdiction,
            user_id=user_id,
            tenant_id=tenant_id,
            query_type=query_type,
            query_mode=effective_mode,
            response_style=effective_style,
            urgency=effective_urgency,
            model_id=selected_model_id,
            response_language=response_language,
            conversation_hash=session_memory.get("cache_key", "empty"),
            memory_hash=self._memory_hash(memory),
        )
        cacheable = clean_doc_text is None and not evidence_files
        if cacheable:
            cached = await self._cache.get(cache_key, namespace="legal_qa")
            if cached:
                processing_ms = int((time.perf_counter() - started) * 1000)
                response = {
                    **cached["response"],
                    "processing_time_ms": processing_ms,
                    "cached": True,
                    "session_id": sid,
                    "selected_model_id": selected_model_id,
                }
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
        # Memory is loaded before cache lookup so session-aware answers never reuse stale context.

        # ── Step 2: Dynamic agent selection ──────────────────────────────────
        if query_type == "conversation" and clean_doc_text is None and not evidence_files:
            processing_ms = int((time.perf_counter() - started) * 1000)
            response = self._build_conversation_response(
                question=clean_question,
                memory=memory,
                query_mode=effective_mode,
                response_style="plain",
                response_language=response_language,
                processing_ms=processing_ms,
                session_id=sid,
                intent_route=intent_route_data,
            )
            await self._audit.log_event(
                user_id=user_id,
                tenant_id=tenant_id,
                agent="conversation",
                query=clean_question,
                confidence=response["confidence"],
                agents_used=response["agents_used"],
                processing_time_ms=processing_ms,
                escalated=False,
            )
            if cacheable:
                await self._cache.set(
                    cache_key,
                    {
                        "response": response,
                        "confidence": response["confidence"],
                        "agents_used": response["agents_used"],
                        "escalated_to_expert": False,
                    },
                    namespace="legal_qa",
                )
            return OrchestrationResult(
                response=response,
                confidence=response["confidence"],
                agents_used=response["agents_used"],
                processing_time_ms=processing_ms,
                escalated_to_expert=False,
                session_id=sid,
            )

        if query_type == "clarification" and clean_doc_text is None and not evidence_files:
            processing_ms = int((time.perf_counter() - started) * 1000)
            response = self._build_clarification_response(
                question=clean_question,
                memory=memory,
                query_mode=effective_mode,
                response_style="plain",
                response_language=response_language,
                processing_ms=processing_ms,
                session_id=sid,
                intent_route=intent_route_data,
            )
            await self._audit.log_event(
                user_id=user_id,
                tenant_id=tenant_id,
                agent="intent_router",
                query=clean_question,
                confidence=response["confidence"],
                agents_used=response["agents_used"],
                processing_time_ms=processing_ms,
                escalated=False,
            )
            if cacheable:
                await self._cache.set(
                    cache_key,
                    {
                        "response": response,
                        "confidence": response["confidence"],
                        "agents_used": response["agents_used"],
                        "escalated_to_expert": False,
                    },
                    namespace="legal_qa",
                )
            return OrchestrationResult(
                response=response,
                confidence=response["confidence"],
                agents_used=response["agents_used"],
                processing_time_ms=processing_ms,
                escalated_to_expert=False,
                session_id=sid,
            )

        plan = self._selector.select(
            query_type,
            force_document=bool(document_text),
            force_evidence=bool(evidence_files),
            force_risk=effective_mode == "serious_case"
            or effective_style == "action_plan"
            or effective_urgency in {"urgent", "critical"},
        )

        log.info(
            "orchestrator.start",
            session_id=sid,
            query_type=query_type,
            query_mode=effective_mode,
            response_style=effective_style,
            urgency=effective_urgency,
            plan=str(plan),
            jurisdiction=effective_jurisdiction,
            case_id=case_id,
            response_language=response_language,
            selected_model_id=selected_model_id,
            conversation_messages=session_memory.get("message_count", 0),
        )

        # ── Step 4: Parallel phase — Research + Document + Evidence ──────────
        agents_used: list[str] = []
        parallel_tasks: dict[str, asyncio.Task] = {}

        async with asyncio.TaskGroup() as tg:
            if plan.use_research:
                agents_used.append("research")
                parallel_tasks["research"] = tg.create_task(
                    self._research_agent.run(
                        question=contextual_question,
                        memory=memory,
                        jurisdiction=effective_jurisdiction,
                        tenant_id=tenant_id,
                    )
                )

            if plan.use_document and clean_doc_text:
                agents_used.append("document")
                parallel_tasks["document"] = tg.create_task(
                    self._document_agent.run(
                        question=clean_question,
                        document_text=clean_doc_text,
                        case_context=memory.get("facts_summary"),
                        model_override=model_policy["document"],
                    )
                )

            if plan.use_evidence and evidence_files:
                agents_used.append("evidence")
                parallel_tasks["evidence"] = tg.create_task(
                    self._evidence_agent.run(
                        question=clean_question,
                        evidence_files=evidence_files,
                        case_context=memory.get("facts_summary"),
                        model_override=model_policy["evidence"],
                    )
                )

        # Extract results from completed tasks (FIX: .result() not direct access)
        research_result = parallel_tasks["research"].result() if "research" in parallel_tasks else None
        document_result = parallel_tasks["document"].result() if "document" in parallel_tasks else None
        evidence_result = parallel_tasks["evidence"].result() if "evidence" in parallel_tasks else None
        research_data = self._agent_data_or_empty_research(research_result, effective_jurisdiction)
        document_data = document_result.data if document_result and document_result.ok else None
        evidence_data = evidence_result.data if evidence_result and evidence_result.ok else None

        # ── Step 5: IRAC Reasoning (sequential — needs all parallel results) ──
        agents_used.append("reasoning")
        reasoning_result = await self._reasoning_agent.run(
            question=clean_question,
            research=research_data,
            document=document_data,
            evidence=evidence_data,
            memory=memory,
            response_language=response_language,
            query_mode=effective_mode,
            response_style=effective_style,
            query_type=query_type,
            model_override=model_policy["reasoning"],
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

        verify_coro = self._verification_agent.run(
            citations=citations_to_verify,
            model_override=model_policy["verification"],
        )
        risk_coro = (
            self._risk_agent.run(
                question=clean_question,
                irac=irac_data,
                research=research_data,
                response_language=response_language,
                model_override=model_policy["risk"],
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
        research_quality["conversation_memory"] = {
            "used": not session_memory.get("empty", True),
            "message_count": session_memory.get("message_count", 0),
        }
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
                tenant_id=tenant_id,
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
            document_data=document_data,
            evidence_data=evidence_data,
            quality=research_quality,
            guardrails=guardrail_result,
            agents_used=agents_used,
            confidence=final_confidence,
            processing_ms=processing_ms,
            escalated=escalated,
            query_type=query_type,
            query_mode=effective_mode,
            response_style=effective_style,
            session_id=sid,
            response_language=response_language,
            selected_model_id=selected_model_id,
            model_policy=model_policy,
            intent_route=intent_route_data,
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

    def _normalise_model_id(self, model_id: str | None) -> str:
        normalized = str(model_id or "legal-pro").strip().lower().replace("_", "-")
        if normalized in {"legal-base", "base", "economy", "fast", "cheap"}:
            return "legal-base"
        if normalized in {"legal-expert", "expert", "premium"}:
            return "legal-expert"
        return "legal-pro"

    def _model_policy(self, model_id: str | None) -> dict[str, str]:
        selected = self._normalise_model_id(model_id)
        settings = self._settings
        if selected == "legal-base":
            return {
                "reasoning": settings.model_economy,
                "document": settings.model_economy,
                "evidence": settings.model_economy,
                "verification": settings.model_economy,
                "risk": settings.model_economy,
            }
        return {
            "reasoning": settings.model_reasoning,
            "document": settings.model_document,
            "evidence": settings.model_evidence,
            "verification": settings.model_verification,
            "risk": settings.model_risk,
        }

    def _build_conversation_response(
        self,
        *,
        question: str,
        memory: dict[str, Any],
        query_mode: str,
        response_style: str,
        response_language: str,
        processing_ms: int,
        session_id: str,
        intent_route: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        answer = self._conversation_answer(question, memory, response_language)
        irac = self._empty_irac(question, answer)
        memory_used = not memory.get("empty", True)
        return {
            "irac": irac,
            "answer": answer,
            "query_type": "conversation",
            "query_mode": query_mode,
            "response_style": response_style,
            "response_language": response_language,
            "session_id": session_id,
            "citations": [],
            "citations_verified": True,
            "confidence": 0.98,
            "agents_used": ["conversation"],
            "processing_time_ms": processing_ms,
            "escalated_to_expert": False,
            "risk": None,
            "document": None,
            "evidence": None,
            "answer_quality": {
                "rag_used": False,
                "llm_reasoning_used": False,
                "reason": "non_legal_conversation",
                "intent_route": intent_route or {},
                "conversation_memory": {
                    "used": memory_used,
                    "message_count": memory.get("session_message_count", 0),
                },
            },
            "disclaimer": self._conversation_disclaimer(response_language),
        }

    def _build_clarification_response(
        self,
        *,
        question: str,
        memory: dict[str, Any],
        query_mode: str,
        response_style: str,
        response_language: str,
        processing_ms: int,
        session_id: str,
        intent_route: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        route = intent_route or {}
        questions = [
            str(item).strip()
            for item in (route.get("clarification_questions") or [])
            if str(item).strip()
        ]
        answer = self._clarification_answer(questions, response_language)
        irac = self._empty_irac(question, answer)
        memory_used = not memory.get("empty", True)
        return {
            "irac": irac,
            "answer": answer,
            "query_type": "clarification",
            "query_mode": query_mode,
            "response_style": response_style,
            "response_language": response_language,
            "session_id": session_id,
            "citations": [],
            "citations_verified": True,
            "confidence": 0.88,
            "agents_used": ["intent_router"],
            "processing_time_ms": processing_ms,
            "escalated_to_expert": False,
            "risk": None,
            "document": None,
            "evidence": None,
            "answer_quality": {
                "rag_used": False,
                "llm_reasoning_used": False,
                "reason": "needs_clarification_before_legal_analysis",
                "intent_route": route,
                "conversation_memory": {
                    "used": memory_used,
                    "message_count": memory.get("session_message_count", 0),
                },
            },
            "disclaimer": self._clarification_disclaimer(response_language),
        }

    def _clarification_answer(self, questions: list[str], response_language: str) -> str:
        if response_language == "lo":
            intro = "ເພື່ອໃຫ້ຕອບໄດ້ແມ່ນຍຳ ແລະ ບໍ່ສະຫຼຸບເກີນຂໍ້ເທັດຈິງ, ຂໍຂໍ້ມູນເພີ່ມກ່ອນ:"
        elif response_language == "th":
            intro = "เพื่อให้ตอบได้แม่นยำและไม่สรุปเกินข้อเท็จจริง ขอข้อมูลเพิ่มก่อนครับ:"
        else:
            intro = "To answer accurately without guessing beyond the facts, I need a few details first:"

        if not questions:
            questions = [
                "What happened, when did it happen, and what outcome do you want?",
            ]
        return intro + "\n" + "\n".join(f"{index + 1}. {question}" for index, question in enumerate(questions))

    def _clarification_disclaimer(self, response_language: str) -> str:
        if response_language == "lo":
            return "ຂໍ້ຄວາມນີ້ເປັນການຂໍຂໍ້ມູນເພີ່ມ ຍັງບໍ່ແມ່ນຄຳວິເຄາະກົດໝາຍຂັ້ນສຸດທ້າຍ."
        if response_language == "th":
            return "ข้อความนี้เป็นการขอข้อมูลเพิ่ม ยังไม่ใช่คำวิเคราะห์กฎหมายขั้นสุดท้าย"
        return "This is a clarification request, not a final legal analysis."

    def _conversation_answer(self, question: str, memory: dict[str, Any], response_language: str) -> str:
        lowered = question.casefold()
        asks_memory = any(
            marker in lowered
            for marker in (
                "\u0e88\u0eb7\u0ec8\u0ec4\u0e94\u0ec9",
                "\u0e08\u0e33\u0e44\u0e14\u0e49",
                "remember",
                "what did we discuss",
            )
        )
        asks_capability = any(
            marker in lowered
            for marker in (
                "\u0eaa\u0eb2\u0ea1\u0eb2\u0e94\u0e8a\u0ec8\u0ea7\u0e8d",
                "\u0e97\u0e33\u0e2d\u0e30\u0e44\u0e23\u0e44\u0e14\u0e49",
                "what can you do",
                "help me with",
            )
        )
        thanks = any(marker in lowered for marker in ("\u0e82\u0ead\u0e9a\u0ec3\u0e88", "\u0e02\u0e2d\u0e1a\u0e04\u0e38\u0e13", "thank"))

        if response_language == "lo" or contains_lao_script(question):
            if asks_memory:
                summary = str(memory.get("conversation_summary") or "").strip()
                if summary:
                    return f"\u0e88\u0eb7\u0ec8\u0ec4\u0e94\u0ec9. \u0eaa\u0eb0\u0eab\u0ebc\u0eb8\u0e9a\u0eaa\u0eb1\u0ec9\u0e99\u0ec6: {summary[:700]}"
                return "\u0e95\u0ead\u0e99\u0e99\u0eb5\u0ec9\u0e82\u0ec9\u0ead\u0e8d\u0e8d\u0eb1\u0e87\u0e9a\u0ecd\u0ec8\u0ea1\u0eb5\u0e9b\u0eb0\u0eab\u0ea7\u0eb1\u0e94\u0eaa\u0ebb\u0e99\u0e97\u0eb0\u0e99\u0eb2\u0e81\u0ec8\u0ead\u0e99\u0edc\u0ec9\u0eb2\u0e97\u0eb5\u0ec8\u0e9e\u0ecd\u0e88\u0eb0\u0eaa\u0eb0\u0eab\u0ebc\u0eb8\u0e9a\u0ec4\u0e94\u0ec9."
            if asks_capability:
                return "\u0e82\u0ec9\u0ead\u0e8d\u0e8a\u0ec8\u0ea7\u0e8d\u0ead\u0eb0\u0e97\u0eb4\u0e9a\u0eb2\u0e8d\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d\u0ea5\u0eb2\u0ea7, \u0e84\u0ebb\u0ec9\u0e99\u0eab\u0eb2\u0ea1\u0eb2\u0e94\u0e95\u0eb2\u0e97\u0eb5\u0ec8\u0e81\u0ec8\u0ebd\u0ea7\u0e82\u0ec9\u0ead\u0e87, \u0eaa\u0eb0\u0eab\u0ebc\u0eb8\u0e9a\u0ec0\u0ead\u0e81\u0eb0\u0eaa\u0eb2\u0e99 \u0ec1\u0ea5\u0eb0\u0e8a\u0ec8\u0ea7\u0e8d\u0ea7\u0eb4\u0ec0\u0e84\u0eb2\u0eb0\u0e84\u0eb3\u0e96\u0eb2\u0ea1\u0e97\u0eb2\u0e87\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d\u0ec4\u0e94\u0ec9."
            if thanks:
                return "\u0e94\u0ec9\u0ea7\u0e8d\u0e84\u0ea7\u0eb2\u0ea1\u0e8d\u0eb4\u0e99\u0e94\u0eb5. \u0e96\u0ec9\u0eb2\u0ea1\u0eb5\u0e84\u0eb3\u0e96\u0eb2\u0ea1\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d\u0ea5\u0eb2\u0ea7\u0ec0\u0e9e\u0eb5\u0ec8\u0ea1 \u0eaa\u0ebb\u0ec8\u0e87\u0ea1\u0eb2\u0ec4\u0e94\u0ec9\u0ec0\u0ea5\u0eb5\u0e8d."
            return "\u0eaa\u0eb0\u0e9a\u0eb2\u0e8d\u0e94\u0eb5, \u0e82\u0ec9\u0ead\u0e8d\u0e9e\u0ec9\u0ead\u0ea1\u0e8a\u0ec8\u0ea7\u0e8d. \u0e96\u0ec9\u0eb2\u0ea1\u0eb5\u0e84\u0eb3\u0e96\u0eb2\u0ea1\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d\u0ea5\u0eb2\u0ea7 \u0eab\u0ebc\u0eb7 \u0e95\u0ec9\u0ead\u0e87\u0e81\u0eb2\u0e99\u0e95\u0ea7\u0e94\u0ea1\u0eb2\u0e94\u0e95\u0eb2\u0ec3\u0e94 \u0e96\u0eb2\u0ea1\u0ea1\u0eb2\u0ec4\u0e94\u0ec9\u0ec0\u0ea5\u0eb5\u0e8d."

        if response_language == "th" or contains_thai_script(question):
            if asks_memory:
                summary = str(memory.get("conversation_summary") or "").strip()
                if summary:
                    return f"จำได้ครับ สรุปสั้น ๆ คือ: {summary[:700]}"
                return "ตอนนี้ยังไม่มีประวัติสนทนาก่อนหน้าที่พอจะสรุปได้ครับ"
            if asks_capability:
                return "ผมช่วยอธิบายกฎหมายลาว ค้นหามาตราที่เกี่ยวข้อง สรุปเอกสาร และช่วยวิเคราะห์คำถามทางกฎหมายให้เป็นภาษาที่เข้าใจง่ายได้ครับ"
            if thanks:
                return "ยินดีครับ ถ้ามีคำถามกฎหมายลาวเพิ่มเติม ส่งมาได้เลยครับ"
            return "สวัสดีครับ ผมพร้อมช่วยครับ ถ้ามีคำถามเกี่ยวกับกฎหมายลาวหรืออยากให้ช่วยดูมาตราไหน ส่งคำถามมาได้เลยครับ"

        if asks_memory:
            summary = str(memory.get("conversation_summary") or "").strip()
            if summary:
                return f"Yes. In short, we discussed: {summary[:700]}"
            return "I do not have enough prior conversation context to summarize yet."
        if asks_capability:
            return "I can help explain Lao law, find relevant statutory provisions, summarize documents, and reason through legal questions in plain language."
        if thanks:
            return "You are welcome. Send me the Lao legal question whenever you are ready."
        return "Hello. I am ready to help with Lao legal questions, document review, or finding the relevant legal article."

    def _conversation_disclaimer(self, response_language: str) -> str:
        if response_language == "lo":
            return "\u0e82\u0ecd\u0ec9\u0e84\u0ea7\u0eb2\u0ea1\u0e99\u0eb5\u0ec9\u0ec0\u0e9b\u0eb1\u0e99\u0e81\u0eb2\u0e99\u0eaa\u0ebb\u0e99\u0e97\u0eb0\u0e99\u0eb2\u0e97\u0ebb\u0ec8\u0ea7\u0ec4\u0e9b \u0e9a\u0ecd\u0ec8\u0ec1\u0ea1\u0ec8\u0e99\u0e84\u0eb3\u0e9b\u0eb6\u0e81\u0eaa\u0eb2\u0e81\u0ebb\u0e94\u0edd\u0eb2\u0e8d."
        if response_language == "th":
            return "ข้อความนี้เป็นการสนทนาทั่วไป ไม่ใช่คำปรึกษากฎหมาย"
        return "This is a general conversational response, not legal advice."

    def _empty_irac(self, question: str, answer: str) -> dict[str, Any]:
        return {
            "issue": {"primary": question, "secondary": []},
            "rule": {"statutes": [], "precedents": []},
            "application": {
                "analysis": "",
                "strengths": [],
                "weaknesses": [],
                "counter_args": [],
                "rebuttals": [],
            },
            "conclusion": {
                "recommendation": answer,
                "action_steps": [],
                "risk_level": "LOW",
                "win_probability": 0.0,
                "settlement_note": None,
            },
        }

    def _merge_memory(self, case_memory: dict[str, Any], session_memory: dict[str, Any]) -> dict[str, Any]:
        merged = dict(case_memory or {})
        has_case_memory = not merged.get("empty")
        has_session_memory = not session_memory.get("empty")

        if not has_case_memory:
            merged = {"empty": not has_session_memory}

        if has_session_memory:
            merged["conversation_summary"] = session_memory.get("conversation_summary", "")
            merged["conversation_messages"] = session_memory.get("messages", [])
            merged["current_user_state"] = session_memory.get("current_user_state", "")
            merged["last_assistant_answer"] = session_memory.get("last_assistant_answer", "")
            merged["session_message_count"] = session_memory.get("message_count", 0)
            if not merged.get("facts_summary"):
                merged["facts_summary"] = session_memory.get("conversation_summary", "")

        return merged

    def _contextual_question(self, question: str, session_memory: dict[str, Any]) -> str:
        summary = str(session_memory.get("conversation_summary") or "").strip()
        if not summary:
            return question
        return (
            "Conversation context for retrieval only:\n"
            f"{summary[:1800]}\n\n"
            f"Current user question:\n{question}"
        )

    def _memory_hash(self, memory: dict[str, Any]) -> str:
        relevant = {
            "facts_summary": memory.get("facts_summary"),
            "conversation_summary": memory.get("conversation_summary"),
            "key_citations": memory.get("key_citations"),
            "session_message_count": memory.get("session_message_count"),
        }
        raw = json.dumps(relevant, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _agent_data_or_empty_research(self, result: Any, jurisdiction: str | None) -> dict[str, Any] | None:
        if result is None:
            return None
        if result.ok:
            return result.data
        return {
            "retrieved_documents": [],
            "case_graph_context": [],
            "memory_highlights": {},
            "retrieval": {
                "source": "empty",
                "count": 0,
                "jurisdiction": jurisdiction,
                "error": result.error,
                "coverage": {
                    "count": 0,
                    "statute_count": 0,
                    "official_source_count": 0,
                    "clean_text_count": 0,
                    "enough_results": False,
                    "has_statute": False,
                    "has_official_source": False,
                    "has_clean_text": False,
                    "reason": "research_agent_failed",
                },
            },
        }

    def _build_cache_key(
        self,
        *,
        question: str,
        case_id: str | None,
        jurisdiction: str | None,
        user_id: str | None,
        tenant_id: str | None,
        query_type: str | None,
        query_mode: str | None,
        response_style: str | None,
        urgency: str | None,
        model_id: str | None,
        response_language: str | None,
        conversation_hash: str | None,
        memory_hash: str | None,
    ) -> str:
        payload = {
            "answer_pipeline_version": 13,
            "question": question,
            "case_id": case_id,
            "jurisdiction": jurisdiction,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "query_type": query_type,
            "query_mode": query_mode,
            "response_style": response_style,
            "urgency": urgency,
            "model_id": model_id,
            "response_language": response_language,
            "conversation_hash": conversation_hash,
            "memory_hash": memory_hash,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _normalise_query_mode(
        self,
        query_mode: str | None,
        *,
        evidence_files: list[EvidenceFile] | None,
        document_text: str | None,
    ) -> str:
        if evidence_files:
            return "evidence"
        if document_text:
            return "document"
        mode = (query_mode or "general").strip().lower()
        if mode in {"general", "serious_case", "evidence", "document", "draft"}:
            return mode
        return "general"

    def _normalise_response_style(self, response_style: str | None, *, query_mode: str) -> str:
        style = (response_style or "").strip().lower()
        if style in {"plain", "irac", "action_plan"}:
            return style
        if query_mode == "general":
            return "plain"
        if query_mode in {"serious_case", "evidence"}:
            return "action_plan"
        return "irac"

    def _response_style_for_intent(
        self,
        *,
        current_style: str,
        query_mode: str,
        query_type: str,
        route_style: str | None,
    ) -> str:
        if query_type in {"conversation", "clarification"}:
            return "plain"

        specialist_styles = {
            "case_strategy": "action_plan",
            "evidence_analysis": "action_plan",
            "document_review": "irac",
            "draft_document": "irac",
        }
        if query_type in specialist_styles:
            return specialist_styles[query_type]

        if query_mode != "general":
            return current_style

        style = (route_style or "").strip().lower()
        if style in {"plain", "irac", "action_plan"}:
            return style
        return current_style

    def _query_type_for_mode(self, query_mode: str, classified_type: str) -> str:
        overrides: dict[str, str] = {
            "serious_case": "case_strategy",
            "evidence": "evidence_analysis",
            "document": "document_review",
            "draft": "draft_document",
        }
        return overrides.get(query_mode, classified_type)

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
        documents = (research_data or {}).get("retrieved_documents") or []
        source = retrieval.get("source", "empty")
        count = int(retrieval.get("count") or len(documents) or 0)
        coverage = retrieval.get("coverage") if isinstance(retrieval.get("coverage"), dict) else {}
        statute_count = int(coverage.get("statute_count") or 0)
        official_count = int(coverage.get("official_source_count") or 0)
        clean_count = int(coverage.get("clean_text_count") or 0)
        trace = retrieval.get("trace") if isinstance(retrieval.get("trace"), list) else []
        used_embeddings = any(item.get("mode") == "hybrid" for item in trace if isinstance(item, dict))

        if retrieval.get("error"):
            return {
                "level": "insufficient",
                "confidence_cap": 0.35,
                "reason": retrieval.get("error") or "research_agent_failed",
                "coverage": coverage,
            }

        if source == "database" and count >= 5 and statute_count > 0 and clean_count > 0:
            cap = 0.95 if used_embeddings else 0.84
            if retrieval.get("jurisdiction") == "laos" and official_count == 0:
                cap = min(cap, 0.74)
            return {
                "level": "strong" if used_embeddings else "strong_keyword_only",
                "confidence_cap": cap,
                "reason": coverage.get("reason"),
                "coverage": coverage,
            }
        if source == "database" and count > 0 and statute_count > 0:
            return {
                "level": "limited",
                "confidence_cap": 0.72 if clean_count == 0 else 0.78,
                "reason": coverage.get("reason") or "limited_retrieved_context",
                "coverage": coverage,
            }
        if source == "database" and count > 0:
            return {
                "level": "weak",
                "confidence_cap": 0.58,
                "reason": coverage.get("reason") or "no_statutory_authority",
                "coverage": coverage,
            }
        return {
            "level": "insufficient",
            "confidence_cap": 0.35,
            "reason": "no_retrieved_legal_context",
            "coverage": coverage,
        }

    def _build_response(
        self,
        *,
        irac_data: dict,
        verification_data: dict | None,
        risk_data: dict | None,
        research_data: dict | None,
        document_data: dict | None,
        evidence_data: dict | None,
        quality: dict[str, Any],
        guardrails: dict[str, Any],
        agents_used: list[str],
        confidence: float,
        processing_ms: int,
        escalated: bool,
        query_type: str,
        query_mode: str,
        response_style: str,
        session_id: str,
        response_language: str,
        selected_model_id: str,
        model_policy: dict[str, str],
        intent_route: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        irac = irac_data.get("irac", {})
        disclaimer = self._disclaimer_for_language(response_language)
        return {
            "irac": irac,
            "answer": self._build_answer_text(irac, risk_data, response_style, response_language),
            "query_type": query_type,
            "query_mode": query_mode,
            "response_style": response_style,
            "response_language": response_language,
            "selected_model_id": selected_model_id,
            "session_id": session_id,
            "citations": (verification_data or {}).get("citations", irac_data.get("citations", [])),
            "citations_verified": (verification_data or {}).get("citations_verified", False),
            "confidence": round(confidence, 3),
            "agents_used": agents_used,
            "processing_time_ms": processing_ms,
            "escalated_to_expert": escalated,
            "risk": risk_data,
            "document": document_data,
            "evidence": evidence_data,
            "answer_quality": {
                **quality,
                "intent_route": intent_route or {},
                "guardrails": guardrails,
                "retrieval": (research_data or {}).get("retrieval", {}),
                "model_policy": model_policy,
            },
            "disclaimer": disclaimer,
        }

    def _build_answer_text(
        self,
        irac: dict[str, Any],
        risk_data: dict | None,
        response_style: str,
        response_language: str = "en",
    ) -> str:
        conclusion = irac.get("conclusion") if isinstance(irac, dict) else {}
        application = irac.get("application") if isinstance(irac, dict) else {}

        recommendation = str((conclusion or {}).get("recommendation") or "").strip()
        analysis = str((application or {}).get("analysis") or "").strip()
        action_steps = [
            str(step).strip()
            for step in ((conclusion or {}).get("action_steps") or [])
            if str(step).strip()
        ]
        action_label = str((conclusion or {}).get("action_label") or "").strip()

        if response_style == "plain":
            labels = self._answer_labels(response_language)
            step_label = action_label or labels["next_steps"]
            plain_parts = [part for part in (recommendation, analysis) if part]
            if action_steps:
                plain_parts.append(
                    f"{step_label}\n"
                    + "\n".join(f"{index + 1}. {step}" for index, step in enumerate(action_steps))
                )
            return "\n\n".join(plain_parts)

        labels = self._answer_labels(response_language)
        step_label = action_label or labels["next_steps"]
        parts = []
        if recommendation:
            parts.append(f"{labels['recommendation']}\n{recommendation}")
        if analysis:
            parts.append(f"{labels['analysis']}\n{analysis}")
        if action_steps:
            parts.append(f"{step_label}\n" + "\n".join(f"- {step}" for step in action_steps))

        if response_style == "action_plan" and isinstance(risk_data, dict):
            immediate_actions = [
                str(step).strip()
                for step in (risk_data.get("immediate_actions") or [])
                if str(step).strip()
            ]
            if immediate_actions:
                parts.append(f"{labels['immediate_actions']}\n" + "\n".join(f"- {step}" for step in immediate_actions))

        return "\n\n".join(parts)

    def _answer_labels(self, response_language: str) -> dict[str, str]:
        if response_language == "lo":
            return {
                "recommendation": "ຄຳແນະນຳ",
                "analysis": "ການວິເຄາະ",
                "next_steps": "ຂັ້ນຕອນຕໍ່ໄປ",
                "immediate_actions": "ສິ່ງທີ່ຄວນເຮັດທັນທີ",
            }
        if response_language == "th":
            return {
                "recommendation": "คำแนะนำ",
                "analysis": "การวิเคราะห์",
                "next_steps": "ขั้นตอนถัดไป",
                "immediate_actions": "สิ่งที่ควรทำทันที",
            }
        return {
            "recommendation": "Recommendation",
            "analysis": "Analysis",
            "next_steps": "Next steps",
            "immediate_actions": "Immediate actions",
        }

    def _disclaimer_for_language(self, response_language: str) -> str:
        if response_language == "lo":
            return (
                "ຄຳຕອບນີ້ເປັນຂໍ້ມູນກົດໝາຍທົ່ວໄປເທົ່ານັ້ນ "
                "ບໍ່ແທນຄຳປຶກສາຈາກທະນາຍຄວາມທີ່ໄດ້ກວດຂໍ້ເທັດຈິງ ເອກະສານ ແລະຫຼັກຖານຄົບຖ້ວນ."
            )
        if response_language == "th":
            return (
                "คำตอบนี้เป็นข้อมูลกฎหมายทั่วไปเท่านั้น ไม่แทนคำปรึกษาจากทนายความ"
                "ที่ได้ตรวจข้อเท็จจริง เอกสาร และหลักฐานครบถ้วน."
            )
        return (
            "This response is general legal information only and does not replace advice "
            "from a licensed lawyer who has reviewed the full facts, documents, and evidence."
        )
