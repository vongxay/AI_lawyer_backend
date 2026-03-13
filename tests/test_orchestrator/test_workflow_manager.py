"""
Tests for WorkflowManager — the most critical component.

Tests cover:
- Full orchestration cycle (happy path)
- Escalation trigger on low confidence
- PII redaction applied before agents
- Agent result extraction (the TaskGroup bug fix)
- Memory update called after orchestration
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.base_agent import AgentResult
from orchestrator.workflow_manager import WorkflowManager


def _make_research_result() -> AgentResult:
    return AgentResult(
        data={"retrieved_documents": [{"type": "law", "title": "test law", "content": "stub"}],
              "case_graph_context": [], "memory_highlights": {}},
        confidence=0.9,
        agent_name="research",
    )


def _make_reasoning_result(confidence: float = 0.85) -> AgentResult:
    return AgentResult(
        data={
            "irac": {
                "issue": {"primary": "test", "secondary": []},
                "rule": {"statutes": [], "precedents": []},
                "application": {"analysis": "test", "strengths": [], "weaknesses": [], "counter_args": [], "rebuttals": []},
                "conclusion": {"recommendation": "test", "action_steps": [], "risk_level": "MEDIUM", "win_probability": 0.7, "settlement_note": None},
            },
            "confidence": confidence,
            "citations": [{"ref": "มาตรา 420", "status": "UNVERIFIED"}],
        },
        confidence=confidence,
        agent_name="reasoning",
    )


def _make_verification_result(verified: bool = True) -> AgentResult:
    return AgentResult(
        data={
            "citations": [{"ref": "มาตรา 420", "status": "VERIFIED"}],
            "citations_verified": verified,
            "rejection_rate": 0.0,
            "_confidence": 1.0,
        },
        confidence=1.0,
        agent_name="verification",
    )


@pytest.fixture
def workflow() -> WorkflowManager:
    """WorkflowManager with all agents mocked."""
    wf = WorkflowManager(supabase=None, redis=None)

    wf._research_agent.run = AsyncMock(return_value=_make_research_result())
    wf._reasoning_agent.run = AsyncMock(return_value=_make_reasoning_result())
    wf._verification_agent.run = AsyncMock(return_value=_make_verification_result())
    wf._document_agent.run = AsyncMock(return_value=AgentResult(data={"clauses": []}, agent_name="document"))
    wf._evidence_agent.run = AsyncMock(return_value=AgentResult(data={"items": []}, agent_name="evidence"))
    wf._risk_agent.run = AsyncMock(return_value=AgentResult(data={"win_probability": 0.7}, agent_name="risk"))

    wf._case_memory.get = AsyncMock(return_value={"empty": True})
    wf._case_memory.update = AsyncMock()
    wf._audit.log_event = AsyncMock()
    wf._expert_queue.enqueue = AsyncMock()

    return wf


class TestWorkflowManager:
    async def test_happy_path_legal_question(self, workflow):
        result = await workflow.orchestrate(
            question="สัญญาเช่าของฉันถูกยกเลิกโดยไม่แจ้งล่วงหน้า ฉันมีสิทธิ์อะไรบ้าง",
            case_id=None,
        )
        assert result.confidence > 0
        assert "irac" in result.response
        assert "disclaimer" in result.response
        assert "research" in result.agents_used
        assert "reasoning" in result.agents_used
        assert "verification" in result.agents_used

    async def test_case_memory_updated_after_orchestration(self, workflow):
        await workflow.orchestrate(question="test", case_id="case-123")
        workflow._case_memory.update.assert_called_once()
        call_kwargs = workflow._case_memory.update.call_args.kwargs
        assert call_kwargs["case_id"] == "case-123"

    async def test_audit_always_logged(self, workflow):
        await workflow.orchestrate(question="test", case_id=None)
        workflow._audit.log_event.assert_called_once()

    async def test_low_confidence_triggers_escalation(self, workflow):
        workflow._reasoning_agent.run = AsyncMock(
            return_value=_make_reasoning_result(confidence=0.4)
        )
        workflow._verification_agent.run = AsyncMock(
            return_value=_make_verification_result(verified=False)
        )
        result = await workflow.orchestrate(question="test", case_id=None)
        assert result.escalated_to_expert is True
        workflow._expert_queue.enqueue.assert_called_once()

    async def test_pii_redacted_from_question(self, workflow):
        """Thai phone number should not appear in the question sent to agents."""
        phone_question = "ช่วยด้วย เบอร์ 0812345678 โดนโกง"
        await workflow.orchestrate(question=phone_question, case_id=None)

        # Check the question passed to research agent was redacted
        research_call = workflow._research_agent.run.call_args
        clean_question = research_call.kwargs["question"]
        assert "0812345678" not in clean_question
        assert "[REDACTED_PHONE]" in clean_question

    async def test_risk_agent_invoked_for_case_strategy(self, workflow):
        """case_strategy queries should invoke risk agent."""
        result = await workflow.orchestrate(
            question="ควรฟ้องหรือเจรจาดี อยากรู้โอกาสชนะ",  # triggers case_strategy
            case_id=None,
        )
        assert "risk" in result.agents_used

    async def test_document_agent_invoked_when_text_provided(self, workflow):
        result = await workflow.orchestrate(
            question="ตรวจสัญญาให้หน่อย",
            case_id=None,
            document_text="CONTRACT CONTENT: This agreement is made...",
        )
        assert "document" in result.agents_used

    async def test_response_has_required_keys(self, workflow):
        result = await workflow.orchestrate(question="test", case_id=None)
        required_keys = {"irac", "citations", "citations_verified", "confidence",
                         "agents_used", "processing_time_ms", "escalated_to_expert", "disclaimer"}
        assert required_keys.issubset(result.response.keys())

    async def test_processing_time_recorded(self, workflow):
        result = await workflow.orchestrate(question="test", case_id=None)
        assert result.processing_time_ms >= 0

    async def test_session_id_generated(self, workflow):
        result = await workflow.orchestrate(question="test", case_id=None)
        assert result.session_id
        assert len(result.session_id) == 36  # UUID4 format
